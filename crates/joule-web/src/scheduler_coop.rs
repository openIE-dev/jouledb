//! Cooperative scheduler — time-sliced task execution with yield points.
//!
//! Models cooperative multitasking in pure Rust: tasks yield voluntarily,
//! the scheduler uses priority-based fair scheduling with starvation
//! prevention, cancellation, progress reporting, and deadline support.

use std::collections::{HashMap, VecDeque};

// ── Task State ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopTaskState {
    Ready,
    Running,
    Yielded,
    Completed,
    Cancelled,
}

// ── Priority ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoopPriority(pub u32);

impl CoopPriority {
    pub const HIGH: CoopPriority = CoopPriority(0);
    pub const NORMAL: CoopPriority = CoopPriority(1);
    pub const LOW: CoopPriority = CoopPriority(2);
}

impl Default for CoopPriority {
    fn default() -> Self {
        CoopPriority::NORMAL
    }
}

// ── CoopTask ───────────────────────────────────────────────────

/// A cooperative task managed by the scheduler.
#[derive(Debug, Clone)]
pub struct CoopTask {
    pub id: u64,
    pub name: String,
    pub state: CoopTaskState,
    pub priority: CoopPriority,
    /// Total work units this task needs to complete.
    pub total_work: u64,
    /// Work units completed so far.
    pub work_done: u64,
    /// CPU time consumed (simulated ms).
    pub cpu_time_ms: u64,
    /// Optional deadline (simulated ms from epoch).
    pub deadline_ms: Option<u64>,
    /// How many times this task was skipped (for starvation detection).
    pub skip_count: u64,
}

impl CoopTask {
    pub fn new(id: u64, name: impl Into<String>, total_work: u64) -> Self {
        Self {
            id,
            name: name.into(),
            state: CoopTaskState::Ready,
            priority: CoopPriority::default(),
            total_work,
            work_done: 0,
            cpu_time_ms: 0,
            deadline_ms: None,
            skip_count: 0,
        }
    }

    pub fn with_priority(mut self, p: CoopPriority) -> Self {
        self.priority = p;
        self
    }

    pub fn with_deadline(mut self, deadline_ms: u64) -> Self {
        self.deadline_ms = Some(deadline_ms);
        self
    }

    pub fn progress(&self) -> f64 {
        if self.total_work == 0 {
            return 1.0;
        }
        self.work_done as f64 / self.total_work as f64
    }

    pub fn is_finished(&self) -> bool {
        self.work_done >= self.total_work
    }
}

// ── Scheduler ──────────────────────────────────────────────────

/// Cooperative scheduler with time-sliced execution.
#[derive(Debug)]
pub struct CoopScheduler {
    tasks: HashMap<u64, CoopTask>,
    /// Ready queues per priority level.
    ready_queues: HashMap<CoopPriority, VecDeque<u64>>,
    /// Time slice in ms for each task execution.
    time_slice_ms: u64,
    /// Work units per ms of execution.
    work_per_ms: u64,
    /// Starvation threshold — boost priority after this many skips.
    starvation_threshold: u64,
    /// Simulated current time.
    current_time_ms: u64,
    /// Log of task ids executed each slice.
    schedule_log: Vec<u64>,
    next_id: u64,
}

impl CoopScheduler {
    pub fn new(time_slice_ms: u64) -> Self {
        Self {
            tasks: HashMap::new(),
            ready_queues: HashMap::new(),
            time_slice_ms,
            work_per_ms: 1,
            starvation_threshold: 5,
            current_time_ms: 0,
            schedule_log: Vec::new(),
            next_id: 1,
        }
    }

    pub fn with_work_per_ms(mut self, w: u64) -> Self {
        self.work_per_ms = w.max(1);
        self
    }

    pub fn with_starvation_threshold(mut self, t: u64) -> Self {
        self.starvation_threshold = t;
        self
    }

    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn current_time(&self) -> u64 {
        self.current_time_ms
    }

    pub fn schedule_log(&self) -> &[u64] {
        &self.schedule_log
    }

    /// Add a task to the scheduler.
    pub fn add_task(&mut self, task: CoopTask) -> u64 {
        let id = task.id;
        let prio = task.priority;
        self.tasks.insert(id, task);
        self.ready_queues.entry(prio).or_default().push_back(id);
        id
    }

    /// Cancel a task by id.
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(task) = self.tasks.get_mut(&id) {
            if task.state == CoopTaskState::Completed || task.state == CoopTaskState::Cancelled {
                return false;
            }
            task.state = CoopTaskState::Cancelled;
            // Remove from ready queues.
            for q in self.ready_queues.values_mut() {
                q.retain(|tid| *tid != id);
            }
            true
        } else {
            false
        }
    }

    /// Get a task by id.
    pub fn get_task(&self, id: u64) -> Option<&CoopTask> {
        self.tasks.get(&id)
    }

    /// Number of tasks in a given state.
    pub fn count_in_state(&self, state: CoopTaskState) -> usize {
        self.tasks.values().filter(|t| t.state == state).count()
    }

    /// Check for starvation and boost priorities.
    fn anti_starvation(&mut self) {
        let threshold = self.starvation_threshold;
        let mut boosts = Vec::new();
        for task in self.tasks.values() {
            if task.skip_count >= threshold
                && task.priority != CoopPriority::HIGH
                && (task.state == CoopTaskState::Ready || task.state == CoopTaskState::Yielded)
            {
                boosts.push(task.id);
            }
        }
        for id in boosts {
            if let Some(task) = self.tasks.get_mut(&id) {
                let old_prio = task.priority;
                let new_prio = CoopPriority(old_prio.0.saturating_sub(1));
                task.priority = new_prio;
                task.skip_count = 0;
                // Move between queues.
                if let Some(q) = self.ready_queues.get_mut(&old_prio) {
                    q.retain(|tid| *tid != id);
                }
                self.ready_queues.entry(new_prio).or_default().push_back(id);
            }
        }
    }

    /// Select the next task to run (highest priority, earliest in queue).
    /// Also increments skip counts for non-selected tasks.
    fn select_next(&mut self) -> Option<u64> {
        // Check deadline-first: tasks with nearest deadline get priority.
        let mut deadline_candidate: Option<(u64, u64)> = None; // (id, deadline)
        for task in self.tasks.values() {
            if task.state == CoopTaskState::Ready || task.state == CoopTaskState::Yielded {
                if let Some(dl) = task.deadline_ms {
                    if deadline_candidate.is_none() || dl < deadline_candidate.unwrap().1 {
                        deadline_candidate = Some((task.id, dl));
                    }
                }
            }
        }
        // If there's a task whose deadline is soon (within 2x time slice), prefer it.
        if let Some((id, dl)) = deadline_candidate {
            if dl <= self.current_time_ms + self.time_slice_ms * 2 {
                return Some(id);
            }
        }

        // Otherwise, pick from highest priority queue.
        let mut priorities: Vec<CoopPriority> = self.ready_queues.keys().copied().collect();
        priorities.sort();
        let mut selected: Option<u64> = None;
        'outer: for prio in priorities {
            if let Some(q) = self.ready_queues.get_mut(&prio) {
                while let Some(id) = q.pop_front() {
                    let state = self.tasks.get(&id).map(|t| t.state);
                    match state {
                        Some(CoopTaskState::Ready) | Some(CoopTaskState::Yielded) => {
                            selected = Some(id);
                            break 'outer;
                        }
                        _ => continue, // Skip completed/cancelled.
                    }
                }
            }
        }
        if let Some(id) = selected {
            // Increment skip count for all other ready tasks.
            for task in self.tasks.values_mut() {
                if task.id != id
                    && (task.state == CoopTaskState::Ready
                        || task.state == CoopTaskState::Yielded)
                {
                    task.skip_count += 1;
                }
            }
        }
        selected
    }

    /// Execute one scheduling round (one time slice).
    /// Returns the id of the task that ran, or None if idle.
    pub fn step(&mut self) -> Option<u64> {
        self.anti_starvation();
        let id = match self.select_next() {
            Some(id) => id,
            None => return None,
        };

        let slice = self.time_slice_ms;
        let work = self.work_per_ms * slice;
        self.current_time_ms += slice;

        if let Some(task) = self.tasks.get_mut(&id) {
            task.state = CoopTaskState::Running;
            task.cpu_time_ms += slice;
            task.work_done = (task.work_done + work).min(task.total_work);
            task.skip_count = 0;

            if task.is_finished() {
                task.state = CoopTaskState::Completed;
            } else {
                // Yield — put back in ready queue.
                task.state = CoopTaskState::Yielded;
                let prio = task.priority;
                self.ready_queues.entry(prio).or_default().push_back(id);
            }
        }

        self.schedule_log.push(id);
        Some(id)
    }

    /// Run until all tasks complete or max_steps reached.
    pub fn run_all(&mut self, max_steps: usize) -> usize {
        let mut steps = 0;
        while steps < max_steps {
            if self.step().is_none() {
                break;
            }
            steps += 1;
        }
        steps
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_task_completion() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(1);
        let id = sched.alloc_id();
        sched.add_task(CoopTask::new(id, "work", 30));
        let steps = sched.run_all(100);
        assert_eq!(steps, 3); // 30 work / (1 * 10) = 3 slices
        assert_eq!(
            sched.get_task(id).unwrap().state,
            CoopTaskState::Completed
        );
    }

    #[test]
    fn test_priority_scheduling() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(1);
        let lo = sched.alloc_id();
        let hi = sched.alloc_id();
        sched.add_task(CoopTask::new(lo, "low", 10).with_priority(CoopPriority::LOW));
        sched.add_task(CoopTask::new(hi, "high", 10).with_priority(CoopPriority::HIGH));
        sched.step(); // Should pick high first.
        assert_eq!(sched.schedule_log()[0], hi);
    }

    #[test]
    fn test_round_robin_same_priority() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(1);
        let a = sched.alloc_id();
        let b = sched.alloc_id();
        sched.add_task(CoopTask::new(a, "a", 30));
        sched.add_task(CoopTask::new(b, "b", 30));
        sched.step();
        sched.step();
        let log = sched.schedule_log();
        // Should alternate (or at least both get scheduled).
        assert!(log.contains(&a));
        assert!(log.contains(&b));
    }

    #[test]
    fn test_yield_and_resume() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(1);
        let id = sched.alloc_id();
        sched.add_task(CoopTask::new(id, "yielder", 25));
        sched.step(); // 10 done
        assert_eq!(sched.get_task(id).unwrap().state, CoopTaskState::Yielded);
        assert_eq!(sched.get_task(id).unwrap().work_done, 10);
        sched.step(); // 20 done
        sched.step(); // 25 done (capped at total_work)
        assert_eq!(
            sched.get_task(id).unwrap().state,
            CoopTaskState::Completed
        );
    }

    #[test]
    fn test_cancellation() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(1);
        let id = sched.alloc_id();
        sched.add_task(CoopTask::new(id, "cancel_me", 100));
        assert!(sched.cancel(id));
        let result = sched.step();
        assert!(result.is_none()); // No runnable tasks.
        assert_eq!(
            sched.get_task(id).unwrap().state,
            CoopTaskState::Cancelled
        );
    }

    #[test]
    fn test_progress_reporting() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(2);
        let id = sched.alloc_id();
        sched.add_task(CoopTask::new(id, "progress", 100));
        sched.step(); // 20 work done
        let progress = sched.get_task(id).unwrap().progress();
        assert!((progress - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_starvation_prevention() {
        let mut sched = CoopScheduler::new(10)
            .with_work_per_ms(1)
            .with_starvation_threshold(3);
        let hi = sched.alloc_id();
        let lo = sched.alloc_id();
        sched.add_task(CoopTask::new(hi, "high", 100).with_priority(CoopPriority::HIGH));
        sched.add_task(CoopTask::new(lo, "low", 100).with_priority(CoopPriority::LOW));
        // Run several steps — low priority should eventually get boosted.
        for _ in 0..10 {
            sched.step();
        }
        let log = sched.schedule_log();
        // Low-priority task should appear in the log due to anti-starvation.
        assert!(log.contains(&lo));
    }

    #[test]
    fn test_deadline_scheduling() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(1);
        let normal = sched.alloc_id();
        let urgent = sched.alloc_id();
        sched.add_task(CoopTask::new(normal, "normal", 10).with_priority(CoopPriority::HIGH));
        sched.add_task(
            CoopTask::new(urgent, "urgent", 10)
                .with_priority(CoopPriority::LOW)
                .with_deadline(15),
        );
        sched.step(); // Deadline task should be preferred (deadline within 2x slice).
        assert_eq!(sched.schedule_log()[0], urgent);
    }

    #[test]
    fn test_cpu_time_tracking() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(1);
        let id = sched.alloc_id();
        sched.add_task(CoopTask::new(id, "tracked", 50));
        sched.run_all(100);
        assert_eq!(sched.get_task(id).unwrap().cpu_time_ms, 50);
    }

    #[test]
    fn test_time_advances() {
        let mut sched = CoopScheduler::new(10).with_work_per_ms(1);
        let id = sched.alloc_id();
        sched.add_task(CoopTask::new(id, "t", 30));
        sched.run_all(100);
        assert_eq!(sched.current_time(), 30);
    }

    #[test]
    fn test_idle_returns_none() {
        let mut sched = CoopScheduler::new(10);
        assert!(sched.step().is_none());
    }
}
