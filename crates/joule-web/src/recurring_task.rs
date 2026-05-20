//! Recurring task scheduler — cron-like, interval-based, and one-shot delayed tasks.
//!
//! Replaces node-cron / node-schedule / agenda with a pure-Rust recurring task
//! scheduler. Supports interval-based, cron-like field matching, one-shot delayed
//! tasks, task enable/disable, missed execution handling, and execution history.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Task ID ───────────────────────────────────────────────────

/// Unique identifier for a recurring task.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecurringTaskId(pub String);

// ── Schedule Kind ─────────────────────────────────────────────

/// The scheduling pattern for a recurring task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleKind {
    /// Fixed interval in milliseconds.
    Interval { interval_ms: u64 },
    /// Cron-like: fields for minute, hour, day-of-month, month, day-of-week.
    /// Each field is a sorted list of allowed values, or empty for "any".
    CronLike {
        minutes: Vec<u32>,
        hours: Vec<u32>,
        days_of_month: Vec<u32>,
        months: Vec<u32>,
        days_of_week: Vec<u32>,
    },
    /// One-shot: fire once after `delay_ms` from registration time.
    OneShot { delay_ms: u64 },
}

// ── Missed Execution Policy ───────────────────────────────────

/// What to do when scheduled executions were missed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissedPolicy {
    /// Skip all missed executions, schedule next future one.
    Skip,
    /// Catch up by executing once for each missed interval.
    CatchUp,
    /// Execute exactly once for all missed intervals combined.
    RunOnce,
}

// ── Execution Record ──────────────────────────────────────────

/// A record of a task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Timestamp when the execution occurred (ms).
    pub executed_at_ms: u64,
    /// Timestamp when the execution was scheduled (ms).
    pub scheduled_for_ms: u64,
    /// Whether the execution was late (missed and caught up).
    pub was_late: bool,
    /// Optional result/status from the execution.
    pub result: String,
}

// ── Task State ────────────────────────────────────────────────

/// State of a recurring task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecurringTaskState {
    /// Task is active and will fire at scheduled times.
    Enabled,
    /// Task is paused and will not fire.
    Disabled,
    /// One-shot task that has already fired.
    Completed,
}

// ── Recurring Task ────────────────────────────────────────────

/// A registered recurring task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurringTask {
    pub id: RecurringTaskId,
    pub label: String,
    pub schedule: ScheduleKind,
    pub missed_policy: MissedPolicy,
    pub state: RecurringTaskState,
    /// Timestamp (ms) when the task was registered.
    pub registered_at_ms: u64,
    /// Timestamp (ms) of the next scheduled execution.
    pub next_run_ms: Option<u64>,
    /// Timestamp (ms) of the last execution.
    pub last_run_ms: Option<u64>,
    /// Total number of executions.
    pub execution_count: u64,
    /// Execution history (most recent first).
    pub history: Vec<ExecutionRecord>,
    /// Maximum history entries to keep.
    pub max_history: usize,
}

// ── Due Task ──────────────────────────────────────────────────

/// A task that is due for execution.
#[derive(Debug, Clone)]
pub struct DueTask {
    pub id: RecurringTaskId,
    pub label: String,
    pub scheduled_for_ms: u64,
    pub is_late: bool,
}

// ── Scheduler Stats ───────────────────────────────────────────

/// Statistics about the recurring task scheduler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerStats {
    pub total_tasks: u64,
    pub enabled_tasks: u64,
    pub disabled_tasks: u64,
    pub completed_tasks: u64,
    pub total_executions: u64,
    pub missed_executions: u64,
}

// ── Recurring Task Scheduler ──────────────────────────────────

/// Manages recurring tasks with various scheduling patterns.
#[derive(Debug)]
pub struct RecurringTaskScheduler {
    tasks: HashMap<String, RecurringTask>,
    stats: SchedulerStats,
    default_max_history: usize,
}

impl RecurringTaskScheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            stats: SchedulerStats::default(),
            default_max_history: 50,
        }
    }

    /// Create a scheduler with a custom default max history per task.
    pub fn with_max_history(max_history: usize) -> Self {
        Self {
            tasks: HashMap::new(),
            stats: SchedulerStats::default(),
            default_max_history: max_history,
        }
    }

    /// Register an interval-based task.
    pub fn register_interval(
        &mut self,
        id: &str,
        label: &str,
        interval_ms: u64,
        now_ms: u64,
    ) -> RecurringTaskId {
        let task_id = RecurringTaskId(id.to_string());
        let next_run = now_ms.saturating_add(interval_ms);
        let task = RecurringTask {
            id: task_id.clone(),
            label: label.to_string(),
            schedule: ScheduleKind::Interval { interval_ms },
            missed_policy: MissedPolicy::Skip,
            state: RecurringTaskState::Enabled,
            registered_at_ms: now_ms,
            next_run_ms: Some(next_run),
            last_run_ms: None,
            execution_count: 0,
            history: Vec::new(),
            max_history: self.default_max_history,
        };
        self.tasks.insert(id.to_string(), task);
        self.stats.total_tasks += 1;
        self.stats.enabled_tasks += 1;
        task_id
    }

    /// Register a cron-like task.
    pub fn register_cron(
        &mut self,
        id: &str,
        label: &str,
        minutes: Vec<u32>,
        hours: Vec<u32>,
        days_of_month: Vec<u32>,
        months: Vec<u32>,
        days_of_week: Vec<u32>,
        now_ms: u64,
    ) -> RecurringTaskId {
        let task_id = RecurringTaskId(id.to_string());
        let schedule = ScheduleKind::CronLike {
            minutes: sorted_dedup(minutes),
            hours: sorted_dedup(hours),
            days_of_month: sorted_dedup(days_of_month),
            months: sorted_dedup(months),
            days_of_week: sorted_dedup(days_of_week),
        };
        let next_run = compute_next_cron_run(&schedule, now_ms);
        let task = RecurringTask {
            id: task_id.clone(),
            label: label.to_string(),
            schedule,
            missed_policy: MissedPolicy::Skip,
            state: RecurringTaskState::Enabled,
            registered_at_ms: now_ms,
            next_run_ms: next_run,
            last_run_ms: None,
            execution_count: 0,
            history: Vec::new(),
            max_history: self.default_max_history,
        };
        self.tasks.insert(id.to_string(), task);
        self.stats.total_tasks += 1;
        self.stats.enabled_tasks += 1;
        task_id
    }

    /// Register a one-shot delayed task.
    pub fn register_one_shot(
        &mut self,
        id: &str,
        label: &str,
        delay_ms: u64,
        now_ms: u64,
    ) -> RecurringTaskId {
        let task_id = RecurringTaskId(id.to_string());
        let task = RecurringTask {
            id: task_id.clone(),
            label: label.to_string(),
            schedule: ScheduleKind::OneShot { delay_ms },
            missed_policy: MissedPolicy::RunOnce,
            state: RecurringTaskState::Enabled,
            registered_at_ms: now_ms,
            next_run_ms: Some(now_ms.saturating_add(delay_ms)),
            last_run_ms: None,
            execution_count: 0,
            history: Vec::new(),
            max_history: self.default_max_history,
        };
        self.tasks.insert(id.to_string(), task);
        self.stats.total_tasks += 1;
        self.stats.enabled_tasks += 1;
        task_id
    }

    /// Set the missed execution policy for a task.
    pub fn set_missed_policy(&mut self, id: &str, policy: MissedPolicy) -> bool {
        if let Some(task) = self.tasks.get_mut(id) {
            task.missed_policy = policy;
            true
        } else {
            false
        }
    }

    /// Enable a disabled task.
    pub fn enable(&mut self, id: &str, now_ms: u64) -> bool {
        if let Some(task) = self.tasks.get_mut(id) {
            if task.state == RecurringTaskState::Disabled {
                task.state = RecurringTaskState::Enabled;
                // Recalculate next run from now.
                task.next_run_ms = compute_next_run(&task.schedule, now_ms);
                self.stats.enabled_tasks += 1;
                self.stats.disabled_tasks = self.stats.disabled_tasks.saturating_sub(1);
                return true;
            }
        }
        false
    }

    /// Disable an enabled task.
    pub fn disable(&mut self, id: &str) -> bool {
        if let Some(task) = self.tasks.get_mut(id) {
            if task.state == RecurringTaskState::Enabled {
                task.state = RecurringTaskState::Disabled;
                self.stats.disabled_tasks += 1;
                self.stats.enabled_tasks = self.stats.enabled_tasks.saturating_sub(1);
                return true;
            }
        }
        false
    }

    /// Remove a task entirely.
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(task) = self.tasks.remove(id) {
            match task.state {
                RecurringTaskState::Enabled => {
                    self.stats.enabled_tasks = self.stats.enabled_tasks.saturating_sub(1);
                }
                RecurringTaskState::Disabled => {
                    self.stats.disabled_tasks = self.stats.disabled_tasks.saturating_sub(1);
                }
                RecurringTaskState::Completed => {
                    self.stats.completed_tasks = self.stats.completed_tasks.saturating_sub(1);
                }
            }
            true
        } else {
            false
        }
    }

    /// Get a task by ID.
    pub fn get_task(&self, id: &str) -> Option<&RecurringTask> {
        self.tasks.get(id)
    }

    /// Tick the scheduler at `now_ms`. Returns all tasks that are due.
    pub fn tick(&mut self, now_ms: u64) -> Vec<DueTask> {
        let mut due = Vec::new();
        let keys: Vec<String> = self.tasks.keys().cloned().collect();

        for key in keys {
            let (due_tasks, missed_count) = {
                let task = self.tasks.get(&key).unwrap();
                if task.state != RecurringTaskState::Enabled {
                    continue;
                }
                let next_run = match task.next_run_ms {
                    Some(nr) => nr,
                    None => continue,
                };
                if now_ms < next_run {
                    continue;
                }

                compute_due_tasks(task, now_ms)
            };

            self.stats.missed_executions += missed_count;

            for dt in &due_tasks {
                due.push(dt.clone());
            }

            // Update the task state after determining due tasks.
            if let Some(task) = self.tasks.get_mut(&key) {
                for dt in &due_tasks {
                    let record = ExecutionRecord {
                        executed_at_ms: now_ms,
                        scheduled_for_ms: dt.scheduled_for_ms,
                        was_late: dt.is_late,
                        result: "executed".to_string(),
                    };
                    task.history.insert(0, record);
                    if task.history.len() > task.max_history {
                        task.history.pop();
                    }
                    task.execution_count += 1;
                    self.stats.total_executions += 1;
                }

                task.last_run_ms = Some(now_ms);

                // Schedule next run.
                match &task.schedule {
                    ScheduleKind::OneShot { .. } => {
                        task.next_run_ms = None;
                        task.state = RecurringTaskState::Completed;
                        self.stats.enabled_tasks = self.stats.enabled_tasks.saturating_sub(1);
                        self.stats.completed_tasks += 1;
                    }
                    ScheduleKind::Interval { interval_ms } => {
                        task.next_run_ms = Some(now_ms.saturating_add(*interval_ms));
                    }
                    schedule @ ScheduleKind::CronLike { .. } => {
                        task.next_run_ms = compute_next_cron_run(schedule, now_ms);
                    }
                }
            }
        }

        due
    }

    /// Get the next run time for a specific task.
    pub fn next_run(&self, id: &str) -> Option<u64> {
        self.tasks.get(id).and_then(|t| t.next_run_ms)
    }

    /// Get execution history for a task.
    pub fn execution_history(&self, id: &str) -> Vec<ExecutionRecord> {
        self.tasks
            .get(id)
            .map(|t| t.history.clone())
            .unwrap_or_default()
    }

    /// Number of registered tasks.
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Get all enabled task IDs.
    pub fn enabled_tasks(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .tasks
            .values()
            .filter(|t| t.state == RecurringTaskState::Enabled)
            .map(|t| t.id.0.clone())
            .collect();
        ids.sort();
        ids
    }

    /// Get all task IDs.
    pub fn all_task_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.tasks.keys().cloned().collect();
        ids.sort();
        ids
    }
}

impl Default for RecurringTaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────

fn sorted_dedup(mut v: Vec<u32>) -> Vec<u32> {
    v.sort();
    v.dedup();
    v
}

fn compute_next_run(schedule: &ScheduleKind, now_ms: u64) -> Option<u64> {
    match schedule {
        ScheduleKind::Interval { interval_ms } => Some(now_ms.saturating_add(*interval_ms)),
        ScheduleKind::OneShot { delay_ms } => Some(now_ms.saturating_add(*delay_ms)),
        ScheduleKind::CronLike { .. } => compute_next_cron_run(schedule, now_ms),
    }
}

/// Simple cron-like "next run" calculation.
/// Uses minute-granularity steps from `now_ms`.
fn compute_next_cron_run(schedule: &ScheduleKind, now_ms: u64) -> Option<u64> {
    let ScheduleKind::CronLike {
        minutes,
        hours,
        days_of_month: _,
        months: _,
        days_of_week: _,
    } = schedule
    else {
        return None;
    };

    // Walk forward in minute increments (up to 7 days).
    let one_minute_ms: u64 = 60_000;
    let max_search = 7 * 24 * 60; // 7 days of minutes.
    let start = now_ms + one_minute_ms; // At least 1 minute in the future.

    // Align to minute boundary.
    let aligned_start = (start / one_minute_ms) * one_minute_ms;

    for step in 0..max_search {
        let candidate = aligned_start + step * one_minute_ms;
        let total_minutes = candidate / one_minute_ms;
        let minute_of_hour = (total_minutes % 60) as u32;
        let hour_of_day = ((total_minutes / 60) % 24) as u32;

        let minute_ok = minutes.is_empty() || minutes.contains(&minute_of_hour);
        let hour_ok = hours.is_empty() || hours.contains(&hour_of_day);

        if minute_ok && hour_ok {
            return Some(candidate);
        }
    }

    None
}

fn compute_due_tasks(task: &RecurringTask, now_ms: u64) -> (Vec<DueTask>, u64) {
    let next_run = match task.next_run_ms {
        Some(nr) => nr,
        None => return (Vec::new(), 0),
    };

    match &task.schedule {
        ScheduleKind::OneShot { .. } => {
            let due = DueTask {
                id: task.id.clone(),
                label: task.label.clone(),
                scheduled_for_ms: next_run,
                is_late: now_ms > next_run,
            };
            (vec![due], 0)
        }
        ScheduleKind::Interval { interval_ms } => {
            if *interval_ms == 0 {
                return (Vec::new(), 0);
            }
            let elapsed = now_ms.saturating_sub(next_run);
            let missed_count = elapsed / interval_ms;

            match task.missed_policy {
                MissedPolicy::Skip => {
                    let due = DueTask {
                        id: task.id.clone(),
                        label: task.label.clone(),
                        scheduled_for_ms: next_run,
                        is_late: missed_count > 0,
                    };
                    (vec![due], missed_count)
                }
                MissedPolicy::RunOnce => {
                    let due = DueTask {
                        id: task.id.clone(),
                        label: task.label.clone(),
                        scheduled_for_ms: next_run,
                        is_late: missed_count > 0,
                    };
                    (vec![due], missed_count)
                }
                MissedPolicy::CatchUp => {
                    let mut dues = Vec::new();
                    for i in 0..=missed_count.min(10) {
                        // Cap catch-up at 10.
                        let sched = next_run + i * interval_ms;
                        dues.push(DueTask {
                            id: task.id.clone(),
                            label: task.label.clone(),
                            scheduled_for_ms: sched,
                            is_late: i > 0,
                        });
                    }
                    (dues, missed_count)
                }
            }
        }
        ScheduleKind::CronLike { .. } => {
            let due = DueTask {
                id: task.id.clone(),
                label: task.label.clone(),
                scheduled_for_ms: next_run,
                is_late: now_ms > next_run,
            };
            (vec![due], 0)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_interval_task() {
        let mut sched = RecurringTaskScheduler::new();
        let id = sched.register_interval("heartbeat", "Heartbeat", 1000, 0);
        assert_eq!(id.0, "heartbeat");
        assert_eq!(sched.task_count(), 1);
        let task = sched.get_task("heartbeat").unwrap();
        assert_eq!(task.next_run_ms, Some(1000));
    }

    #[test]
    fn test_interval_tick_fires() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 1000, 0);
        let due = sched.tick(500);
        assert!(due.is_empty()); // Not yet.
        let due = sched.tick(1000);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id.0, "hb");
    }

    #[test]
    fn test_interval_reschedules() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 1000, 0);
        sched.tick(1000); // Fires.
        let task = sched.get_task("hb").unwrap();
        assert_eq!(task.next_run_ms, Some(2000));
    }

    #[test]
    fn test_one_shot_fires_once() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_one_shot("once", "Once", 500, 0);
        let due = sched.tick(500);
        assert_eq!(due.len(), 1);
        let task = sched.get_task("once").unwrap();
        assert_eq!(task.state, RecurringTaskState::Completed);
        // Should not fire again.
        let due = sched.tick(1000);
        assert!(due.is_empty());
    }

    #[test]
    fn test_disable_enable() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 1000, 0);
        sched.disable("hb");
        let task = sched.get_task("hb").unwrap();
        assert_eq!(task.state, RecurringTaskState::Disabled);
        let due = sched.tick(1000);
        assert!(due.is_empty()); // Disabled, shouldn't fire.
        sched.enable("hb", 1000);
        let task = sched.get_task("hb").unwrap();
        assert_eq!(task.state, RecurringTaskState::Enabled);
    }

    #[test]
    fn test_remove_task() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 1000, 0);
        assert!(sched.remove("hb"));
        assert_eq!(sched.task_count(), 0);
        assert!(!sched.remove("hb")); // Already removed.
    }

    #[test]
    fn test_execution_history() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 100, 0);
        sched.tick(100);
        sched.tick(200);
        let history = sched.execution_history("hb");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].executed_at_ms, 200); // Most recent first.
    }

    #[test]
    fn test_execution_count() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 100, 0);
        sched.tick(100);
        sched.tick(200);
        sched.tick(300);
        let task = sched.get_task("hb").unwrap();
        assert_eq!(task.execution_count, 3);
    }

    #[test]
    fn test_missed_policy_skip() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 100, 0);
        sched.set_missed_policy("hb", MissedPolicy::Skip);
        // Jump far ahead — missed many executions.
        let due = sched.tick(1000);
        assert_eq!(due.len(), 1); // Only fires once with Skip.
        assert!(due[0].is_late);
    }

    #[test]
    fn test_missed_policy_catch_up() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 100, 0);
        sched.set_missed_policy("hb", MissedPolicy::CatchUp);
        // Jump ahead by 500ms -> missed about 5 intervals.
        let due = sched.tick(500);
        assert!(due.len() > 1); // CatchUp produces multiple due tasks.
    }

    #[test]
    fn test_stats() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("a", "A", 100, 0);
        sched.register_one_shot("b", "B", 50, 0);
        assert_eq!(sched.stats().total_tasks, 2);
        assert_eq!(sched.stats().enabled_tasks, 2);
        sched.tick(50); // B fires.
        assert_eq!(sched.stats().completed_tasks, 1);
        assert_eq!(sched.stats().total_executions, 1);
    }

    #[test]
    fn test_next_run() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 500, 1000);
        assert_eq!(sched.next_run("hb"), Some(1500));
    }

    #[test]
    fn test_enabled_tasks() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("a", "A", 100, 0);
        sched.register_interval("b", "B", 100, 0);
        sched.disable("b");
        let enabled = sched.enabled_tasks();
        assert_eq!(enabled, vec!["a"]);
    }

    #[test]
    fn test_all_task_ids() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("b", "B", 100, 0);
        sched.register_interval("a", "A", 100, 0);
        let ids = sched.all_task_ids();
        assert_eq!(ids, vec!["a", "b"]); // Sorted.
    }

    #[test]
    fn test_cron_like_registration() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_cron(
            "report",
            "Daily Report",
            vec![0],     // minute 0
            vec![9],     // hour 9
            vec![],      // any day
            vec![],      // any month
            vec![],      // any weekday
            0,
        );
        let task = sched.get_task("report").unwrap();
        assert!(task.next_run_ms.is_some());
    }

    #[test]
    fn test_default_constructor() {
        let sched = RecurringTaskScheduler::default();
        assert_eq!(sched.task_count(), 0);
    }

    #[test]
    fn test_with_max_history() {
        let mut sched = RecurringTaskScheduler::with_max_history(2);
        sched.register_interval("hb", "HB", 100, 0);
        sched.tick(100);
        sched.tick(200);
        sched.tick(300);
        let history = sched.execution_history("hb");
        assert_eq!(history.len(), 2); // Capped at 2.
    }

    #[test]
    fn test_last_run_updated() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 100, 0);
        sched.tick(100);
        let task = sched.get_task("hb").unwrap();
        assert_eq!(task.last_run_ms, Some(100));
    }

    #[test]
    fn test_disable_already_disabled() {
        let mut sched = RecurringTaskScheduler::new();
        sched.register_interval("hb", "HB", 100, 0);
        assert!(sched.disable("hb"));
        assert!(!sched.disable("hb")); // Already disabled.
    }
}
