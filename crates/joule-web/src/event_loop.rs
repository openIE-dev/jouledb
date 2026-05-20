//! Event loop simulation — microtask/macrotask queues, rAF, timers, idle callbacks.
//!
//! Models the browser event loop in pure Rust: microtasks drain before
//! macrotasks, requestAnimationFrame scheduling, setTimeout/setInterval,
//! and idle callbacks. No actual async runtime — deterministic simulation.

use std::collections::{BTreeMap, VecDeque};

// ── Callback ───────────────────────────────────────────────────

/// A unique callback id.
pub type CallbackId = u64;

/// Represents a scheduled callback.
#[derive(Debug, Clone)]
pub struct Callback {
    pub id: CallbackId,
    pub label: String,
    pub executed: bool,
}

impl Callback {
    fn new(id: CallbackId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            executed: false,
        }
    }
}

// ── Timer ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Timer {
    id: CallbackId,
    label: String,
    delay_ms: u64,
    repeating: bool,
    next_fire: u64,
}

// ── Event Loop ─────────────────────────────────────────────────

/// Simulated browser event loop.
#[derive(Debug)]
pub struct EventLoop {
    /// Monotonic time in ms.
    time_ms: u64,
    /// Tick counter (one tick = one full event loop iteration).
    tick_count: u64,
    /// Microtask queue (promises, queueMicrotask).
    microtasks: VecDeque<Callback>,
    /// Macrotask queue (setTimeout callbacks that have fired, I/O, etc.).
    macrotasks: VecDeque<Callback>,
    /// requestAnimationFrame callbacks for the next frame.
    raf_queue: Vec<Callback>,
    /// Idle callbacks (requestIdleCallback).
    idle_queue: VecDeque<Callback>,
    /// Scheduled timers (setTimeout / setInterval), keyed by next_fire time.
    timers: BTreeMap<u64, Vec<Timer>>,
    /// All timers by id for cancellation.
    timer_ids: std::collections::HashMap<CallbackId, u64>,
    /// Log of executed callback ids in order.
    execution_log: Vec<CallbackId>,
    /// Next callback id.
    next_id: CallbackId,
    /// Frame interval for rAF (default ~16ms for 60fps).
    frame_interval_ms: u64,
    /// Time of last rAF frame.
    last_frame_ms: u64,
}

impl EventLoop {
    pub fn new() -> Self {
        Self {
            time_ms: 0,
            tick_count: 0,
            microtasks: VecDeque::new(),
            macrotasks: VecDeque::new(),
            raf_queue: Vec::new(),
            idle_queue: VecDeque::new(),
            timers: BTreeMap::new(),
            timer_ids: std::collections::HashMap::new(),
            execution_log: Vec::new(),
            next_id: 1,
            frame_interval_ms: 16,
            last_frame_ms: 0,
        }
    }

    fn alloc_id(&mut self) -> CallbackId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn time_ms(&self) -> u64 {
        self.time_ms
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    pub fn execution_log(&self) -> &[CallbackId] {
        &self.execution_log
    }

    // ── Scheduling ─────────────────────────────────────────────

    /// Queue a microtask (like Promise.then, queueMicrotask).
    pub fn queue_microtask(&mut self, label: impl Into<String>) -> CallbackId {
        let id = self.alloc_id();
        self.microtasks.push_back(Callback::new(id, label));
        id
    }

    /// Queue a macrotask.
    pub fn queue_macrotask(&mut self, label: impl Into<String>) -> CallbackId {
        let id = self.alloc_id();
        self.macrotasks.push_back(Callback::new(id, label));
        id
    }

    /// Schedule a requestAnimationFrame callback.
    pub fn request_animation_frame(&mut self, label: impl Into<String>) -> CallbackId {
        let id = self.alloc_id();
        self.raf_queue.push(Callback::new(id, label));
        id
    }

    /// Schedule a setTimeout.
    pub fn set_timeout(&mut self, label: impl Into<String>, delay_ms: u64) -> CallbackId {
        let id = self.alloc_id();
        let fire = self.time_ms + delay_ms;
        let timer = Timer {
            id,
            label: label.into(),
            delay_ms,
            repeating: false,
            next_fire: fire,
        };
        self.timers.entry(fire).or_default().push(timer);
        self.timer_ids.insert(id, fire);
        id
    }

    /// Schedule a setInterval.
    pub fn set_interval(&mut self, label: impl Into<String>, interval_ms: u64) -> CallbackId {
        let id = self.alloc_id();
        let fire = self.time_ms + interval_ms;
        let timer = Timer {
            id,
            label: label.into(),
            delay_ms: interval_ms,
            repeating: true,
            next_fire: fire,
        };
        self.timers.entry(fire).or_default().push(timer);
        self.timer_ids.insert(id, fire);
        id
    }

    /// Cancel a timer by id.
    pub fn clear_timer(&mut self, id: CallbackId) {
        if let Some(fire_time) = self.timer_ids.remove(&id) {
            if let Some(timers) = self.timers.get_mut(&fire_time) {
                timers.retain(|t| t.id != id);
                if timers.is_empty() {
                    self.timers.remove(&fire_time);
                }
            }
        }
    }

    /// Schedule an idle callback (requestIdleCallback).
    pub fn request_idle_callback(&mut self, label: impl Into<String>) -> CallbackId {
        let id = self.alloc_id();
        self.idle_queue.push_back(Callback::new(id, label));
        id
    }

    // ── Execution ──────────────────────────────────────────────

    /// Drain all microtasks (microtasks can enqueue more microtasks).
    fn drain_microtasks(&mut self) {
        while let Some(mut cb) = self.microtasks.pop_front() {
            cb.executed = true;
            self.execution_log.push(cb.id);
        }
    }

    /// Fire any timers whose fire time <= current time.
    fn fire_timers(&mut self) {
        let due_times: Vec<u64> = self
            .timers
            .range(..=self.time_ms)
            .map(|(k, _)| *k)
            .collect();

        for t in due_times {
            if let Some(timers) = self.timers.remove(&t) {
                for timer in timers {
                    self.timer_ids.remove(&timer.id);
                    // Push as macrotask.
                    self.macrotasks
                        .push_back(Callback::new(timer.id, &timer.label));
                    // Re-schedule intervals.
                    if timer.repeating {
                        let next_fire = self.time_ms + timer.delay_ms;
                        let new_timer = Timer {
                            id: timer.id,
                            label: timer.label,
                            delay_ms: timer.delay_ms,
                            repeating: true,
                            next_fire,
                        };
                        self.timers.entry(next_fire).or_default().push(new_timer);
                        self.timer_ids.insert(timer.id, next_fire);
                    }
                }
            }
        }
    }

    /// Run one event loop tick: microtasks → macrotask → rAF → idle.
    /// Advances time by `advance_ms`.
    pub fn tick(&mut self, advance_ms: u64) -> Vec<CallbackId> {
        self.time_ms += advance_ms;
        self.tick_count += 1;
        let log_start = self.execution_log.len();

        // 1. Fire due timers (enqueue as macrotasks).
        self.fire_timers();

        // 2. Drain microtasks.
        self.drain_microtasks();

        // 3. Execute one macrotask, then drain microtasks again.
        if let Some(mut cb) = self.macrotasks.pop_front() {
            cb.executed = true;
            self.execution_log.push(cb.id);
            self.drain_microtasks();
        }

        // 4. rAF — if enough time has passed since last frame.
        if self.time_ms - self.last_frame_ms >= self.frame_interval_ms {
            self.last_frame_ms = self.time_ms;
            let raf_cbs: Vec<Callback> = self.raf_queue.drain(..).collect();
            for mut cb in raf_cbs {
                cb.executed = true;
                self.execution_log.push(cb.id);
            }
            self.drain_microtasks();
        }

        // 5. Idle callbacks — only if nothing else is pending.
        if self.microtasks.is_empty() && self.macrotasks.is_empty() {
            if let Some(mut cb) = self.idle_queue.pop_front() {
                cb.executed = true;
                self.execution_log.push(cb.id);
            }
        }

        self.execution_log[log_start..].to_vec()
    }

    /// Run ticks until no more work remains, up to max_ticks.
    pub fn run_to_completion(&mut self, tick_ms: u64, max_ticks: u64) -> u64 {
        let mut ticks = 0;
        while ticks < max_ticks {
            let has_work = !self.microtasks.is_empty()
                || !self.macrotasks.is_empty()
                || !self.raf_queue.is_empty()
                || !self.idle_queue.is_empty()
                || !self.timers.is_empty();
            if !has_work {
                break;
            }
            self.tick(tick_ms);
            ticks += 1;
        }
        ticks
    }

    /// Number of pending items across all queues.
    pub fn pending_count(&self) -> usize {
        self.microtasks.len()
            + self.macrotasks.len()
            + self.raf_queue.len()
            + self.idle_queue.len()
            + self.timers.values().map(|v| v.len()).sum::<usize>()
    }
}

impl Default for EventLoop {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microtask_before_macrotask() {
        let mut el = EventLoop::new();
        let macro_id = el.queue_macrotask("macro1");
        let micro_id = el.queue_microtask("micro1");
        let executed = el.tick(1);
        // Microtask should execute before macrotask.
        assert_eq!(executed[0], micro_id);
        assert_eq!(executed[1], macro_id);
    }

    #[test]
    fn test_set_timeout() {
        let mut el = EventLoop::new();
        let id = el.set_timeout("delayed", 100);
        let executed = el.tick(50);
        assert!(!executed.contains(&id)); // Not yet.
        let executed = el.tick(51);
        assert!(executed.contains(&id)); // Now fired (time=101 >= 100).
    }

    #[test]
    fn test_set_interval() {
        let mut el = EventLoop::new();
        let id = el.set_interval("repeat", 50);
        el.tick(50); // fires at t=50
        assert!(el.execution_log().contains(&id));
        let count_before = el.execution_log().iter().filter(|&&x| x == id).count();
        el.tick(50); // fires again at t=100
        let count_after = el.execution_log().iter().filter(|&&x| x == id).count();
        assert_eq!(count_after, count_before + 1);
    }

    #[test]
    fn test_clear_timer() {
        let mut el = EventLoop::new();
        let id = el.set_timeout("cancel_me", 100);
        el.clear_timer(id);
        el.tick(200);
        assert!(!el.execution_log().contains(&id));
    }

    #[test]
    fn test_request_animation_frame() {
        let mut el = EventLoop::new();
        let id = el.request_animation_frame("paint");
        let executed = el.tick(16);
        assert!(executed.contains(&id));
    }

    #[test]
    fn test_raf_not_fired_before_frame_interval() {
        let mut el = EventLoop::new();
        // First tick at t=0 fires rAF (0-0 >= 16 is false, but let's check).
        // Actually, initial last_frame_ms=0, time starts at 0 + advance.
        let id = el.request_animation_frame("paint");
        let executed = el.tick(5); // t=5, 5-0 < 16
        assert!(!executed.contains(&id));
        // Next tick to t=21.
        let executed = el.tick(16); // t=21, 21-0 >= 16
        assert!(executed.contains(&id));
    }

    #[test]
    fn test_idle_callback() {
        let mut el = EventLoop::new();
        let id = el.request_idle_callback("idle_work");
        let executed = el.tick(16);
        // Idle should run when nothing else is pending.
        assert!(executed.contains(&id));
    }

    #[test]
    fn test_idle_deferred_when_busy() {
        let mut el = EventLoop::new();
        let idle_id = el.request_idle_callback("idle");
        el.queue_macrotask("busy1");
        el.queue_macrotask("busy2");
        let executed = el.tick(16);
        // Two macrotasks pending: one executes this tick, so idle should NOT run.
        assert!(!executed.contains(&idle_id));
    }

    #[test]
    fn test_tick_counting() {
        let mut el = EventLoop::new();
        assert_eq!(el.tick_count(), 0);
        el.tick(1);
        el.tick(1);
        el.tick(1);
        assert_eq!(el.tick_count(), 3);
    }

    #[test]
    fn test_run_to_completion() {
        let mut el = EventLoop::new();
        el.queue_microtask("m1");
        el.queue_macrotask("t1");
        el.queue_macrotask("t2");
        let ticks = el.run_to_completion(1, 100);
        assert!(ticks <= 100);
        assert_eq!(el.execution_log().len(), 3);
    }

    #[test]
    fn test_execution_order_complex() {
        let mut el = EventLoop::new();
        // Microtask → macrotask → rAF → idle — in one tick.
        let micro = el.queue_microtask("micro");
        let mac = el.queue_macrotask("macro");
        let raf = el.request_animation_frame("raf");
        let idle = el.request_idle_callback("idle");
        let executed = el.tick(16);
        assert_eq!(executed[0], micro);
        assert_eq!(executed[1], mac);
        assert_eq!(executed[2], raf);
        // idle may or may not run depending on queue state.
        // After macro+raf, queues are empty, so idle runs.
        assert!(executed.contains(&idle));
    }

    #[test]
    fn test_pending_count() {
        let mut el = EventLoop::new();
        el.queue_microtask("m");
        el.queue_macrotask("t");
        el.set_timeout("d", 100);
        assert_eq!(el.pending_count(), 3);
    }
}
