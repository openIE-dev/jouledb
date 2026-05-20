//! Side effect management with scheduling, cleanup, dependency comparison,
//! batching, priority ordering, and cancellation.
//!
//! Modeled after React's `useEffect` but as a standalone, synchronous system.

use std::collections::BTreeMap;

// ── Types ──

/// Unique identifier for an effect.
pub type EffectId = u64;

/// Priority for effect ordering. Lower numbers run first.
pub type Priority = i32;

/// Dependency values for an effect (shallow equality comparison).
pub type Dependencies = Vec<String>;

// ── Effect ──

/// An effect with optional cleanup and dependency tracking.
pub struct Effect {
    pub id: EffectId,
    pub dependencies: Option<Dependencies>,
    pub priority: Priority,
    run_fn: Box<dyn FnMut() -> Option<Box<dyn FnOnce()>>>,
    cleanup_fn: Option<Box<dyn FnOnce()>>,
    cancelled: bool,
    /// Previous dependency values for comparison.
    prev_deps: Option<Dependencies>,
    /// How many times this effect has run.
    run_count: u64,
}

impl Effect {
    /// Create a new effect. The `run_fn` is called on execution and may return
    /// a cleanup function that runs before the next execution or on removal.
    pub fn new(
        id: EffectId,
        dependencies: Option<Dependencies>,
        run_fn: impl FnMut() -> Option<Box<dyn FnOnce()>> + 'static,
    ) -> Self {
        Self {
            id,
            dependencies: dependencies.clone(),
            priority: 0,
            run_fn: Box::new(run_fn),
            cleanup_fn: None,
            cancelled: false,
            prev_deps: None,
            run_count: 0,
        }
    }

    /// Create a one-time effect (empty dependency list — runs once).
    pub fn once(id: EffectId, mut run_fn: impl FnMut() + 'static) -> Self {
        Self::new(id, Some(Vec::new()), move || {
            run_fn();
            None
        })
    }

    /// Set effect priority (lower runs first).
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Run the effect, handling cleanup of previous execution.
    fn execute(&mut self) {
        // Run cleanup from previous execution
        if let Some(cleanup) = self.cleanup_fn.take() {
            cleanup();
        }
        // Run the effect and store any new cleanup
        self.cleanup_fn = (self.run_fn)();
        self.prev_deps = self.dependencies.clone();
        self.run_count += 1;
    }

    /// Run only the cleanup function without re-running the effect.
    fn run_cleanup(&mut self) {
        if let Some(cleanup) = self.cleanup_fn.take() {
            cleanup();
        }
    }

    /// Check if dependencies have changed (shallow equality).
    fn deps_changed(&self) -> bool {
        match (&self.dependencies, &self.prev_deps) {
            // No deps = always run
            (None, _) => true,
            // Empty deps = only run once
            (Some(deps), None) if deps.is_empty() => self.run_count == 0,
            (Some(deps), None) => !deps.is_empty() || self.run_count == 0,
            (Some(current), Some(previous)) => current != previous,
        }
    }

    pub fn run_count(&self) -> u64 {
        self.run_count
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

// ── EffectScheduler ──

/// Manages a collection of effects, running them after state updates.
pub struct EffectScheduler {
    effects: BTreeMap<EffectId, Effect>,
    /// Whether we're batching effect execution.
    batching: bool,
    /// Effects queued during a batch.
    batch_queue: Vec<EffectId>,
}

impl Default for EffectScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectScheduler {
    pub fn new() -> Self {
        Self {
            effects: BTreeMap::new(),
            batching: false,
            batch_queue: Vec::new(),
        }
    }

    /// Schedule an effect. If not batching, it runs immediately if deps changed.
    pub fn schedule(&mut self, effect: Effect) {
        let id = effect.id;
        self.effects.insert(id, effect);
        if self.batching {
            if !self.batch_queue.contains(&id) {
                self.batch_queue.push(id);
            }
        } else {
            self.run_effect(id);
        }
    }

    /// Update the dependencies for an effect and re-run if they changed.
    pub fn update_deps(&mut self, id: EffectId, new_deps: Dependencies) {
        if let Some(effect) = self.effects.get_mut(&id) {
            effect.dependencies = Some(new_deps);
            if effect.deps_changed() {
                if self.batching {
                    if !self.batch_queue.contains(&id) {
                        self.batch_queue.push(id);
                    }
                } else {
                    // Need to use run_effect which borrows self
                    // So we do it after the borrow ends
                }
            }
        }
        // Re-check and run outside the borrow
        let should_run = self
            .effects
            .get(&id)
            .map_or(false, |e| e.deps_changed() && !self.batching);
        if should_run {
            self.run_effect(id);
        }
    }

    /// Run a specific effect if its dependencies changed.
    fn run_effect(&mut self, id: EffectId) {
        if let Some(effect) = self.effects.get_mut(&id) {
            if effect.cancelled {
                return;
            }
            if effect.deps_changed() {
                effect.execute();
            }
        }
    }

    /// Run all scheduled effects, ordered by priority.
    pub fn flush(&mut self) {
        let mut ids: Vec<(Priority, EffectId)> = self
            .effects
            .iter()
            .filter(|(_, e)| !e.cancelled && e.deps_changed())
            .map(|(&id, e)| (e.priority, id))
            .collect();
        ids.sort_by_key(|(pri, id)| (*pri, *id));

        for (_, id) in ids {
            self.run_effect(id);
        }
    }

    /// Cancel an effect, running its cleanup.
    pub fn cancel(&mut self, id: EffectId) {
        if let Some(effect) = self.effects.get_mut(&id) {
            effect.cancelled = true;
            effect.run_cleanup();
        }
    }

    /// Remove an effect entirely, running its cleanup.
    pub fn remove(&mut self, id: EffectId) {
        if let Some(mut effect) = self.effects.remove(&id) {
            effect.run_cleanup();
        }
    }

    /// Begin batching — effects scheduled during this period are deferred.
    pub fn begin_batch(&mut self) {
        self.batching = true;
        self.batch_queue.clear();
    }

    /// End batching and run all queued effects in priority order.
    pub fn end_batch(&mut self) {
        self.batching = false;
        // Gather queued effects with priorities
        let mut queued: Vec<(Priority, EffectId)> = self
            .batch_queue
            .drain(..)
            .filter_map(|id| {
                self.effects.get(&id).map(|e| (e.priority, id))
            })
            .collect();
        queued.sort_by_key(|(pri, id)| (*pri, *id));

        for (_, id) in queued {
            self.run_effect(id);
        }
    }

    /// Number of registered effects.
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Get the run count of a specific effect.
    pub fn run_count(&self, id: EffectId) -> u64 {
        self.effects.get(&id).map_or(0, |e| e.run_count)
    }

    /// Check whether a specific effect is cancelled.
    pub fn is_cancelled(&self, id: EffectId) -> bool {
        self.effects.get(&id).map_or(false, |e| e.cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn basic_effect_runs() {
        let counter = Rc::new(Cell::new(0));
        let c = counter.clone();
        let mut sched = EffectScheduler::new();
        sched.schedule(Effect::new(1, None, move || {
            c.set(c.get() + 1);
            None
        }));
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn effect_cleanup_on_rerun() {
        let log = Rc::new(RefCell::new(Vec::<String>::new()));
        let l1 = log.clone();
        let l2 = log.clone();
        use std::cell::RefCell;

        let mut sched = EffectScheduler::new();
        let deps = vec!["v1".to_string()];
        sched.schedule(Effect::new(1, Some(deps), move || {
            l1.borrow_mut().push("run".to_string());
            let l = l2.clone();
            Some(Box::new(move || {
                l.borrow_mut().push("cleanup".to_string());
            }) as Box<dyn FnOnce()>)
        }));

        assert_eq!(*log.borrow(), vec!["run"]);

        // Update deps to trigger re-run
        sched.update_deps(1, vec!["v2".to_string()]);
        assert_eq!(*log.borrow(), vec!["run", "cleanup", "run"]);
    }

    #[test]
    fn one_time_effect() {
        let counter = Rc::new(Cell::new(0));
        let c = counter.clone();
        let mut sched = EffectScheduler::new();
        sched.schedule(Effect::once(1, move || {
            c.set(c.get() + 1);
        }));
        assert_eq!(counter.get(), 1);

        // Flushing again should not re-run (empty deps, already ran once)
        sched.flush();
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn deps_unchanged_skips_run() {
        let counter = Rc::new(Cell::new(0));
        let c = counter.clone();
        let mut sched = EffectScheduler::new();
        let deps = vec!["stable".to_string()];
        sched.schedule(Effect::new(1, Some(deps), move || {
            c.set(c.get() + 1);
            None
        }));
        assert_eq!(counter.get(), 1);

        // Same deps — should not re-run
        sched.update_deps(1, vec!["stable".to_string()]);
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn effect_priority_ordering() {
        use std::cell::RefCell;
        let order = Rc::new(RefCell::new(Vec::<&str>::new()));

        let mut sched = EffectScheduler::new();

        let o1 = order.clone();
        sched.schedule(Effect::new(1, None, move || {
            o1.borrow_mut().push("low");
            None
        }).with_priority(10));

        let o2 = order.clone();
        sched.schedule(Effect::new(2, None, move || {
            o2.borrow_mut().push("high");
            None
        }).with_priority(1));

        // Both ran on schedule (not batched). Clear and use flush.
        order.borrow_mut().clear();

        // Mark both as needing re-run by setting None deps (always run)
        // We need to re-schedule with fresh deps
        let o3 = order.clone();
        let o4 = order.clone();
        let mut sched2 = EffectScheduler::new();
        sched2.begin_batch();
        sched2.schedule(Effect::new(1, None, move || {
            o3.borrow_mut().push("low");
            None
        }).with_priority(10));
        sched2.schedule(Effect::new(2, None, move || {
            o4.borrow_mut().push("high");
            None
        }).with_priority(1));
        sched2.end_batch();

        assert_eq!(*order.borrow(), vec!["high", "low"]);
    }

    #[test]
    fn cancel_effect() {
        let counter = Rc::new(Cell::new(0));
        let c = counter.clone();
        let mut sched = EffectScheduler::new();
        sched.schedule(Effect::new(1, None, move || {
            c.set(c.get() + 1);
            None
        }));
        assert_eq!(counter.get(), 1);

        sched.cancel(1);
        assert!(sched.is_cancelled(1));

        // Flushing should not run cancelled effect
        sched.flush();
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn cancel_runs_cleanup() {
        use std::cell::RefCell;
        let log = Rc::new(RefCell::new(Vec::<String>::new()));
        let l = log.clone();
        let mut sched = EffectScheduler::new();
        sched.schedule(Effect::new(1, None, move || {
            let ll = l.clone();
            Some(Box::new(move || {
                ll.borrow_mut().push("cleaned".to_string());
            }) as Box<dyn FnOnce()>)
        }));
        assert!(log.borrow().is_empty());
        sched.cancel(1);
        assert_eq!(*log.borrow(), vec!["cleaned"]);
    }

    #[test]
    fn remove_effect() {
        let mut sched = EffectScheduler::new();
        sched.schedule(Effect::new(1, None, || None));
        assert_eq!(sched.len(), 1);
        sched.remove(1);
        assert_eq!(sched.len(), 0);
    }

    #[test]
    fn batched_execution() {
        let counter = Rc::new(Cell::new(0));
        let mut sched = EffectScheduler::new();

        sched.begin_batch();

        let c = counter.clone();
        sched.schedule(Effect::new(1, None, move || {
            c.set(c.get() + 1);
            None
        }));

        let c2 = counter.clone();
        sched.schedule(Effect::new(2, None, move || {
            c2.set(c2.get() + 10);
            None
        }));

        // Nothing ran yet
        assert_eq!(counter.get(), 0);

        sched.end_batch();
        assert_eq!(counter.get(), 11);
    }

    #[test]
    fn run_count_tracking() {
        let mut sched = EffectScheduler::new();
        sched.schedule(Effect::new(1, None, || None));
        assert_eq!(sched.run_count(1), 1);

        // Re-run via flush (None deps = always dirty)
        sched.flush();
        assert_eq!(sched.run_count(1), 2);
    }

    #[test]
    fn empty_scheduler() {
        let sched = EffectScheduler::new();
        assert!(sched.is_empty());
        assert_eq!(sched.len(), 0);
        assert_eq!(sched.run_count(999), 0);
        assert!(!sched.is_cancelled(999));
    }
}
