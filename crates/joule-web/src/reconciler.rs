//! React-like reconciler — fiber work units, work loop with priority, update
//! queue, effect list, commit phase, batched updates, and concurrent mode simulation.
//!
//! Replaces React Fiber / Preact reconciler with a pure-Rust implementation.
//! Models work as fiber nodes in a linked tree, processes them in a priority-
//! driven work loop, batches state updates, and commits DOM changes in a
//! separate phase.

use std::collections::{HashMap, VecDeque};

// ── Priority ────────────────────────────────────────────────────────────

/// Update priority levels (lower numeric = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Immediate — synchronous, user-blocking.
    Immediate = 0,
    /// User interaction (click, key).
    UserBlocking = 1,
    /// Normal rendering priority.
    Normal = 2,
    /// Low — can be deferred.
    Low = 3,
    /// Idle — only when nothing else is pending.
    Idle = 4,
}

impl Priority {
    /// Time slice budget in milliseconds for this priority.
    pub fn time_slice_ms(&self) -> u64 {
        match self {
            Priority::Immediate => u64::MAX,
            Priority::UserBlocking => 250,
            Priority::Normal => 50,
            Priority::Low => 25,
            Priority::Idle => 5,
        }
    }
}

// ── Fiber node ──────────────────────────────────────────────────────────

/// The type of work a fiber represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberTag {
    Host,
    Component,
    Text,
    Root,
}

/// Effect flags for the commit phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectTag {
    Placement,
    Update,
    Deletion,
    None,
}

/// A fiber node — the unit of work in the reconciler.
#[derive(Debug, Clone)]
pub struct Fiber {
    pub id: u64,
    pub tag: FiberTag,
    pub fiber_type: String,
    pub props: HashMap<String, String>,
    pub state: HashMap<String, String>,
    pub effect_tag: EffectTag,
    pub child_id: Option<u64>,
    pub sibling_id: Option<u64>,
    pub parent_id: Option<u64>,
    pub alternate_id: Option<u64>,
    pub priority: Priority,
    pub processed: bool,
}

impl Fiber {
    pub fn new(id: u64, tag: FiberTag, fiber_type: &str) -> Self {
        Self {
            id,
            tag,
            fiber_type: fiber_type.to_string(),
            props: HashMap::new(),
            state: HashMap::new(),
            effect_tag: EffectTag::None,
            child_id: None,
            sibling_id: None,
            parent_id: None,
            alternate_id: None,
            priority: Priority::Normal,
            processed: false,
        }
    }

    pub fn with_props(mut self, key: &str, value: &str) -> Self {
        self.props.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_effect(mut self, effect: EffectTag) -> Self {
        self.effect_tag = effect;
        self
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
}

// ── Update queue ────────────────────────────────────────────────────────

/// A pending state update.
#[derive(Debug, Clone)]
pub struct StateUpdate {
    pub fiber_id: u64,
    pub key: String,
    pub value: String,
    pub priority: Priority,
    pub batch_id: Option<u64>,
}

impl StateUpdate {
    pub fn new(fiber_id: u64, key: &str, value: &str) -> Self {
        Self {
            fiber_id,
            key: key.to_string(),
            value: value.to_string(),
            priority: Priority::Normal,
            batch_id: None,
        }
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_batch(mut self, batch_id: u64) -> Self {
        self.batch_id = Some(batch_id);
        self
    }
}

// ── Commit record ───────────────────────────────────────────────────────

/// A DOM mutation produced during the commit phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitOp {
    pub fiber_id: u64,
    pub effect: EffectTag,
    pub fiber_type: String,
}

// ── Reconciler ──────────────────────────────────────────────────────────

/// The reconciler — manages fibers, update queue, work loop, and commit.
pub struct Reconciler {
    fibers: HashMap<u64, Fiber>,
    root_id: Option<u64>,
    update_queue: VecDeque<StateUpdate>,
    effect_list: Vec<u64>,
    commit_log: Vec<CommitOp>,
    next_fiber_id: u64,
    current_batch_id: Option<u64>,
    next_batch_id: u64,
    work_done_count: u64,
    is_batching: bool,
}

impl Reconciler {
    pub fn new() -> Self {
        Self {
            fibers: HashMap::new(),
            root_id: None,
            update_queue: VecDeque::new(),
            effect_list: Vec::new(),
            commit_log: Vec::new(),
            next_fiber_id: 1,
            current_batch_id: None,
            next_batch_id: 1,
            work_done_count: 0,
            is_batching: false,
        }
    }

    /// Create a fiber and add it to the tree. Returns the fiber ID.
    pub fn create_fiber(&mut self, tag: FiberTag, fiber_type: &str) -> u64 {
        let id = self.next_fiber_id;
        self.next_fiber_id += 1;
        let fiber = Fiber::new(id, tag, fiber_type);
        self.fibers.insert(id, fiber);
        id
    }

    /// Set the root fiber.
    pub fn set_root(&mut self, id: u64) {
        self.root_id = Some(id);
    }

    /// Get a fiber by ID.
    pub fn get_fiber(&self, id: u64) -> Option<&Fiber> {
        self.fibers.get(&id)
    }

    /// Get a mutable fiber by ID.
    pub fn get_fiber_mut(&mut self, id: u64) -> Option<&mut Fiber> {
        self.fibers.get_mut(&id)
    }

    /// Set the child of a fiber.
    pub fn set_child(&mut self, parent_id: u64, child_id: u64) {
        if let Some(parent) = self.fibers.get_mut(&parent_id) {
            parent.child_id = Some(child_id);
        }
        if let Some(child) = self.fibers.get_mut(&child_id) {
            child.parent_id = Some(parent_id);
        }
    }

    /// Set the sibling of a fiber.
    pub fn set_sibling(&mut self, fiber_id: u64, sibling_id: u64) {
        if let Some(fiber) = self.fibers.get_mut(&fiber_id) {
            fiber.sibling_id = Some(sibling_id);
        }
    }

    /// Set the effect tag on a fiber and add it to the effect list.
    pub fn mark_effect(&mut self, fiber_id: u64, effect: EffectTag) {
        if let Some(fiber) = self.fibers.get_mut(&fiber_id) {
            fiber.effect_tag = effect;
        }
        if effect != EffectTag::None && !self.effect_list.contains(&fiber_id) {
            self.effect_list.push(fiber_id);
        }
    }

    // ── Update queue ────────────────────────────────────────────────────

    /// Enqueue a state update.
    pub fn enqueue_update(&mut self, mut update: StateUpdate) {
        if self.is_batching {
            update.batch_id = self.current_batch_id;
        }
        self.update_queue.push_back(update);
    }

    /// Start batching updates.
    pub fn start_batch(&mut self) -> u64 {
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;
        self.current_batch_id = Some(batch_id);
        self.is_batching = true;
        batch_id
    }

    /// End batching and return the batch ID.
    pub fn end_batch(&mut self) -> Option<u64> {
        let batch = self.current_batch_id.take();
        self.is_batching = false;
        batch
    }

    /// Process pending updates — apply state changes to fibers.
    pub fn process_updates(&mut self) -> usize {
        let mut processed = 0;
        while let Some(update) = self.update_queue.pop_front() {
            if let Some(fiber) = self.fibers.get_mut(&update.fiber_id) {
                fiber.state.insert(update.key, update.value);
                fiber.effect_tag = EffectTag::Update;
                if !self.effect_list.contains(&update.fiber_id) {
                    self.effect_list.push(update.fiber_id);
                }
            }
            processed += 1;
        }
        processed
    }

    /// Number of pending updates.
    pub fn pending_updates(&self) -> usize {
        self.update_queue.len()
    }

    // ── Work loop ───────────────────────────────────────────────────────

    /// Perform one unit of work on a fiber. Returns the next fiber ID to process.
    pub fn perform_unit_of_work(&mut self, fiber_id: u64) -> Option<u64> {
        if let Some(fiber) = self.fibers.get_mut(&fiber_id) {
            fiber.processed = true;
            self.work_done_count += 1;
        }

        let (child_id, sibling_id, parent_id) = {
            let fiber = self.fibers.get(&fiber_id)?;
            (fiber.child_id, fiber.sibling_id, fiber.parent_id)
        };

        // Walk: child first, then sibling, then uncle (parent's sibling), etc.
        if let Some(cid) = child_id {
            return Some(cid);
        }

        let mut current = fiber_id;
        loop {
            let (sib, par) = {
                let f = self.fibers.get(&current)?;
                (f.sibling_id, f.parent_id)
            };
            if let Some(sid) = sib {
                return Some(sid);
            }
            match par {
                Some(pid) => current = pid,
                None => return None,
            }
        }
    }

    /// Run the work loop starting from a fiber, with a work unit budget.
    pub fn work_loop(&mut self, start_id: u64, max_units: usize) -> usize {
        let mut units = 0;
        let mut next = Some(start_id);

        while let Some(fid) = next {
            if units >= max_units {
                break;
            }
            next = self.perform_unit_of_work(fid);
            units += 1;
        }

        units
    }

    /// Run the work loop with priority-based time slicing.
    pub fn work_loop_with_priority(
        &mut self,
        start_id: u64,
        priority: Priority,
    ) -> usize {
        // Convert time slice to a work unit budget (approximation)
        let max_units = match priority {
            Priority::Immediate => usize::MAX,
            Priority::UserBlocking => 100,
            Priority::Normal => 50,
            Priority::Low => 25,
            Priority::Idle => 10,
        };
        self.work_loop(start_id, max_units)
    }

    // ── Commit phase ────────────────────────────────────────────────────

    /// Commit the effect list — produce DOM mutation operations.
    pub fn commit(&mut self) -> Vec<CommitOp> {
        let mut ops = Vec::new();

        let effect_ids: Vec<u64> = self.effect_list.drain(..).collect();
        for fid in effect_ids {
            if let Some(fiber) = self.fibers.get(&fid) {
                if fiber.effect_tag != EffectTag::None {
                    ops.push(CommitOp {
                        fiber_id: fid,
                        effect: fiber.effect_tag,
                        fiber_type: fiber.fiber_type.clone(),
                    });
                }
            }
            // Reset effect tag after commit
            if let Some(fiber) = self.fibers.get_mut(&fid) {
                fiber.effect_tag = EffectTag::None;
            }
        }

        self.commit_log.extend(ops.clone());
        ops
    }

    /// Get the full commit log.
    pub fn commit_log(&self) -> &[CommitOp] {
        &self.commit_log
    }

    /// Total work units processed.
    pub fn work_done_count(&self) -> u64 {
        self.work_done_count
    }

    /// Total fiber count.
    pub fn fiber_count(&self) -> usize {
        self.fibers.len()
    }

    /// Get the effect list (fiber IDs with pending effects).
    pub fn effect_list(&self) -> &[u64] {
        &self.effect_list
    }

    /// Remove a fiber (deletion).
    pub fn delete_fiber(&mut self, id: u64) {
        self.mark_effect(id, EffectTag::Deletion);
    }
}

impl Default for Reconciler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_tree() -> Reconciler {
        let mut r = Reconciler::new();
        let root = r.create_fiber(FiberTag::Root, "root");
        let div = r.create_fiber(FiberTag::Host, "div");
        let p = r.create_fiber(FiberTag::Host, "p");
        let span = r.create_fiber(FiberTag::Host, "span");

        r.set_root(root);
        r.set_child(root, div);
        r.set_child(div, p);
        r.set_sibling(p, span);

        r
    }

    #[test]
    fn create_fiber_assigns_ids() {
        let mut r = Reconciler::new();
        let a = r.create_fiber(FiberTag::Host, "div");
        let b = r.create_fiber(FiberTag::Host, "span");
        assert_ne!(a, b);
    }

    #[test]
    fn fiber_count() {
        let r = setup_tree();
        assert_eq!(r.fiber_count(), 4);
    }

    #[test]
    fn set_child_parent_link() {
        let r = setup_tree();
        let root = r.get_fiber(1).unwrap();
        assert_eq!(root.child_id, Some(2));
        let div = r.get_fiber(2).unwrap();
        assert_eq!(div.parent_id, Some(1));
    }

    #[test]
    fn set_sibling_link() {
        let r = setup_tree();
        let p = r.get_fiber(3).unwrap();
        assert_eq!(p.sibling_id, Some(4));
    }

    #[test]
    fn work_loop_traverses_tree() {
        let mut r = setup_tree();
        let units = r.work_loop(1, 100);
        assert_eq!(units, 4);

        // All fibers should be processed
        for id in 1..=4 {
            assert!(r.get_fiber(id).unwrap().processed);
        }
    }

    #[test]
    fn work_loop_respects_budget() {
        let mut r = setup_tree();
        let units = r.work_loop(1, 2);
        assert_eq!(units, 2);
        assert_eq!(r.work_done_count(), 2);
    }

    #[test]
    fn enqueue_and_process_updates() {
        let mut r = setup_tree();
        r.enqueue_update(StateUpdate::new(2, "count", "5"));
        r.enqueue_update(StateUpdate::new(2, "name", "hello"));
        assert_eq!(r.pending_updates(), 2);

        let processed = r.process_updates();
        assert_eq!(processed, 2);
        assert_eq!(r.pending_updates(), 0);

        let fiber = r.get_fiber(2).unwrap();
        assert_eq!(fiber.state.get("count").unwrap(), "5");
        assert_eq!(fiber.state.get("name").unwrap(), "hello");
    }

    #[test]
    fn batched_updates() {
        let mut r = setup_tree();
        let batch = r.start_batch();
        r.enqueue_update(StateUpdate::new(2, "a", "1"));
        r.enqueue_update(StateUpdate::new(2, "b", "2"));
        let ended = r.end_batch();
        assert_eq!(ended, Some(batch));

        // All updates should share the batch ID
        let updates: Vec<_> = r.update_queue.iter().collect();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].batch_id, Some(batch));
        assert_eq!(updates[1].batch_id, Some(batch));
    }

    #[test]
    fn mark_effect_adds_to_list() {
        let mut r = setup_tree();
        r.mark_effect(2, EffectTag::Update);
        r.mark_effect(3, EffectTag::Placement);
        assert_eq!(r.effect_list().len(), 2);
    }

    #[test]
    fn commit_produces_ops() {
        let mut r = setup_tree();
        r.mark_effect(2, EffectTag::Update);
        r.mark_effect(3, EffectTag::Placement);

        let ops = r.commit();
        assert_eq!(ops.len(), 2);

        // Effect list should be cleared after commit
        assert!(r.effect_list().is_empty());
    }

    #[test]
    fn commit_resets_effect_tags() {
        let mut r = setup_tree();
        r.mark_effect(2, EffectTag::Update);
        r.commit();

        let fiber = r.get_fiber(2).unwrap();
        assert_eq!(fiber.effect_tag, EffectTag::None);
    }

    #[test]
    fn commit_log_accumulated() {
        let mut r = setup_tree();
        r.mark_effect(2, EffectTag::Placement);
        r.commit();
        r.mark_effect(3, EffectTag::Update);
        r.commit();

        assert_eq!(r.commit_log().len(), 2);
    }

    #[test]
    fn delete_fiber_marks_deletion() {
        let mut r = setup_tree();
        r.delete_fiber(3);
        let fiber = r.get_fiber(3).unwrap();
        assert_eq!(fiber.effect_tag, EffectTag::Deletion);
        assert!(r.effect_list().contains(&3));
    }

    #[test]
    fn priority_ordering() {
        assert!(Priority::Immediate < Priority::UserBlocking);
        assert!(Priority::UserBlocking < Priority::Normal);
        assert!(Priority::Normal < Priority::Low);
        assert!(Priority::Low < Priority::Idle);
    }

    #[test]
    fn priority_time_slices() {
        assert!(Priority::Immediate.time_slice_ms() > Priority::Normal.time_slice_ms());
        assert!(Priority::Normal.time_slice_ms() > Priority::Idle.time_slice_ms());
    }

    #[test]
    fn work_loop_with_priority() {
        let mut r = setup_tree();
        let units = r.work_loop_with_priority(1, Priority::Normal);
        assert_eq!(units, 4); // tree has 4 nodes, budget is 50
    }

    #[test]
    fn fiber_props() {
        let mut r = Reconciler::new();
        let id = r.create_fiber(FiberTag::Host, "button");
        if let Some(fiber) = r.get_fiber_mut(id) {
            fiber.props.insert("label".to_string(), "Click".to_string());
        }
        let fiber = r.get_fiber(id).unwrap();
        assert_eq!(fiber.props.get("label").unwrap(), "Click");
    }

    #[test]
    fn update_with_priority() {
        let mut r = setup_tree();
        let update = StateUpdate::new(2, "x", "1").with_priority(Priority::Immediate);
        r.enqueue_update(update);

        let u = &r.update_queue[0];
        assert_eq!(u.priority, Priority::Immediate);
    }

    #[test]
    fn no_effect_produces_no_commit_op() {
        let mut r = setup_tree();
        r.mark_effect(2, EffectTag::None);
        let ops = r.commit();
        assert!(ops.is_empty());
    }

    #[test]
    fn process_updates_marks_fiber_for_update() {
        let mut r = setup_tree();
        r.enqueue_update(StateUpdate::new(2, "k", "v"));
        r.process_updates();

        assert!(r.effect_list().contains(&2));
        let fiber = r.get_fiber(2).unwrap();
        assert_eq!(fiber.effect_tag, EffectTag::Update);
    }
}
