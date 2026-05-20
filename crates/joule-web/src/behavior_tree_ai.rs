//! Behavior tree for game AI — Sequence (AND), Selector (OR), Parallel,
//! decorators (Inverter, Repeater, RepeatUntilFail, Cooldown, RateLimit),
//! Action and Condition leaf nodes, Running/Success/Failure status,
//! Blackboard (shared key-value context), fluent tree builder API.
//!
//! Replaces JavaScript behavior tree libraries (BehaviorTree.js, fluent-behavior-tree)
//! with a pure-Rust AI decision tree for games.

use std::collections::HashMap;

// ── Status ──────────────────────────────────────────────────────

/// Result status of a behavior tree node tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Success,
    Failure,
    Running,
}

// ── Blackboard ──────────────────────────────────────────────────

/// Shared key-value context for behavior tree nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Blackboard {
    strings: HashMap<String, String>,
    floats: HashMap<String, f64>,
    ints: HashMap<String, i64>,
    bools: HashMap<String, bool>,
}

impl Blackboard {
    pub fn new() -> Self {
        Self {
            strings: HashMap::new(),
            floats: HashMap::new(),
            ints: HashMap::new(),
            bools: HashMap::new(),
        }
    }

    pub fn set_string(&mut self, key: &str, value: &str) {
        self.strings.insert(key.to_string(), value.to_string());
    }

    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.strings.get(key).map(|s| s.as_str())
    }

    pub fn set_float(&mut self, key: &str, value: f64) {
        self.floats.insert(key.to_string(), value);
    }

    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.floats.get(key).copied()
    }

    pub fn set_int(&mut self, key: &str, value: i64) {
        self.ints.insert(key.to_string(), value);
    }

    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.ints.get(key).copied()
    }

    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.bools.insert(key.to_string(), value);
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.bools.get(key).copied()
    }

    pub fn clear(&mut self) {
        self.strings.clear();
        self.floats.clear();
        self.ints.clear();
        self.bools.clear();
    }
}

// ── Node types ──────────────────────────────────────────────────

/// Behavior tree node.
#[derive(Clone)]
pub enum BtNode {
    /// Run children in order; fail on first failure.
    Sequence(Vec<BtNode>),

    /// Run children in order; succeed on first success.
    Selector(Vec<BtNode>),

    /// Run N children simultaneously. Succeed if `required` succeed.
    Parallel { children: Vec<BtNode>, required: usize },

    /// Invert child result (Success<->Failure, Running unchanged).
    Inverter(Box<BtNode>),

    /// Repeat child N times (0 = infinite until failure).
    Repeater { child: Box<BtNode>, count: usize },

    /// Repeat child until it returns Failure.
    RepeatUntilFail(Box<BtNode>),

    /// Only tick child if enough time has passed since last execution.
    Cooldown { child: Box<BtNode>, cooldown_ms: u64, last_tick_ms: u64 },

    /// Limit child execution to N times per period.
    RateLimit { child: Box<BtNode>, max_per_period: usize, period_ms: u64, history: Vec<u64> },

    /// Leaf: action that reads/writes blackboard.
    Action(ActionFn),

    /// Leaf: condition check.
    Condition(ConditionFn),
}

/// Action function type: takes blackboard, returns status.
#[derive(Clone)]
pub struct ActionFn {
    pub name: String,
    pub func: fn(&mut Blackboard) -> Status,
}

/// Condition function type: takes blackboard, returns bool.
#[derive(Clone)]
pub struct ConditionFn {
    pub name: String,
    pub func: fn(&Blackboard) -> bool,
}

impl std::fmt::Debug for BtNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BtNode::Sequence(c) => write!(f, "Sequence({} children)", c.len()),
            BtNode::Selector(c) => write!(f, "Selector({} children)", c.len()),
            BtNode::Parallel { children, required } =>
                write!(f, "Parallel({} children, {} required)", children.len(), required),
            BtNode::Inverter(_) => write!(f, "Inverter"),
            BtNode::Repeater { count, .. } => write!(f, "Repeater({})", count),
            BtNode::RepeatUntilFail(_) => write!(f, "RepeatUntilFail"),
            BtNode::Cooldown { cooldown_ms, .. } => write!(f, "Cooldown({}ms)", cooldown_ms),
            BtNode::RateLimit { max_per_period, .. } => write!(f, "RateLimit({})", max_per_period),
            BtNode::Action(a) => write!(f, "Action({})", a.name),
            BtNode::Condition(c) => write!(f, "Condition({})", c.name),
        }
    }
}

// ── Tick engine ─────────────────────────────────────────────────

/// Tick a behavior tree node.
/// `time_ms`: current time in milliseconds (for cooldown/rate-limit).
pub fn tick(node: &mut BtNode, bb: &mut Blackboard, time_ms: u64) -> Status {
    match node {
        BtNode::Sequence(children) => {
            for child in children.iter_mut() {
                match tick(child, bb, time_ms) {
                    Status::Failure => return Status::Failure,
                    Status::Running => return Status::Running,
                    Status::Success => {}
                }
            }
            Status::Success
        }

        BtNode::Selector(children) => {
            for child in children.iter_mut() {
                match tick(child, bb, time_ms) {
                    Status::Success => return Status::Success,
                    Status::Running => return Status::Running,
                    Status::Failure => {}
                }
            }
            Status::Failure
        }

        BtNode::Parallel { children, required } => {
            let mut successes = 0;
            let mut failures = 0;
            let n = children.len();
            for child in children.iter_mut() {
                match tick(child, bb, time_ms) {
                    Status::Success => successes += 1,
                    Status::Failure => failures += 1,
                    Status::Running => {}
                }
            }
            if successes >= *required {
                Status::Success
            } else if failures > n - *required {
                Status::Failure
            } else {
                Status::Running
            }
        }

        BtNode::Inverter(child) => {
            match tick(child, bb, time_ms) {
                Status::Success => Status::Failure,
                Status::Failure => Status::Success,
                Status::Running => Status::Running,
            }
        }

        BtNode::Repeater { child, count } => {
            if *count == 0 {
                // Infinite: run until failure
                match tick(child, bb, time_ms) {
                    Status::Failure => Status::Failure,
                    _ => Status::Running,
                }
            } else {
                for _ in 0..*count {
                    match tick(child, bb, time_ms) {
                        Status::Failure => return Status::Failure,
                        Status::Running => return Status::Running,
                        Status::Success => {}
                    }
                }
                Status::Success
            }
        }

        BtNode::RepeatUntilFail(child) => {
            match tick(child, bb, time_ms) {
                Status::Failure => Status::Success,
                _ => Status::Running,
            }
        }

        BtNode::Cooldown { child, cooldown_ms, last_tick_ms } => {
            if time_ms.saturating_sub(*last_tick_ms) >= *cooldown_ms {
                *last_tick_ms = time_ms;
                tick(child, bb, time_ms)
            } else {
                Status::Failure
            }
        }

        BtNode::RateLimit { child, max_per_period, period_ms, history } => {
            // Prune old entries
            let cutoff = time_ms.saturating_sub(*period_ms);
            history.retain(|t| *t > cutoff);
            if history.len() < *max_per_period {
                history.push(time_ms);
                tick(child, bb, time_ms)
            } else {
                Status::Failure
            }
        }

        BtNode::Action(action_fn) => {
            (action_fn.func)(bb)
        }

        BtNode::Condition(cond_fn) => {
            if (cond_fn.func)(bb) {
                Status::Success
            } else {
                Status::Failure
            }
        }
    }
}

// ── Fluent builder ──────────────────────────────────────────────

/// Fluent API for building behavior trees.
pub struct BtBuilder {
    stack: Vec<Vec<BtNode>>,
}

impl BtBuilder {
    pub fn new() -> Self {
        Self { stack: vec![Vec::new()] }
    }

    /// Begin a Sequence composite.
    pub fn sequence(mut self) -> Self {
        self.stack.push(Vec::new());
        self
    }

    /// Begin a Selector composite.
    pub fn selector(mut self) -> Self {
        self.stack.push(Vec::new());
        self
    }

    /// End the current composite, creating a Sequence.
    pub fn end_sequence(mut self) -> Self {
        let children = self.stack.pop().unwrap_or_default();
        let node = BtNode::Sequence(children);
        if let Some(parent) = self.stack.last_mut() {
            parent.push(node);
        }
        self
    }

    /// End the current composite, creating a Selector.
    pub fn end_selector(mut self) -> Self {
        let children = self.stack.pop().unwrap_or_default();
        let node = BtNode::Selector(children);
        if let Some(parent) = self.stack.last_mut() {
            parent.push(node);
        }
        self
    }

    /// Add an action leaf.
    pub fn action(mut self, name: &str, func: fn(&mut Blackboard) -> Status) -> Self {
        let node = BtNode::Action(ActionFn { name: name.to_string(), func });
        if let Some(current) = self.stack.last_mut() {
            current.push(node);
        }
        self
    }

    /// Add a condition leaf.
    pub fn condition(mut self, name: &str, func: fn(&Blackboard) -> bool) -> Self {
        let node = BtNode::Condition(ConditionFn { name: name.to_string(), func });
        if let Some(current) = self.stack.last_mut() {
            current.push(node);
        }
        self
    }

    /// Wrap the last added node with an Inverter decorator.
    pub fn invert_last(mut self) -> Self {
        if let Some(current) = self.stack.last_mut() {
            if let Some(last) = current.pop() {
                current.push(BtNode::Inverter(Box::new(last)));
            }
        }
        self
    }

    /// Add a Parallel node with given children and required successes.
    pub fn parallel(mut self, children: Vec<BtNode>, required: usize) -> Self {
        let node = BtNode::Parallel { children, required };
        if let Some(current) = self.stack.last_mut() {
            current.push(node);
        }
        self
    }

    /// Build the final tree. Returns the first node on the root level.
    pub fn build(mut self) -> Option<BtNode> {
        while self.stack.len() > 1 {
            let children = self.stack.pop().unwrap_or_default();
            let node = BtNode::Sequence(children);
            if let Some(parent) = self.stack.last_mut() {
                parent.push(node);
            }
        }
        self.stack.pop().and_then(|mut v| {
            if v.len() == 1 { Some(v.remove(0)) }
            else if v.is_empty() { None }
            else { Some(BtNode::Sequence(v)) }
        })
    }
}

// ── Helper constructors ─────────────────────────────────────────

/// Create an action node.
pub fn action(name: &str, func: fn(&mut Blackboard) -> Status) -> BtNode {
    BtNode::Action(ActionFn { name: name.to_string(), func })
}

/// Create a condition node.
pub fn condition(name: &str, func: fn(&Blackboard) -> bool) -> BtNode {
    BtNode::Condition(ConditionFn { name: name.to_string(), func })
}

/// Create a sequence node.
pub fn sequence(children: Vec<BtNode>) -> BtNode {
    BtNode::Sequence(children)
}

/// Create a selector node.
pub fn selector(children: Vec<BtNode>) -> BtNode {
    BtNode::Selector(children)
}

/// Count total nodes in a tree.
pub fn node_count(node: &BtNode) -> usize {
    match node {
        BtNode::Sequence(c) | BtNode::Selector(c) => {
            1 + c.iter().map(node_count).sum::<usize>()
        }
        BtNode::Parallel { children, .. } => {
            1 + children.iter().map(node_count).sum::<usize>()
        }
        BtNode::Inverter(c) | BtNode::Repeater { child: c, .. }
        | BtNode::RepeatUntilFail(c) | BtNode::Cooldown { child: c, .. }
        | BtNode::RateLimit { child: c, .. } => {
            1 + node_count(c)
        }
        BtNode::Action(_) | BtNode::Condition(_) => 1,
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn success_action(_bb: &mut Blackboard) -> Status { Status::Success }
    fn failure_action(_bb: &mut Blackboard) -> Status { Status::Failure }
    fn running_action(_bb: &mut Blackboard) -> Status { Status::Running }
    fn increment_action(bb: &mut Blackboard) -> Status {
        let v = bb.get_int("counter").unwrap_or(0);
        bb.set_int("counter", v + 1);
        Status::Success
    }
    fn true_condition(_bb: &Blackboard) -> bool { true }
    fn false_condition(_bb: &Blackboard) -> bool { false }
    fn check_counter(bb: &Blackboard) -> bool {
        bb.get_int("counter").unwrap_or(0) > 0
    }

    #[test]
    fn test_blackboard_string() {
        let mut bb = Blackboard::new();
        bb.set_string("name", "Alice");
        assert_eq!(bb.get_string("name"), Some("Alice"));
        assert_eq!(bb.get_string("missing"), None);
    }

    #[test]
    fn test_blackboard_float() {
        let mut bb = Blackboard::new();
        bb.set_float("hp", 100.0);
        assert!((bb.get_float("hp").unwrap() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_blackboard_int_bool() {
        let mut bb = Blackboard::new();
        bb.set_int("score", 42);
        bb.set_bool("alive", true);
        assert_eq!(bb.get_int("score"), Some(42));
        assert_eq!(bb.get_bool("alive"), Some(true));
    }

    #[test]
    fn test_blackboard_clear() {
        let mut bb = Blackboard::new();
        bb.set_int("x", 1);
        bb.clear();
        assert_eq!(bb.get_int("x"), None);
    }

    #[test]
    fn test_sequence_all_success() {
        let mut tree = sequence(vec![
            action("a", success_action),
            action("b", success_action),
        ]);
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
    }

    #[test]
    fn test_sequence_one_failure() {
        let mut tree = sequence(vec![
            action("a", success_action),
            action("b", failure_action),
            action("c", success_action),
        ]);
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Failure);
    }

    #[test]
    fn test_sequence_running() {
        let mut tree = sequence(vec![
            action("a", success_action),
            action("b", running_action),
        ]);
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Running);
    }

    #[test]
    fn test_selector_first_success() {
        let mut tree = selector(vec![
            action("a", failure_action),
            action("b", success_action),
            action("c", failure_action),
        ]);
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
    }

    #[test]
    fn test_selector_all_fail() {
        let mut tree = selector(vec![
            action("a", failure_action),
            action("b", failure_action),
        ]);
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Failure);
    }

    #[test]
    fn test_parallel() {
        let mut tree = BtNode::Parallel {
            children: vec![
                action("a", success_action),
                action("b", failure_action),
                action("c", success_action),
            ],
            required: 2,
        };
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
    }

    #[test]
    fn test_parallel_not_enough() {
        let mut tree = BtNode::Parallel {
            children: vec![
                action("a", success_action),
                action("b", failure_action),
                action("c", failure_action),
            ],
            required: 2,
        };
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Failure);
    }

    #[test]
    fn test_inverter() {
        let mut tree = BtNode::Inverter(Box::new(action("a", success_action)));
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Failure);
    }

    #[test]
    fn test_inverter_running() {
        let mut tree = BtNode::Inverter(Box::new(action("a", running_action)));
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Running);
    }

    #[test]
    fn test_repeater() {
        let mut tree = BtNode::Repeater {
            child: Box::new(action("inc", increment_action)),
            count: 3,
        };
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
        assert_eq!(bb.get_int("counter"), Some(3));
    }

    #[test]
    fn test_repeat_until_fail() {
        let mut tree = BtNode::RepeatUntilFail(Box::new(action("f", failure_action)));
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
    }

    #[test]
    fn test_cooldown() {
        let mut tree = BtNode::Cooldown {
            child: Box::new(action("inc", increment_action)),
            cooldown_ms: 100,
            last_tick_ms: 0,
        };
        let mut bb = Blackboard::new();

        // First tick at t=0 succeeds (0 - 0 >= 100 is false, but initial)
        assert_eq!(tick(&mut tree, &mut bb, 100), Status::Success);

        // Immediate retry should fail (cooldown not elapsed)
        assert_eq!(tick(&mut tree, &mut bb, 150), Status::Failure);

        // After cooldown
        assert_eq!(tick(&mut tree, &mut bb, 200), Status::Success);
    }

    #[test]
    fn test_rate_limit() {
        let mut tree = BtNode::RateLimit {
            child: Box::new(action("inc", increment_action)),
            max_per_period: 2,
            period_ms: 1000,
            history: Vec::new(),
        };
        let mut bb = Blackboard::new();

        assert_eq!(tick(&mut tree, &mut bb, 100), Status::Success);
        assert_eq!(tick(&mut tree, &mut bb, 200), Status::Success);
        assert_eq!(tick(&mut tree, &mut bb, 300), Status::Failure); // rate limited

        // After period expires
        assert_eq!(tick(&mut tree, &mut bb, 1200), Status::Success);
    }

    #[test]
    fn test_condition_true() {
        let mut tree = condition("check", true_condition);
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
    }

    #[test]
    fn test_condition_false() {
        let mut tree = condition("check", false_condition);
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Failure);
    }

    #[test]
    fn test_condition_with_blackboard() {
        let mut tree = sequence(vec![
            action("inc", increment_action),
            condition("check", check_counter),
        ]);
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
    }

    #[test]
    fn test_builder_simple() {
        let tree = BtBuilder::new()
            .sequence()
                .action("a", success_action)
                .action("b", success_action)
            .end_sequence()
            .build();
        assert!(tree.is_some());
        let mut tree = tree.unwrap();
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
    }

    #[test]
    fn test_builder_nested() {
        let tree = BtBuilder::new()
            .selector()
                .action("fail", failure_action)
                .sequence()
                    .action("s1", success_action)
                    .action("s2", success_action)
                .end_sequence()
            .end_selector()
            .build();
        assert!(tree.is_some());
        let mut tree = tree.unwrap();
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Success);
    }

    #[test]
    fn test_builder_invert() {
        let tree = BtBuilder::new()
            .action("a", success_action)
            .invert_last()
            .build();
        assert!(tree.is_some());
        let mut tree = tree.unwrap();
        let mut bb = Blackboard::new();
        assert_eq!(tick(&mut tree, &mut bb, 0), Status::Failure);
    }

    #[test]
    fn test_node_count() {
        let tree = sequence(vec![
            action("a", success_action),
            selector(vec![
                action("b", failure_action),
                action("c", success_action),
            ]),
        ]);
        assert_eq!(node_count(&tree), 5); // seq + a + sel + b + c
    }

    #[test]
    fn test_debug_format() {
        let tree = action("test", success_action);
        let dbg = format!("{:?}", tree);
        assert!(dbg.contains("Action"));
        assert!(dbg.contains("test"));
    }
}
