//! Behavior trees — sequence, selector, parallel, decorator, action, condition
//! nodes with blackboard shared state, subtree reuse, and priority support.
//!
//! Replaces JS behavior tree libraries (BehaviorTree.js, fluent-behavior-tree)
//! with a pure-Rust tree executor for game AI.

use std::collections::HashMap;

// ── Status ──────────────────────────────────────────────────────

/// Result of ticking a behavior tree node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Node completed successfully.
    Success,
    /// Node failed.
    Failure,
    /// Node still running, needs more ticks.
    Running,
}

// ── Blackboard ──────────────────────────────────────────────────

/// Shared key-value store for tree nodes to communicate.
#[derive(Debug, Clone)]
pub struct Blackboard {
    ints: HashMap<String, i64>,
    floats: HashMap<String, f64>,
    strings: HashMap<String, String>,
    bools: HashMap<String, bool>,
}

impl Blackboard {
    pub fn new() -> Self {
        Self {
            ints: HashMap::new(),
            floats: HashMap::new(),
            strings: HashMap::new(),
            bools: HashMap::new(),
        }
    }

    pub fn set_int(&mut self, key: &str, val: i64) { self.ints.insert(key.to_string(), val); }
    pub fn get_int(&self, key: &str) -> Option<i64> { self.ints.get(key).copied() }
    pub fn set_float(&mut self, key: &str, val: f64) { self.floats.insert(key.to_string(), val); }
    pub fn get_float(&self, key: &str) -> Option<f64> { self.floats.get(key).copied() }
    pub fn set_string(&mut self, key: &str, val: &str) { self.strings.insert(key.to_string(), val.to_string()); }
    pub fn get_string(&self, key: &str) -> Option<&str> { self.strings.get(key).map(|s| s.as_str()) }
    pub fn set_bool(&mut self, key: &str, val: bool) { self.bools.insert(key.to_string(), val); }
    pub fn get_bool(&self, key: &str) -> Option<bool> { self.bools.get(key).copied() }
    pub fn clear(&mut self) {
        self.ints.clear();
        self.floats.clear();
        self.strings.clear();
        self.bools.clear();
    }
}

impl Default for Blackboard {
    fn default() -> Self { Self::new() }
}

// ── Decorator kind ──────────────────────────────────────────────

/// Decorator transforms the child's status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoratorKind {
    /// Invert success/failure, pass through running.
    Inverter,
    /// Always return success (unless running).
    AlwaysSucceed,
    /// Always return failure (unless running).
    AlwaysFail,
    /// Repeat child N times (stored in repeat_count).
    Repeat,
    /// Retry on failure up to N times.
    RetryOnFailure,
}

// ── Parallel policy ─────────────────────────────────────────────

/// How a parallel node determines its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelPolicy {
    /// Succeed when ALL children succeed.
    RequireAll,
    /// Succeed when ANY child succeeds.
    RequireOne,
}

// ── Node ────────────────────────────────────────────────────────

/// A behavior tree node.
#[derive(Debug, Clone)]
pub enum Node {
    /// Run children in order; fail on first failure.
    Sequence {
        name: String,
        children: Vec<Node>,
    },
    /// Try children in order; succeed on first success.
    Selector {
        name: String,
        children: Vec<Node>,
    },
    /// Run all children each tick.
    Parallel {
        name: String,
        policy: ParallelPolicy,
        children: Vec<Node>,
    },
    /// Modify child result.
    Decorator {
        name: String,
        kind: DecoratorKind,
        repeat_count: u32,
        child: Box<Node>,
    },
    /// Leaf node that performs an action via a named function.
    Action {
        name: String,
    },
    /// Leaf node that checks a condition on the blackboard.
    Condition {
        name: String,
        /// Key in the blackboard bools map.
        key: String,
        /// Expected value.
        expected: bool,
    },
}

impl Node {
    pub fn sequence(name: &str, children: Vec<Node>) -> Self {
        Self::Sequence { name: name.to_string(), children }
    }

    pub fn selector(name: &str, children: Vec<Node>) -> Self {
        Self::Selector { name: name.to_string(), children }
    }

    pub fn parallel(name: &str, policy: ParallelPolicy, children: Vec<Node>) -> Self {
        Self::Parallel { name: name.to_string(), policy, children }
    }

    pub fn decorator(name: &str, kind: DecoratorKind, child: Node) -> Self {
        Self::Decorator {
            name: name.to_string(),
            kind,
            repeat_count: 1,
            child: Box::new(child),
        }
    }

    pub fn repeat(name: &str, count: u32, child: Node) -> Self {
        Self::Decorator {
            name: name.to_string(),
            kind: DecoratorKind::Repeat,
            repeat_count: count,
            child: Box::new(child),
        }
    }

    pub fn retry(name: &str, max_retries: u32, child: Node) -> Self {
        Self::Decorator {
            name: name.to_string(),
            kind: DecoratorKind::RetryOnFailure,
            repeat_count: max_retries,
            child: Box::new(child),
        }
    }

    pub fn action(name: &str) -> Self {
        Self::Action { name: name.to_string() }
    }

    pub fn condition(name: &str, key: &str, expected: bool) -> Self {
        Self::Condition {
            name: name.to_string(),
            key: key.to_string(),
            expected,
        }
    }

    /// Get the node name.
    pub fn name(&self) -> &str {
        match self {
            Self::Sequence { name, .. }
            | Self::Selector { name, .. }
            | Self::Parallel { name, .. }
            | Self::Decorator { name, .. }
            | Self::Action { name }
            | Self::Condition { name, .. } => name,
        }
    }
}

// ── Tree executor ───────────────────────────────────────────────

/// Action handler: maps action names to functions.
pub type ActionFn = fn(&mut Blackboard) -> Status;

/// Behavior tree executor.
pub struct BehaviorTree {
    root: Node,
    actions: HashMap<String, ActionFn>,
}

impl BehaviorTree {
    /// Create a new tree with the given root node.
    pub fn new(root: Node) -> Self {
        Self { root, actions: HashMap::new() }
    }

    /// Register an action handler.
    pub fn register_action(&mut self, name: &str, handler: ActionFn) {
        self.actions.insert(name.to_string(), handler);
    }

    /// Tick the tree.
    pub fn tick(&self, bb: &mut Blackboard) -> Status {
        self.tick_node(&self.root, bb)
    }

    fn tick_node(&self, node: &Node, bb: &mut Blackboard) -> Status {
        match node {
            Node::Sequence { children, .. } => {
                for child in children {
                    match self.tick_node(child, bb) {
                        Status::Failure => return Status::Failure,
                        Status::Running => return Status::Running,
                        Status::Success => {}
                    }
                }
                Status::Success
            }
            Node::Selector { children, .. } => {
                for child in children {
                    match self.tick_node(child, bb) {
                        Status::Success => return Status::Success,
                        Status::Running => return Status::Running,
                        Status::Failure => {}
                    }
                }
                Status::Failure
            }
            Node::Parallel { policy, children, .. } => {
                let mut successes = 0;
                let mut failures = 0;
                let mut running = 0;
                for child in children {
                    match self.tick_node(child, bb) {
                        Status::Success => successes += 1,
                        Status::Failure => failures += 1,
                        Status::Running => running += 1,
                    }
                }
                match policy {
                    ParallelPolicy::RequireAll => {
                        if failures > 0 { Status::Failure }
                        else if running > 0 { Status::Running }
                        else { Status::Success }
                    }
                    ParallelPolicy::RequireOne => {
                        if successes > 0 { Status::Success }
                        else if running > 0 { Status::Running }
                        else { Status::Failure }
                    }
                }
            }
            Node::Decorator { kind, repeat_count, child, .. } => {
                match kind {
                    DecoratorKind::Inverter => match self.tick_node(child, bb) {
                        Status::Success => Status::Failure,
                        Status::Failure => Status::Success,
                        Status::Running => Status::Running,
                    },
                    DecoratorKind::AlwaysSucceed => {
                        let s = self.tick_node(child, bb);
                        if s == Status::Running { Status::Running } else { Status::Success }
                    }
                    DecoratorKind::AlwaysFail => {
                        let s = self.tick_node(child, bb);
                        if s == Status::Running { Status::Running } else { Status::Failure }
                    }
                    DecoratorKind::Repeat => {
                        for _ in 0..*repeat_count {
                            match self.tick_node(child, bb) {
                                Status::Running => return Status::Running,
                                Status::Failure => return Status::Failure,
                                Status::Success => {}
                            }
                        }
                        Status::Success
                    }
                    DecoratorKind::RetryOnFailure => {
                        for _ in 0..*repeat_count {
                            match self.tick_node(child, bb) {
                                Status::Success => return Status::Success,
                                Status::Running => return Status::Running,
                                Status::Failure => {}
                            }
                        }
                        Status::Failure
                    }
                }
            }
            Node::Action { name } => {
                if let Some(handler) = self.actions.get(name) {
                    handler(bb)
                } else {
                    Status::Failure
                }
            }
            Node::Condition { key, expected, .. } => {
                match bb.get_bool(key) {
                    Some(val) if val == *expected => Status::Success,
                    _ => Status::Failure,
                }
            }
        }
    }
}

// ── Priority selector ───────────────────────────────────────────

/// A priority-based selector: children are sorted by priority (highest first).
pub fn priority_selector(name: &str, mut children: Vec<(i32, Node)>) -> Node {
    children.sort_by(|a, b| b.0.cmp(&a.0));
    Node::selector(name, children.into_iter().map(|(_, n)| n).collect())
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_all_succeed() {
        let mut bb = Blackboard::new();
        let tree = BehaviorTree::new(Node::sequence("root", vec![
            Node::action("a"),
            Node::action("b"),
        ]));
        // No handlers registered → failure on first action.
        assert_eq!(tree.tick(&mut bb), Status::Failure);
    }

    #[test]
    fn sequence_succeeds_with_handlers() {
        let mut bb = Blackboard::new();
        let mut tree = BehaviorTree::new(Node::sequence("root", vec![
            Node::action("a"),
            Node::action("b"),
        ]));
        tree.register_action("a", |_| Status::Success);
        tree.register_action("b", |_| Status::Success);
        assert_eq!(tree.tick(&mut bb), Status::Success);
    }

    #[test]
    fn selector_first_success() {
        let mut bb = Blackboard::new();
        let mut tree = BehaviorTree::new(Node::selector("root", vec![
            Node::action("fail"),
            Node::action("ok"),
        ]));
        tree.register_action("fail", |_| Status::Failure);
        tree.register_action("ok", |_| Status::Success);
        assert_eq!(tree.tick(&mut bb), Status::Success);
    }

    #[test]
    fn condition_checks_blackboard() {
        let mut bb = Blackboard::new();
        bb.set_bool("enemy_visible", true);
        let tree = BehaviorTree::new(Node::condition("see_enemy", "enemy_visible", true));
        assert_eq!(tree.tick(&mut bb), Status::Success);
        bb.set_bool("enemy_visible", false);
        assert_eq!(tree.tick(&mut bb), Status::Failure);
    }

    #[test]
    fn inverter_decorator() {
        let mut bb = Blackboard::new();
        let mut tree = BehaviorTree::new(
            Node::decorator("inv", DecoratorKind::Inverter, Node::action("ok")),
        );
        tree.register_action("ok", |_| Status::Success);
        assert_eq!(tree.tick(&mut bb), Status::Failure);
    }

    #[test]
    fn always_succeed_decorator() {
        let mut bb = Blackboard::new();
        let tree = BehaviorTree::new(
            Node::decorator("always", DecoratorKind::AlwaysSucceed, Node::action("missing")),
        );
        assert_eq!(tree.tick(&mut bb), Status::Success);
    }

    #[test]
    fn repeat_decorator() {
        let mut bb = Blackboard::new();
        bb.set_int("counter", 0);
        let mut tree = BehaviorTree::new(Node::repeat("rep", 3, Node::action("inc")));
        tree.register_action("inc", |bb| {
            let c = bb.get_int("counter").unwrap_or(0);
            bb.set_int("counter", c + 1);
            Status::Success
        });
        assert_eq!(tree.tick(&mut bb), Status::Success);
        assert_eq!(bb.get_int("counter"), Some(3));
    }

    #[test]
    fn parallel_require_all() {
        let mut bb = Blackboard::new();
        let mut tree = BehaviorTree::new(Node::parallel(
            "par",
            ParallelPolicy::RequireAll,
            vec![Node::action("a"), Node::action("b")],
        ));
        tree.register_action("a", |_| Status::Success);
        tree.register_action("b", |_| Status::Success);
        assert_eq!(tree.tick(&mut bb), Status::Success);
    }

    #[test]
    fn parallel_require_one() {
        let mut bb = Blackboard::new();
        let mut tree = BehaviorTree::new(Node::parallel(
            "par",
            ParallelPolicy::RequireOne,
            vec![Node::action("a"), Node::action("b")],
        ));
        tree.register_action("a", |_| Status::Failure);
        tree.register_action("b", |_| Status::Success);
        assert_eq!(tree.tick(&mut bb), Status::Success);
    }

    #[test]
    fn priority_selector_ordering() {
        let mut bb = Blackboard::new();
        bb.set_int("called", 0);
        let node = priority_selector("prio", vec![
            (1, Node::action("low")),
            (10, Node::action("high")),
            (5, Node::action("mid")),
        ]);
        let mut tree = BehaviorTree::new(node);
        // high runs first — it succeeds, so low/mid never run.
        tree.register_action("high", |bb| { bb.set_int("called", 10); Status::Success });
        tree.register_action("mid", |bb| { bb.set_int("called", 5); Status::Success });
        tree.register_action("low", |bb| { bb.set_int("called", 1); Status::Success });
        assert_eq!(tree.tick(&mut bb), Status::Success);
        assert_eq!(bb.get_int("called"), Some(10));
    }

    #[test]
    fn blackboard_types() {
        let mut bb = Blackboard::new();
        bb.set_int("hp", 100);
        bb.set_float("speed", 3.14);
        bb.set_string("name", "bot");
        bb.set_bool("alive", true);
        assert_eq!(bb.get_int("hp"), Some(100));
        assert!((bb.get_float("speed").unwrap() - 3.14).abs() < 1e-9);
        assert_eq!(bb.get_string("name"), Some("bot"));
        assert_eq!(bb.get_bool("alive"), Some(true));
        bb.clear();
        assert_eq!(bb.get_int("hp"), None);
    }

    #[test]
    fn retry_decorator() {
        let mut bb = Blackboard::new();
        bb.set_int("attempts", 0);
        let mut tree = BehaviorTree::new(Node::retry("retry", 3, Node::action("flaky")));
        tree.register_action("flaky", |bb| {
            let a = bb.get_int("attempts").unwrap_or(0);
            bb.set_int("attempts", a + 1);
            if a < 2 { Status::Failure } else { Status::Success }
        });
        assert_eq!(tree.tick(&mut bb), Status::Success);
        assert_eq!(bb.get_int("attempts"), Some(3));
    }

    #[test]
    fn subtree_reuse() {
        let patrol = Node::sequence("patrol", vec![
            Node::action("move_to_waypoint"),
            Node::action("wait"),
        ]);
        // Reuse the same subtree pattern via clone.
        let root = Node::selector("root", vec![
            Node::sequence("combat", vec![
                Node::condition("check_enemy", "enemy_near", true),
                Node::action("attack"),
            ]),
            patrol.clone(),
        ]);
        let mut bb = Blackboard::new();
        let mut tree = BehaviorTree::new(root);
        tree.register_action("move_to_waypoint", |_| Status::Success);
        tree.register_action("wait", |_| Status::Success);
        // No enemy → combat fails → patrol runs.
        assert_eq!(tree.tick(&mut bb), Status::Success);
    }
}
