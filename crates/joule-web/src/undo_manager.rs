//! Undo/redo manager with the command pattern, transaction grouping,
//! max history size, branch history (tree, not linear), and serializable history.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Command ──────────────────────────────────────────────────

/// A reversible command. The `execute` and `undo` closures must be
/// mirror images of each other.
pub struct Command {
    /// Human-readable label for this command.
    pub label: String,
    /// Forward execution closure.
    execute: Box<dyn FnMut()>,
    /// Reverse execution closure.
    undo: Box<dyn FnMut()>,
}

impl Command {
    pub fn new(
        label: impl Into<String>,
        execute: impl FnMut() + 'static,
        undo: impl FnMut() + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            execute: Box::new(execute),
            undo: Box::new(undo),
        }
    }

    /// Run the forward action.
    pub fn execute(&mut self) {
        (self.execute)();
    }

    /// Run the reverse action.
    pub fn undo(&mut self) {
        (self.undo)();
    }
}

// ── UndoManager ──────────────────────────────────────────────

/// Linear undo/redo manager with transaction grouping.
pub struct UndoManager {
    undo_stack: Vec<Vec<Command>>,
    redo_stack: Vec<Vec<Command>>,
    max_history: usize,
    /// Current transaction: commands are grouped until commit.
    transaction: Option<Vec<Command>>,
}

impl Default for UndoManager {
    fn default() -> Self {
        Self::new(100)
    }
}

impl UndoManager {
    pub fn new(max_history: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_history,
            transaction: None,
        }
    }

    /// Execute a command and push it onto the undo stack.
    pub fn execute(&mut self, mut cmd: Command) {
        cmd.execute();

        if let Some(tx) = &mut self.transaction {
            tx.push(cmd);
        } else {
            self.undo_stack.push(vec![cmd]);
            self.redo_stack.clear();
            self.enforce_max_history();
        }
    }

    /// Begin a transaction. All commands until `commit()` are grouped.
    pub fn begin_transaction(&mut self) {
        self.transaction = Some(Vec::new());
    }

    /// Commit the current transaction as a single undo group.
    pub fn commit_transaction(&mut self) {
        if let Some(commands) = self.transaction.take() {
            if !commands.is_empty() {
                self.undo_stack.push(commands);
                self.redo_stack.clear();
                self.enforce_max_history();
            }
        }
    }

    /// Abort the current transaction, undoing all commands in it.
    pub fn abort_transaction(&mut self) {
        if let Some(mut commands) = self.transaction.take() {
            // Undo in reverse order
            for cmd in commands.iter_mut().rev() {
                cmd.undo();
            }
        }
    }

    /// Whether a transaction is in progress.
    pub fn in_transaction(&self) -> bool {
        self.transaction.is_some()
    }

    /// Undo the last command group. Returns true if successful.
    pub fn undo(&mut self) -> bool {
        if let Some(mut group) = self.undo_stack.pop() {
            for cmd in group.iter_mut().rev() {
                cmd.undo();
            }
            self.redo_stack.push(group);
            true
        } else {
            false
        }
    }

    /// Redo the last undone command group. Returns true if successful.
    pub fn redo(&mut self) -> bool {
        if let Some(mut group) = self.redo_stack.pop() {
            for cmd in group.iter_mut() {
                cmd.execute();
            }
            self.undo_stack.push(group);
            true
        } else {
            false
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Number of undo steps available.
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of redo steps available.
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Get labels of undo history (most recent first).
    pub fn undo_labels(&self) -> Vec<Vec<String>> {
        self.undo_stack
            .iter()
            .rev()
            .map(|group| group.iter().map(|c| c.label.clone()).collect())
            .collect()
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    fn enforce_max_history(&mut self) {
        while self.undo_stack.len() > self.max_history {
            self.undo_stack.remove(0);
        }
    }
}

// ── Branch History (Tree) ────────────────────────────────────

/// A node in the undo history tree. Each node represents a state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryNode {
    /// Unique ID of this node.
    pub id: usize,
    /// Parent node ID (None for root).
    pub parent: Option<usize>,
    /// Label describing how we got to this state.
    pub label: String,
    /// Serialized state at this point.
    pub state: String,
    /// Child node IDs (branches).
    pub children: Vec<usize>,
    /// Timestamp (milliseconds since epoch or arbitrary counter).
    pub timestamp: u64,
}

/// Tree-based history that preserves all branches rather than
/// discarding the redo stack on new edits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchHistory {
    nodes: Vec<HistoryNode>,
    current: usize,
    next_id: usize,
    max_nodes: usize,
}

impl BranchHistory {
    /// Create a new branch history with the initial state.
    pub fn new(initial_state: String, max_nodes: usize) -> Self {
        let root = HistoryNode {
            id: 0,
            parent: None,
            label: "initial".to_string(),
            state: initial_state,
            children: Vec::new(),
            timestamp: 0,
        };
        Self {
            nodes: vec![root],
            current: 0,
            next_id: 1,
            max_nodes,
        }
    }

    /// Push a new state, creating a new branch from the current node.
    pub fn push(&mut self, label: impl Into<String>, state: String, timestamp: u64) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let node = HistoryNode {
            id,
            parent: Some(self.current),
            label: label.into(),
            state,
            children: Vec::new(),
            timestamp,
        };

        // Add as child of current
        self.nodes[self.current].children.push(id);
        self.nodes.push(node);
        self.current = id;

        // Garbage collect if needed
        self.gc();

        id
    }

    /// Navigate to a specific node by ID. Returns true if successful.
    pub fn goto(&mut self, node_id: usize) -> bool {
        if node_id < self.nodes.len() {
            self.current = node_id;
            true
        } else {
            false
        }
    }

    /// Undo: move to parent. Returns true if successful.
    pub fn undo(&mut self) -> bool {
        if let Some(parent) = self.nodes[self.current].parent {
            self.current = parent;
            true
        } else {
            false
        }
    }

    /// Redo: move to the last child (most recent branch).
    /// Returns true if successful.
    pub fn redo(&mut self) -> bool {
        let children = self.nodes[self.current].children.clone();
        if let Some(&last_child) = children.last() {
            self.current = last_child;
            true
        } else {
            false
        }
    }

    /// Redo to a specific branch (child index).
    pub fn redo_branch(&mut self, branch_index: usize) -> bool {
        let children = self.nodes[self.current].children.clone();
        if branch_index < children.len() {
            self.current = children[branch_index];
            true
        } else {
            false
        }
    }

    /// Get the current node.
    pub fn current_node(&self) -> &HistoryNode {
        &self.nodes[self.current]
    }

    /// Get the current state.
    pub fn current_state(&self) -> &str {
        &self.nodes[self.current].state
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[HistoryNode] {
        &self.nodes
    }

    /// Number of branches at the current node.
    pub fn branch_count(&self) -> usize {
        self.nodes[self.current].children.len()
    }

    /// Get the path from root to current node (list of node IDs).
    pub fn path_to_current(&self) -> Vec<usize> {
        let mut path = Vec::new();
        let mut id = self.current;
        path.push(id);
        while let Some(parent) = self.nodes[id].parent {
            path.push(parent);
            id = parent;
        }
        path.reverse();
        path
    }

    /// Total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Serialize the history to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize history from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    fn gc(&mut self) {
        // Simple GC: if we exceed max_nodes, we don't actually remove nodes
        // (that would invalidate IDs), but we could in a more sophisticated impl.
        // For now, just cap total allocations by warning.
        let _ = self.max_nodes;
    }

    /// Generate a DOT graph representation of the history tree.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph history {\n  rankdir=TB;\n");
        for node in &self.nodes {
            let shape = if node.id == self.current {
                "doublecircle"
            } else {
                "circle"
            };
            dot.push_str(&format!(
                "  n{} [label=\"{}\" shape={}];\n",
                node.id, node.label, shape
            ));
            for &child in &node.children {
                dot.push_str(&format!("  n{} -> n{};\n", node.id, child));
            }
        }
        dot.push_str("}\n");
        dot
    }
}

// ── Serializable History Entry ───────────────────────────────

/// A serializable history entry for linear history export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub label: String,
    pub timestamp: u64,
    pub state_snapshot: String,
}

/// Serializable linear history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableHistory {
    pub entries: Vec<HistoryEntry>,
    pub current_index: usize,
}

impl SerializableHistory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            current_index: 0,
        }
    }

    pub fn push(&mut self, label: impl Into<String>, state: String, timestamp: u64) {
        // Truncate any redo entries
        self.entries.truncate(self.current_index + 1);
        self.entries.push(HistoryEntry {
            label: label.into(),
            timestamp,
            state_snapshot: state,
        });
        self.current_index = self.entries.len() - 1;
    }

    pub fn undo(&mut self) -> Option<&HistoryEntry> {
        if self.current_index > 0 {
            self.current_index -= 1;
            Some(&self.entries[self.current_index])
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<&HistoryEntry> {
        if self.current_index + 1 < self.entries.len() {
            self.current_index += 1;
            Some(&self.entries[self.current_index])
        } else {
            None
        }
    }

    pub fn current(&self) -> Option<&HistoryEntry> {
        self.entries.get(self.current_index)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for SerializableHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn make_counter_command(counter: &Rc<RefCell<i32>>, delta: i32) -> Command {
        let c1 = counter.clone();
        let c2 = counter.clone();
        Command::new(
            format!("add {}", delta),
            move || *c1.borrow_mut() += delta,
            move || *c2.borrow_mut() -= delta,
        )
    }

    #[test]
    fn execute_and_undo() {
        let counter = Rc::new(RefCell::new(0));
        let mut mgr = UndoManager::new(100);
        mgr.execute(make_counter_command(&counter, 5));
        assert_eq!(*counter.borrow(), 5);
        mgr.undo();
        assert_eq!(*counter.borrow(), 0);
    }

    #[test]
    fn undo_and_redo() {
        let counter = Rc::new(RefCell::new(0));
        let mut mgr = UndoManager::new(100);
        mgr.execute(make_counter_command(&counter, 10));
        mgr.execute(make_counter_command(&counter, 20));
        assert_eq!(*counter.borrow(), 30);
        mgr.undo();
        assert_eq!(*counter.borrow(), 10);
        mgr.redo();
        assert_eq!(*counter.borrow(), 30);
    }

    #[test]
    fn redo_cleared_on_new_execute() {
        let counter = Rc::new(RefCell::new(0));
        let mut mgr = UndoManager::new(100);
        mgr.execute(make_counter_command(&counter, 10));
        mgr.execute(make_counter_command(&counter, 20));
        mgr.undo();
        assert!(mgr.can_redo());
        mgr.execute(make_counter_command(&counter, 5));
        assert!(!mgr.can_redo());
    }

    #[test]
    fn transaction_grouping() {
        let counter = Rc::new(RefCell::new(0));
        let mut mgr = UndoManager::new(100);
        mgr.begin_transaction();
        mgr.execute(make_counter_command(&counter, 1));
        mgr.execute(make_counter_command(&counter, 2));
        mgr.execute(make_counter_command(&counter, 3));
        mgr.commit_transaction();
        assert_eq!(*counter.borrow(), 6);
        assert_eq!(mgr.undo_count(), 1); // grouped as one

        mgr.undo();
        assert_eq!(*counter.borrow(), 0);
    }

    #[test]
    fn transaction_abort() {
        let counter = Rc::new(RefCell::new(0));
        let mut mgr = UndoManager::new(100);
        mgr.begin_transaction();
        mgr.execute(make_counter_command(&counter, 10));
        mgr.execute(make_counter_command(&counter, 20));
        assert_eq!(*counter.borrow(), 30);
        mgr.abort_transaction();
        assert_eq!(*counter.borrow(), 0);
        assert_eq!(mgr.undo_count(), 0);
    }

    #[test]
    fn max_history_enforced() {
        let counter = Rc::new(RefCell::new(0));
        let mut mgr = UndoManager::new(3);
        for i in 1..=5 {
            mgr.execute(make_counter_command(&counter, i));
        }
        assert_eq!(mgr.undo_count(), 3);
    }

    #[test]
    fn undo_labels() {
        let counter = Rc::new(RefCell::new(0));
        let mut mgr = UndoManager::new(100);
        mgr.execute(make_counter_command(&counter, 1));
        mgr.execute(make_counter_command(&counter, 2));
        let labels = mgr.undo_labels();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], vec!["add 2"]);
        assert_eq!(labels[1], vec!["add 1"]);
    }

    #[test]
    fn clear_history() {
        let counter = Rc::new(RefCell::new(0));
        let mut mgr = UndoManager::new(100);
        mgr.execute(make_counter_command(&counter, 1));
        mgr.clear();
        assert!(!mgr.can_undo());
        assert!(!mgr.can_redo());
    }

    #[test]
    fn branch_history_basic() {
        let mut bh = BranchHistory::new("state0".into(), 100);
        assert_eq!(bh.current_state(), "state0");

        bh.push("edit1", "state1".into(), 1);
        assert_eq!(bh.current_state(), "state1");

        bh.push("edit2", "state2".into(), 2);
        assert_eq!(bh.current_state(), "state2");

        assert!(bh.undo());
        assert_eq!(bh.current_state(), "state1");

        assert!(bh.redo());
        assert_eq!(bh.current_state(), "state2");
    }

    #[test]
    fn branch_history_branching() {
        let mut bh = BranchHistory::new("root".into(), 100);
        bh.push("a", "a".into(), 1);
        bh.push("a2", "a2".into(), 2);
        bh.undo(); // back to "a"
        bh.push("b", "b".into(), 3); // new branch
        assert_eq!(bh.current_state(), "b");

        // Node "a" should have 2 children: "a2" and "b"
        bh.undo();
        assert_eq!(bh.branch_count(), 2);
    }

    #[test]
    fn branch_history_redo_specific_branch() {
        let mut bh = BranchHistory::new("root".into(), 100);
        bh.push("a", "a".into(), 1);
        let id_a = bh.current_node().id;
        bh.undo();
        bh.push("b", "b".into(), 2);
        bh.undo();

        // Branch 0 should be "a", branch 1 should be "b"
        assert!(bh.redo_branch(0));
        assert_eq!(bh.current_state(), "a");

        bh.undo();
        assert!(bh.redo_branch(1));
        assert_eq!(bh.current_state(), "b");
    }

    #[test]
    fn branch_history_path_to_current() {
        let mut bh = BranchHistory::new("root".into(), 100);
        bh.push("a", "a".into(), 1);
        bh.push("b", "b".into(), 2);
        let path = bh.path_to_current();
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn branch_history_goto() {
        let mut bh = BranchHistory::new("root".into(), 100);
        bh.push("a", "a".into(), 1);
        bh.push("b", "b".into(), 2);
        assert!(bh.goto(0));
        assert_eq!(bh.current_state(), "root");
        assert!(bh.goto(2));
        assert_eq!(bh.current_state(), "b");
        assert!(!bh.goto(999));
    }

    #[test]
    fn branch_history_serialization() {
        let mut bh = BranchHistory::new("root".into(), 100);
        bh.push("edit", "state1".into(), 1);
        let json = bh.to_json().unwrap();
        let restored = BranchHistory::from_json(&json).unwrap();
        assert_eq!(restored.current_state(), "state1");
        assert_eq!(restored.node_count(), 2);
    }

    #[test]
    fn branch_history_dot_export() {
        let mut bh = BranchHistory::new("root".into(), 100);
        bh.push("edit", "s1".into(), 1);
        let dot = bh.to_dot();
        assert!(dot.contains("digraph history"));
        assert!(dot.contains("n0"));
        assert!(dot.contains("n1"));
        assert!(dot.contains("n0 -> n1"));
    }

    #[test]
    fn serializable_history_undo_redo() {
        let mut hist = SerializableHistory::new();
        hist.push("init", "s0".into(), 0);
        hist.push("edit1", "s1".into(), 1);
        hist.push("edit2", "s2".into(), 2);

        let entry = hist.undo().unwrap();
        assert_eq!(entry.state_snapshot, "s1");

        let entry = hist.redo().unwrap();
        assert_eq!(entry.state_snapshot, "s2");
    }

    #[test]
    fn serializable_history_json() {
        let mut hist = SerializableHistory::new();
        hist.push("init", "s0".into(), 0);
        let json = hist.to_json().unwrap();
        let restored = SerializableHistory::from_json(&json).unwrap();
        assert_eq!(restored.entries.len(), 1);
    }

    #[test]
    fn in_transaction_flag() {
        let mut mgr = UndoManager::new(100);
        assert!(!mgr.in_transaction());
        mgr.begin_transaction();
        assert!(mgr.in_transaction());
        mgr.commit_transaction();
        assert!(!mgr.in_transaction());
    }

    #[test]
    fn empty_undo_returns_false() {
        let mut mgr = UndoManager::new(100);
        assert!(!mgr.undo());
        assert!(!mgr.redo());
    }
}
