//! Debugger model with breakpoints, call stacks, stepping, and variable watches.
//!
//! Provides a pure-Rust debugging abstraction: manage breakpoints (conditional
//! or unconditional), step through execution, inspect call stacks, and watch
//! variables — all without requiring an actual debug adapter protocol connection.

use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// A single breakpoint in a source file.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub id: String,
    pub file: String,
    pub line: u32,
    pub condition: Option<String>,
    pub enabled: bool,
    pub hit_count: u64,
}

impl Breakpoint {
    /// Create a new enabled breakpoint with zero hits.
    pub fn new(file: &str, line: u32) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            file: file.to_string(),
            line,
            condition: None,
            enabled: true,
            hit_count: 0,
        }
    }

    /// Create a conditional breakpoint.
    pub fn with_condition(file: &str, line: u32, condition: &str) -> Self {
        let mut bp = Self::new(file, line);
        bp.condition = Some(condition.to_string());
        bp
    }
}

/// State of a debug session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugState {
    Running,
    Paused,
    Stopped,
}

/// A single frame on the call stack.
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub function_name: String,
    pub file: String,
    pub line: u32,
    pub scope_variables: HashMap<String, String>,
}

impl CallFrame {
    pub fn new(function_name: &str, file: &str, line: u32) -> Self {
        Self {
            function_name: function_name.to_string(),
            file: file.to_string(),
            line,
            scope_variables: HashMap::new(),
        }
    }

    /// Add a variable to this frame's scope.
    pub fn set_variable(&mut self, name: &str, value: &str) {
        self.scope_variables
            .insert(name.to_string(), value.to_string());
    }
}

/// Step command issued to the debugger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepCommand {
    StepIn,
    StepOver,
    StepOut,
    Continue,
}

/// A watched variable expression.
#[derive(Debug, Clone)]
pub struct WatchEntry {
    pub expression: String,
    pub current_value: Option<String>,
}

// ── BreakpointManager ──

/// Manages a collection of breakpoints.
#[derive(Debug, Default)]
pub struct BreakpointManager {
    breakpoints: Vec<Breakpoint>,
}

impl BreakpointManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a breakpoint and return its id.
    pub fn add(&mut self, bp: Breakpoint) -> String {
        let id = bp.id.clone();
        self.breakpoints.push(bp);
        id
    }

    /// Remove a breakpoint by id. Returns true if found.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.breakpoints.len();
        self.breakpoints.retain(|bp| bp.id != id);
        self.breakpoints.len() < before
    }

    /// Toggle enabled state of a breakpoint. Returns new state or None.
    pub fn toggle(&mut self, id: &str) -> Option<bool> {
        for bp in &mut self.breakpoints {
            if bp.id == id {
                bp.enabled = !bp.enabled;
                return Some(bp.enabled);
            }
        }
        None
    }

    /// List all breakpoints.
    pub fn list(&self) -> &[Breakpoint] {
        &self.breakpoints
    }

    /// Find breakpoints at a given file and line.
    pub fn at_location(&self, file: &str, line: u32) -> Vec<&Breakpoint> {
        self.breakpoints
            .iter()
            .filter(|bp| bp.file == file && bp.line == line && bp.enabled)
            .collect()
    }

    /// Record a hit on a breakpoint.
    pub fn record_hit(&mut self, id: &str) {
        if let Some(bp) = self.breakpoints.iter_mut().find(|bp| bp.id == id) {
            bp.hit_count += 1;
        }
    }
}

// ── DebugSession ──

/// A debug session with state, call stack, breakpoints, and watches.
pub struct DebugSession {
    pub state: DebugState,
    pub call_stack: Vec<CallFrame>,
    pub breakpoints: BreakpointManager,
    pub watches: Vec<WatchEntry>,
    pub step_history: Vec<StepCommand>,
    current_file: String,
    current_line: u32,
}

impl DebugSession {
    pub fn new() -> Self {
        Self {
            state: DebugState::Stopped,
            call_stack: Vec::new(),
            breakpoints: BreakpointManager::new(),
            watches: Vec::new(),
            step_history: Vec::new(),
            current_file: String::new(),
            current_line: 0,
        }
    }

    /// Start the debug session.
    pub fn start(&mut self) {
        self.state = DebugState::Running;
        self.step_history.clear();
    }

    /// Pause execution.
    pub fn pause(&mut self) {
        if self.state == DebugState::Running {
            self.state = DebugState::Paused;
        }
    }

    /// Stop the session.
    pub fn stop(&mut self) {
        self.state = DebugState::Stopped;
        self.call_stack.clear();
    }

    /// Push a call frame onto the stack.
    pub fn push_frame(&mut self, frame: CallFrame) {
        self.current_file = frame.file.clone();
        self.current_line = frame.line;
        self.call_stack.push(frame);
    }

    /// Pop the top call frame.
    pub fn pop_frame(&mut self) -> Option<CallFrame> {
        let frame = self.call_stack.pop();
        if let Some(top) = self.call_stack.last() {
            self.current_file = top.file.clone();
            self.current_line = top.line;
        }
        frame
    }

    /// Execute a step command.
    pub fn step(&mut self, cmd: StepCommand) {
        self.step_history.push(cmd);
        match cmd {
            StepCommand::Continue => {
                self.state = DebugState::Running;
            }
            StepCommand::StepIn | StepCommand::StepOver => {
                self.state = DebugState::Paused;
                self.current_line += 1;
            }
            StepCommand::StepOut => {
                self.state = DebugState::Paused;
                if self.call_stack.len() > 1 {
                    self.call_stack.pop();
                    if let Some(top) = self.call_stack.last() {
                        self.current_file = top.file.clone();
                        self.current_line = top.line;
                    }
                }
            }
        }
    }

    /// Add a watch expression.
    pub fn add_watch(&mut self, expression: &str) {
        self.watches.push(WatchEntry {
            expression: expression.to_string(),
            current_value: None,
        });
    }

    /// Remove a watch expression.
    pub fn remove_watch(&mut self, expression: &str) -> bool {
        let before = self.watches.len();
        self.watches.retain(|w| w.expression != expression);
        self.watches.len() < before
    }

    /// Update a watch value.
    pub fn update_watch(&mut self, expression: &str, value: &str) {
        if let Some(w) = self.watches.iter_mut().find(|w| w.expression == expression) {
            w.current_value = Some(value.to_string());
        }
    }

    /// Evaluate whether a conditional breakpoint should fire.
    /// Uses a simple expression evaluator: "hit_count > N" or "true"/"false".
    pub fn evaluate_condition(bp: &Breakpoint) -> bool {
        match &bp.condition {
            None => true,
            Some(cond) => {
                let trimmed = cond.trim();
                if trimmed == "true" {
                    return true;
                }
                if trimmed == "false" {
                    return false;
                }
                // Parse "hit_count > N" or "hit_count >= N"
                if let Some(rest) = trimmed.strip_prefix("hit_count") {
                    let rest = rest.trim();
                    if let Some(n_str) = rest.strip_prefix(">=") {
                        if let Ok(n) = n_str.trim().parse::<u64>() {
                            return bp.hit_count >= n;
                        }
                    }
                    if let Some(n_str) = rest.strip_prefix('>') {
                        if let Ok(n) = n_str.trim().parse::<u64>() {
                            return bp.hit_count > n;
                        }
                    }
                    if let Some(n_str) = rest.strip_prefix("==") {
                        if let Ok(n) = n_str.trim().parse::<u64>() {
                            return bp.hit_count == n;
                        }
                    }
                }
                // Default: treat as truthy
                true
            }
        }
    }

    /// Simulate hitting a location: check breakpoints and pause if needed.
    /// Returns list of breakpoint ids that fired.
    pub fn hit_location(&mut self, file: &str, line: u32) -> Vec<String> {
        self.current_file = file.to_string();
        self.current_line = line;

        let matching: Vec<String> = self
            .breakpoints
            .at_location(file, line)
            .into_iter()
            .map(|bp| bp.id.clone())
            .collect();

        let mut fired = Vec::new();
        for id in &matching {
            self.breakpoints.record_hit(id);
            // Re-borrow to evaluate condition
            if let Some(bp) = self
                .breakpoints
                .list()
                .iter()
                .find(|bp| bp.id == *id)
            {
                if Self::evaluate_condition(bp) {
                    fired.push(id.clone());
                }
            }
        }

        if !fired.is_empty() {
            self.state = DebugState::Paused;
        }

        fired
    }

    /// Current position.
    pub fn current_position(&self) -> (&str, u32) {
        (&self.current_file, self.current_line)
    }
}

impl Default for DebugSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breakpoint_creation() {
        let bp = Breakpoint::new("main.rs", 42);
        assert_eq!(bp.file, "main.rs");
        assert_eq!(bp.line, 42);
        assert!(bp.enabled);
        assert_eq!(bp.hit_count, 0);
        assert!(bp.condition.is_none());
    }

    #[test]
    fn test_conditional_breakpoint() {
        let bp = Breakpoint::with_condition("lib.rs", 10, "hit_count > 3");
        assert_eq!(bp.condition.as_deref(), Some("hit_count > 3"));
    }

    #[test]
    fn test_breakpoint_manager_add_remove() {
        let mut mgr = BreakpointManager::new();
        let id = mgr.add(Breakpoint::new("a.rs", 1));
        assert_eq!(mgr.list().len(), 1);
        assert!(mgr.remove(&id));
        assert_eq!(mgr.list().len(), 0);
        assert!(!mgr.remove("nonexistent"));
    }

    #[test]
    fn test_breakpoint_toggle() {
        let mut mgr = BreakpointManager::new();
        let id = mgr.add(Breakpoint::new("a.rs", 5));
        assert_eq!(mgr.toggle(&id), Some(false));
        assert_eq!(mgr.toggle(&id), Some(true));
        assert_eq!(mgr.toggle("nope"), None);
    }

    #[test]
    fn test_breakpoint_at_location() {
        let mut mgr = BreakpointManager::new();
        mgr.add(Breakpoint::new("foo.rs", 10));
        mgr.add(Breakpoint::new("foo.rs", 20));
        mgr.add(Breakpoint::new("bar.rs", 10));
        assert_eq!(mgr.at_location("foo.rs", 10).len(), 1);
        assert_eq!(mgr.at_location("bar.rs", 10).len(), 1);
        assert_eq!(mgr.at_location("foo.rs", 99).len(), 0);
    }

    #[test]
    fn test_session_lifecycle() {
        let mut sess = DebugSession::new();
        assert_eq!(sess.state, DebugState::Stopped);
        sess.start();
        assert_eq!(sess.state, DebugState::Running);
        sess.pause();
        assert_eq!(sess.state, DebugState::Paused);
        sess.stop();
        assert_eq!(sess.state, DebugState::Stopped);
    }

    #[test]
    fn test_call_stack() {
        let mut sess = DebugSession::new();
        sess.push_frame(CallFrame::new("main", "main.rs", 1));
        sess.push_frame(CallFrame::new("foo", "lib.rs", 10));
        assert_eq!(sess.call_stack.len(), 2);
        let popped = sess.pop_frame().unwrap();
        assert_eq!(popped.function_name, "foo");
        assert_eq!(sess.current_position(), ("main.rs", 1));
    }

    #[test]
    fn test_step_commands() {
        let mut sess = DebugSession::new();
        sess.start();
        sess.push_frame(CallFrame::new("main", "main.rs", 5));
        sess.step(StepCommand::StepOver);
        assert_eq!(sess.state, DebugState::Paused);
        assert_eq!(sess.current_position().1, 6);
        sess.step(StepCommand::Continue);
        assert_eq!(sess.state, DebugState::Running);
        assert_eq!(sess.step_history.len(), 2);
    }

    #[test]
    fn test_watches() {
        let mut sess = DebugSession::new();
        sess.add_watch("x + 1");
        sess.add_watch("arr.len()");
        assert_eq!(sess.watches.len(), 2);
        sess.update_watch("x + 1", "42");
        assert_eq!(sess.watches[0].current_value.as_deref(), Some("42"));
        assert!(sess.remove_watch("arr.len()"));
        assert_eq!(sess.watches.len(), 1);
    }

    #[test]
    fn test_conditional_evaluation() {
        let mut bp = Breakpoint::with_condition("a.rs", 1, "hit_count > 3");
        bp.hit_count = 2;
        assert!(!DebugSession::evaluate_condition(&bp));
        bp.hit_count = 4;
        assert!(DebugSession::evaluate_condition(&bp));

        let bp2 = Breakpoint::with_condition("a.rs", 1, "false");
        assert!(!DebugSession::evaluate_condition(&bp2));

        let bp3 = Breakpoint::new("a.rs", 1);
        assert!(DebugSession::evaluate_condition(&bp3));
    }

    #[test]
    fn test_hit_location_fires_breakpoint() {
        let mut sess = DebugSession::new();
        sess.start();
        let id = sess.breakpoints.add(Breakpoint::new("main.rs", 10));
        let fired = sess.hit_location("main.rs", 10);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0], id);
        assert_eq!(sess.state, DebugState::Paused);
    }

    #[test]
    fn test_hit_location_no_match() {
        let mut sess = DebugSession::new();
        sess.start();
        sess.breakpoints.add(Breakpoint::new("main.rs", 10));
        let fired = sess.hit_location("main.rs", 99);
        assert!(fired.is_empty());
        assert_eq!(sess.state, DebugState::Running);
    }

    #[test]
    fn test_scope_variables() {
        let mut frame = CallFrame::new("compute", "math.rs", 5);
        frame.set_variable("x", "10");
        frame.set_variable("y", "20");
        assert_eq!(frame.scope_variables.get("x").unwrap(), "10");
        assert_eq!(frame.scope_variables.len(), 2);
    }

    #[test]
    fn test_step_out_pops_frame() {
        let mut sess = DebugSession::new();
        sess.start();
        sess.push_frame(CallFrame::new("main", "main.rs", 1));
        sess.push_frame(CallFrame::new("inner", "lib.rs", 50));
        sess.step(StepCommand::StepOut);
        assert_eq!(sess.call_stack.len(), 1);
        assert_eq!(sess.current_position(), ("main.rs", 1));
    }
}
