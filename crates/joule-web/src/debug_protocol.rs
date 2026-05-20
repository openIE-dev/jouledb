//! Debug protocol concepts — breakpoint management, step/continue/pause,
//! variable inspection, call stack frames, watch expressions, conditional
//! breakpoints, debug session management, and DAP-like message format.

use std::collections::HashMap;
use uuid::Uuid;

// ── Message Types ────────────────────────────────────────────────

/// DAP-like message type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    Request,
    Response,
    Event,
}

/// A DAP-like debug protocol message.
#[derive(Debug, Clone)]
pub struct DebugMessage {
    pub seq: u64,
    pub msg_type: MessageType,
    pub command: String,
    pub body: HashMap<String, String>,
    pub success: bool,
    pub request_seq: Option<u64>,
}

impl DebugMessage {
    /// Create a new request message.
    pub fn request(seq: u64, command: &str) -> Self {
        Self {
            seq,
            msg_type: MessageType::Request,
            command: command.to_string(),
            body: HashMap::new(),
            success: true,
            request_seq: None,
        }
    }

    /// Create a response to a request.
    pub fn response(seq: u64, request_seq: u64, command: &str, success: bool) -> Self {
        Self {
            seq,
            msg_type: MessageType::Response,
            command: command.to_string(),
            body: HashMap::new(),
            success,
            request_seq: Some(request_seq),
        }
    }

    /// Create an event message.
    pub fn event(seq: u64, event_name: &str) -> Self {
        Self {
            seq,
            msg_type: MessageType::Event,
            command: event_name.to_string(),
            body: HashMap::new(),
            success: true,
            request_seq: None,
        }
    }

    /// Add a body field.
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.body.insert(key.to_string(), value.to_string());
        self
    }

    /// Encode to a simple text wire format: "type|seq|command|key=val,key=val".
    pub fn encode(&self) -> String {
        let type_str = match self.msg_type {
            MessageType::Request => "req",
            MessageType::Response => "res",
            MessageType::Event => "evt",
        };

        let mut body_parts: Vec<String> = self
            .body
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        body_parts.sort(); // deterministic output
        let body_str = body_parts.join(",");

        let req_seq_str = self
            .request_seq
            .map(|s| s.to_string())
            .unwrap_or_default();

        format!(
            "{}|{}|{}|{}|{}|{}",
            type_str, self.seq, self.command, self.success as u8, req_seq_str, body_str
        )
    }

    /// Decode from wire format.
    pub fn decode(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.splitn(6, '|').collect();
        if parts.len() < 6 {
            return None;
        }

        let msg_type = match parts[0] {
            "req" => MessageType::Request,
            "res" => MessageType::Response,
            "evt" => MessageType::Event,
            _ => return None,
        };

        let seq = parts[1].parse::<u64>().ok()?;
        let command = parts[2].to_string();
        let success = parts[3] != "0";
        let request_seq = if parts[4].is_empty() {
            None
        } else {
            parts[4].parse::<u64>().ok()
        };

        let mut body = HashMap::new();
        if !parts[5].is_empty() {
            for pair in parts[5].split(',') {
                if let Some(eq_pos) = pair.find('=') {
                    let key = pair[..eq_pos].to_string();
                    let val = pair[eq_pos + 1..].to_string();
                    body.insert(key, val);
                }
            }
        }

        Some(Self {
            seq,
            msg_type,
            command,
            body,
            success,
            request_seq,
        })
    }
}

// ── Breakpoint ───────────────────────────────────────────────────

/// A breakpoint in a source file.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub id: String,
    pub file: String,
    pub line: u32,
    pub column: Option<u32>,
    pub condition: Option<String>,
    pub hit_condition: Option<HitCondition>,
    pub enabled: bool,
    pub hit_count: u64,
    pub log_message: Option<String>,
}

impl Breakpoint {
    pub fn new(file: &str, line: u32) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            file: file.to_string(),
            line,
            column: None,
            condition: None,
            hit_condition: None,
            enabled: true,
            hit_count: 0,
            log_message: None,
        }
    }

    /// Create a conditional breakpoint.
    pub fn with_condition(mut self, condition: &str) -> Self {
        self.condition = Some(condition.to_string());
        self
    }

    /// Create a breakpoint with a hit condition (break after N hits).
    pub fn with_hit_condition(mut self, condition: HitCondition) -> Self {
        self.hit_condition = Some(condition);
        self
    }

    /// Create a logpoint (does not stop, just logs).
    pub fn with_log_message(mut self, msg: &str) -> Self {
        self.log_message = Some(msg.to_string());
        self
    }

    /// Check if this breakpoint matches a given file and line.
    pub fn matches(&self, file: &str, line: u32) -> bool {
        self.enabled && self.file == file && self.line == line
    }

    /// Record a hit and return true if the breakpoint should fire.
    pub fn hit(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        self.hit_count += 1;

        match &self.hit_condition {
            Some(HitCondition::Equal(n)) => self.hit_count == *n,
            Some(HitCondition::GreaterThan(n)) => self.hit_count > *n,
            Some(HitCondition::Multiple(n)) => *n > 0 && self.hit_count % *n == 0,
            None => true,
        }
    }

    /// Is this a logpoint?
    pub fn is_logpoint(&self) -> bool {
        self.log_message.is_some()
    }
}

// ── Hit Condition ────────────────────────────────────────────────

/// Condition for when a breakpoint should fire based on hit count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HitCondition {
    /// Break when hit count == N.
    Equal(u64),
    /// Break when hit count > N.
    GreaterThan(u64),
    /// Break every N-th hit.
    Multiple(u64),
}

// ── Step Command ─────────────────────────────────────────────────

/// Step command issued to the debugger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepCommand {
    Continue,
    StepOver,
    StepInto,
    StepOut,
    Pause,
    Restart,
    Terminate,
}

// ── Debug State ──────────────────────────────────────────────────

/// State of a debug session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugState {
    Initialized,
    Running,
    Paused,
    Stopped,
    Terminated,
}

// ── Call Stack Frame ─────────────────────────────────────────────

/// A single frame on the call stack.
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub id: u32,
    pub function_name: String,
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub scopes: Vec<Scope>,
}

impl StackFrame {
    pub fn new(id: u32, function_name: &str, file: &str, line: u32) -> Self {
        Self {
            id,
            function_name: function_name.to_string(),
            file: file.to_string(),
            line,
            column: 0,
            scopes: Vec::new(),
        }
    }

    /// Add a scope to this frame.
    pub fn add_scope(&mut self, scope: Scope) {
        self.scopes.push(scope);
    }

    /// Get all variables across all scopes.
    pub fn all_variables(&self) -> Vec<&Variable> {
        self.scopes.iter().flat_map(|s| &s.variables).collect()
    }
}

// ── Scope ────────────────────────────────────────────────────────

/// A variable scope within a stack frame.
#[derive(Debug, Clone)]
pub struct Scope {
    pub name: String,
    pub kind: ScopeKind,
    pub variables: Vec<Variable>,
}

impl Scope {
    pub fn new(name: &str, kind: ScopeKind) -> Self {
        Self {
            name: name.to_string(),
            kind,
            variables: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, variable: Variable) {
        self.variables.push(variable);
    }
}

/// Kind of variable scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Local,
    Arguments,
    Closure,
    Global,
}

// ── Variable ─────────────────────────────────────────────────────

/// A variable visible in a scope.
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub value: String,
    pub var_type: String,
    pub children: Vec<Variable>,
}

impl Variable {
    pub fn new(name: &str, value: &str, var_type: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
            var_type: var_type.to_string(),
            children: Vec::new(),
        }
    }

    pub fn with_child(mut self, child: Variable) -> Self {
        self.children.push(child);
        self
    }

    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }
}

// ── Watch Expression ─────────────────────────────────────────────

/// A watch expression and its last evaluated result.
#[derive(Debug, Clone)]
pub struct WatchExpression {
    pub id: String,
    pub expression: String,
    pub result: Option<String>,
    pub error: Option<String>,
}

impl WatchExpression {
    pub fn new(expression: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            expression: expression.to_string(),
            result: None,
            error: None,
        }
    }

    /// Update the watch result.
    pub fn evaluate(&mut self, result: Result<String, String>) {
        match result {
            Ok(val) => {
                self.result = Some(val);
                self.error = None;
            }
            Err(err) => {
                self.result = None;
                self.error = Some(err);
            }
        }
    }

    pub fn has_error(&self) -> bool {
        self.error.is_some()
    }
}

// ── Debug Session ────────────────────────────────────────────────

/// A debug session managing breakpoints, state, and watches.
pub struct DebugSession {
    pub id: String,
    state: DebugState,
    breakpoints: Vec<Breakpoint>,
    call_stack: Vec<StackFrame>,
    watches: Vec<WatchExpression>,
    message_log: Vec<DebugMessage>,
    next_seq: u64,
}

impl DebugSession {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            state: DebugState::Initialized,
            breakpoints: Vec::new(),
            call_stack: Vec::new(),
            watches: Vec::new(),
            message_log: Vec::new(),
            next_seq: 1,
        }
    }

    pub fn state(&self) -> DebugState {
        self.state
    }

    /// Start the session (transition to Running).
    pub fn start(&mut self) -> bool {
        if self.state == DebugState::Initialized || self.state == DebugState::Stopped {
            self.state = DebugState::Running;
            self.log_event("started");
            true
        } else {
            false
        }
    }

    /// Execute a step command.
    pub fn step(&mut self, command: StepCommand) -> bool {
        match command {
            StepCommand::Continue => {
                if self.state == DebugState::Paused {
                    self.state = DebugState::Running;
                    self.log_event("continued");
                    true
                } else {
                    false
                }
            }
            StepCommand::Pause => {
                if self.state == DebugState::Running {
                    self.state = DebugState::Paused;
                    self.log_event("paused");
                    true
                } else {
                    false
                }
            }
            StepCommand::Terminate => {
                self.state = DebugState::Terminated;
                self.log_event("terminated");
                true
            }
            StepCommand::Restart => {
                self.state = DebugState::Running;
                self.call_stack.clear();
                self.log_event("restarted");
                true
            }
            StepCommand::StepOver | StepCommand::StepInto | StepCommand::StepOut => {
                if self.state == DebugState::Paused {
                    self.log_event(&format!("step_{:?}", command));
                    true
                } else {
                    false
                }
            }
        }
    }

    fn log_event(&mut self, event_name: &str) {
        let msg = DebugMessage::event(self.next_seq, event_name);
        self.next_seq += 1;
        self.message_log.push(msg);
    }

    // ── Breakpoints ──

    /// Add a breakpoint and return its id.
    pub fn add_breakpoint(&mut self, bp: Breakpoint) -> String {
        let id = bp.id.clone();
        self.breakpoints.push(bp);
        id
    }

    /// Remove a breakpoint by id. Returns true if found.
    pub fn remove_breakpoint(&mut self, id: &str) -> bool {
        let len_before = self.breakpoints.len();
        self.breakpoints.retain(|bp| bp.id != id);
        self.breakpoints.len() < len_before
    }

    /// Toggle a breakpoint's enabled state.
    pub fn toggle_breakpoint(&mut self, id: &str) -> bool {
        for bp in &mut self.breakpoints {
            if bp.id == id {
                bp.enabled = !bp.enabled;
                return true;
            }
        }
        false
    }

    /// Get all breakpoints.
    pub fn breakpoints(&self) -> &[Breakpoint] {
        &self.breakpoints
    }

    /// Find breakpoints at a given file and line.
    pub fn breakpoints_at(&self, file: &str, line: u32) -> Vec<&Breakpoint> {
        self.breakpoints
            .iter()
            .filter(|bp| bp.matches(file, line))
            .collect()
    }

    /// Check if execution should stop at a given location.
    /// Returns the ids of breakpoints that fired.
    pub fn check_breakpoints(&mut self, file: &str, line: u32) -> Vec<String> {
        let mut fired = Vec::new();
        for bp in &mut self.breakpoints {
            if bp.matches(file, line) && bp.hit() {
                fired.push(bp.id.clone());
            }
        }
        if !fired.is_empty() {
            self.state = DebugState::Paused;
        }
        fired
    }

    // ── Call Stack ──

    /// Push a frame onto the call stack.
    pub fn push_frame(&mut self, frame: StackFrame) {
        self.call_stack.push(frame);
    }

    /// Pop the top frame from the call stack.
    pub fn pop_frame(&mut self) -> Option<StackFrame> {
        self.call_stack.pop()
    }

    /// Get the current call stack (top frame last).
    pub fn call_stack(&self) -> &[StackFrame] {
        &self.call_stack
    }

    /// Get the top (most recent) frame.
    pub fn top_frame(&self) -> Option<&StackFrame> {
        self.call_stack.last()
    }

    // ── Watches ──

    /// Add a watch expression.
    pub fn add_watch(&mut self, expr: &str) -> String {
        let watch = WatchExpression::new(expr);
        let id = watch.id.clone();
        self.watches.push(watch);
        id
    }

    /// Remove a watch by id.
    pub fn remove_watch(&mut self, id: &str) -> bool {
        let len_before = self.watches.len();
        self.watches.retain(|w| w.id != id);
        self.watches.len() < len_before
    }

    /// Evaluate a watch expression (simplified: sets result or error).
    pub fn evaluate_watch(&mut self, id: &str, result: Result<String, String>) -> bool {
        for watch in &mut self.watches {
            if watch.id == id {
                watch.evaluate(result);
                return true;
            }
        }
        false
    }

    /// Get all watches.
    pub fn watches(&self) -> &[WatchExpression] {
        &self.watches
    }

    /// Get the message log.
    pub fn message_log(&self) -> &[DebugMessage] {
        &self.message_log
    }

    /// Number of breakpoints.
    pub fn breakpoint_count(&self) -> usize {
        self.breakpoints.len()
    }
}

impl Default for DebugSession {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_lifecycle() {
        let mut s = DebugSession::new();
        assert_eq!(s.state(), DebugState::Initialized);
        assert!(s.start());
        assert_eq!(s.state(), DebugState::Running);
    }

    #[test]
    fn test_pause_continue() {
        let mut s = DebugSession::new();
        s.start();
        assert!(s.step(StepCommand::Pause));
        assert_eq!(s.state(), DebugState::Paused);
        assert!(s.step(StepCommand::Continue));
        assert_eq!(s.state(), DebugState::Running);
    }

    #[test]
    fn test_terminate() {
        let mut s = DebugSession::new();
        s.start();
        assert!(s.step(StepCommand::Terminate));
        assert_eq!(s.state(), DebugState::Terminated);
    }

    #[test]
    fn test_restart() {
        let mut s = DebugSession::new();
        s.start();
        s.push_frame(StackFrame::new(0, "main", "main.rs", 1));
        s.step(StepCommand::Pause);
        assert!(s.step(StepCommand::Restart));
        assert_eq!(s.state(), DebugState::Running);
        assert!(s.call_stack().is_empty());
    }

    #[test]
    fn test_step_commands_require_paused() {
        let mut s = DebugSession::new();
        s.start();
        assert!(!s.step(StepCommand::StepOver));
        s.step(StepCommand::Pause);
        assert!(s.step(StepCommand::StepOver));
    }

    #[test]
    fn test_add_remove_breakpoint() {
        let mut s = DebugSession::new();
        let bp = Breakpoint::new("main.rs", 10);
        let id = s.add_breakpoint(bp);
        assert_eq!(s.breakpoint_count(), 1);
        assert!(s.remove_breakpoint(&id));
        assert_eq!(s.breakpoint_count(), 0);
    }

    #[test]
    fn test_conditional_breakpoint() {
        let bp = Breakpoint::new("main.rs", 10).with_condition("x > 5");
        assert_eq!(bp.condition.as_deref(), Some("x > 5"));
    }

    #[test]
    fn test_hit_condition_equal() {
        let mut bp = Breakpoint::new("f.rs", 1).with_hit_condition(HitCondition::Equal(3));
        assert!(!bp.hit()); // 1st hit
        assert!(!bp.hit()); // 2nd hit
        assert!(bp.hit()); // 3rd hit - fires!
        assert!(!bp.hit()); // 4th hit
    }

    #[test]
    fn test_hit_condition_greater_than() {
        let mut bp = Breakpoint::new("f.rs", 1).with_hit_condition(HitCondition::GreaterThan(2));
        assert!(!bp.hit()); // 1
        assert!(!bp.hit()); // 2
        assert!(bp.hit()); // 3 > 2
    }

    #[test]
    fn test_hit_condition_multiple() {
        let mut bp = Breakpoint::new("f.rs", 1).with_hit_condition(HitCondition::Multiple(2));
        assert!(!bp.hit()); // 1
        assert!(bp.hit()); // 2
        assert!(!bp.hit()); // 3
        assert!(bp.hit()); // 4
    }

    #[test]
    fn test_toggle_breakpoint() {
        let mut s = DebugSession::new();
        let bp = Breakpoint::new("f.rs", 5);
        let id = s.add_breakpoint(bp);
        assert!(s.toggle_breakpoint(&id));
        assert!(!s.breakpoints()[0].enabled);
        assert!(s.toggle_breakpoint(&id));
        assert!(s.breakpoints()[0].enabled);
    }

    #[test]
    fn test_breakpoints_at() {
        let mut s = DebugSession::new();
        s.add_breakpoint(Breakpoint::new("a.rs", 10));
        s.add_breakpoint(Breakpoint::new("a.rs", 20));
        s.add_breakpoint(Breakpoint::new("b.rs", 10));
        let at = s.breakpoints_at("a.rs", 10);
        assert_eq!(at.len(), 1);
    }

    #[test]
    fn test_check_breakpoints() {
        let mut s = DebugSession::new();
        s.start();
        s.add_breakpoint(Breakpoint::new("f.rs", 5));
        let fired = s.check_breakpoints("f.rs", 5);
        assert_eq!(fired.len(), 1);
        assert_eq!(s.state(), DebugState::Paused);
    }

    #[test]
    fn test_call_stack() {
        let mut s = DebugSession::new();
        s.push_frame(StackFrame::new(0, "main", "main.rs", 1));
        s.push_frame(StackFrame::new(1, "foo", "foo.rs", 10));
        assert_eq!(s.call_stack().len(), 2);
        assert_eq!(s.top_frame().unwrap().function_name, "foo");
        let popped = s.pop_frame().unwrap();
        assert_eq!(popped.function_name, "foo");
    }

    #[test]
    fn test_stack_frame_variables() {
        let mut frame = StackFrame::new(0, "main", "main.rs", 1);
        let mut scope = Scope::new("Locals", ScopeKind::Local);
        scope.add_variable(Variable::new("x", "42", "i32"));
        scope.add_variable(Variable::new("y", "hello", "String"));
        frame.add_scope(scope);
        let vars = frame.all_variables();
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_watch_expression() {
        let mut s = DebugSession::new();
        let wid = s.add_watch("x + y");
        assert!(s.evaluate_watch(&wid, Ok("42".to_string())));
        assert_eq!(s.watches()[0].result.as_deref(), Some("42"));
        assert!(!s.watches()[0].has_error());
    }

    #[test]
    fn test_watch_expression_error() {
        let mut s = DebugSession::new();
        let wid = s.add_watch("bad_expr");
        s.evaluate_watch(&wid, Err("undefined".to_string()));
        assert!(s.watches()[0].has_error());
        assert!(s.watches()[0].result.is_none());
    }

    #[test]
    fn test_remove_watch() {
        let mut s = DebugSession::new();
        let wid = s.add_watch("x");
        assert!(s.remove_watch(&wid));
        assert!(s.watches().is_empty());
    }

    #[test]
    fn test_message_encode_decode() {
        let msg = DebugMessage::request(1, "setBreakpoints")
            .with_field("file", "main.rs")
            .with_field("line", "10");
        let encoded = msg.encode();
        let decoded = DebugMessage::decode(&encoded).unwrap();
        assert_eq!(decoded.seq, 1);
        assert_eq!(decoded.command, "setBreakpoints");
        assert_eq!(decoded.msg_type, MessageType::Request);
    }

    #[test]
    fn test_response_message() {
        let msg = DebugMessage::response(2, 1, "setBreakpoints", true);
        assert_eq!(msg.msg_type, MessageType::Response);
        assert_eq!(msg.request_seq, Some(1));
        assert!(msg.success);
    }

    #[test]
    fn test_event_message() {
        let msg = DebugMessage::event(3, "stopped");
        assert_eq!(msg.msg_type, MessageType::Event);
        let encoded = msg.encode();
        assert!(encoded.starts_with("evt|"));
    }

    #[test]
    fn test_logpoint() {
        let bp = Breakpoint::new("f.rs", 1).with_log_message("value is {x}");
        assert!(bp.is_logpoint());
        assert_eq!(bp.log_message.as_deref(), Some("value is {x}"));
    }

    #[test]
    fn test_variable_children() {
        let var = Variable::new("obj", "{...}", "Object")
            .with_child(Variable::new("a", "1", "i32"))
            .with_child(Variable::new("b", "2", "i32"));
        assert!(var.has_children());
        assert_eq!(var.children.len(), 2);
    }

    #[test]
    fn test_message_log() {
        let mut s = DebugSession::new();
        s.start();
        s.step(StepCommand::Pause);
        assert!(s.message_log().len() >= 2);
    }

    #[test]
    fn test_disabled_breakpoint_no_fire() {
        let mut bp = Breakpoint::new("f.rs", 1);
        bp.enabled = false;
        assert!(!bp.hit());
        assert!(!bp.matches("f.rs", 1));
    }
}
