//! Command bus pattern — single-handler dispatch, middleware chain, validation,
//! retry, command queuing, and typed results.
//!
//! Replaces JS command bus libraries (simple-command-bus, NestJS CQRS) with a
//! pure-Rust command bus supporting middleware pipelines (logging, validation,
//! retry), handler registration, and command result tracking.

use std::collections::{HashMap, VecDeque};

// ── Errors ─────────────────────────────────────────────────────

/// Command bus domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandBusError {
    /// No handler registered for this command type.
    NoHandler(String),
    /// Multiple handlers registered (only one allowed per type).
    DuplicateHandler(String),
    /// Command validation failed.
    ValidationFailed { command_type: String, reason: String },
    /// Handler returned an error.
    HandlerFailed { command_type: String, reason: String },
    /// Retry exhausted.
    RetryExhausted { command_type: String, attempts: u32 },
    /// Command not found.
    CommandNotFound(String),
    /// Middleware rejected the command.
    MiddlewareRejected { middleware: String, reason: String },
    /// Bus is stopped.
    BusStopped,
}

impl std::fmt::Display for CommandBusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoHandler(t) => write!(f, "no handler for command type: {t}"),
            Self::DuplicateHandler(t) => write!(f, "duplicate handler for: {t}"),
            Self::ValidationFailed { command_type, reason } => {
                write!(f, "validation failed for {command_type}: {reason}")
            }
            Self::HandlerFailed { command_type, reason } => {
                write!(f, "handler failed for {command_type}: {reason}")
            }
            Self::RetryExhausted { command_type, attempts } => {
                write!(f, "retry exhausted for {command_type} after {attempts} attempts")
            }
            Self::CommandNotFound(id) => write!(f, "command not found: {id}"),
            Self::MiddlewareRejected { middleware, reason } => {
                write!(f, "middleware {middleware} rejected: {reason}")
            }
            Self::BusStopped => write!(f, "command bus is stopped"),
        }
    }
}

impl std::error::Error for CommandBusError {}

// ── Command ───────────────────────────────────────────────────

/// A command to be dispatched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub id: String,
    pub command_type: String,
    pub payload: String,
    pub metadata: HashMap<String, String>,
    pub timestamp_ms: u64,
}

impl Command {
    pub fn new(
        id: impl Into<String>,
        command_type: impl Into<String>,
        payload: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            command_type: command_type.into(),
            payload: payload.into(),
            metadata: HashMap::new(),
            timestamp_ms: 0,
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

// ── Command Status ────────────────────────────────────────────

/// Lifecycle status of a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandStatus {
    Queued,
    Validating,
    Executing,
    Completed,
    Failed,
    Rejected,
}

// ── Command Result ────────────────────────────────────────────

/// The result of executing a command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub command_id: String,
    pub command_type: String,
    pub status: CommandStatus,
    pub output: Option<String>,
    pub error: Option<String>,
    pub attempts: u32,
    pub middleware_log: Vec<String>,
}

// ── Validation Rule ───────────────────────────────────────────

/// A validation rule for commands.
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub name: String,
    pub command_type: String,
    /// If true, the payload must not be empty.
    pub require_non_empty_payload: bool,
    /// Required metadata keys.
    pub required_metadata: Vec<String>,
    /// Maximum payload length (0 = no limit).
    pub max_payload_len: usize,
}

impl ValidationRule {
    pub fn new(name: impl Into<String>, command_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command_type: command_type.into(),
            require_non_empty_payload: false,
            required_metadata: Vec::new(),
            max_payload_len: 0,
        }
    }

    pub fn require_payload(mut self) -> Self {
        self.require_non_empty_payload = true;
        self
    }

    pub fn require_meta(mut self, key: impl Into<String>) -> Self {
        self.required_metadata.push(key.into());
        self
    }

    pub fn max_payload(mut self, len: usize) -> Self {
        self.max_payload_len = len;
        self
    }

    /// Validate a command against this rule.
    pub fn validate(&self, cmd: &Command) -> Result<(), String> {
        if self.require_non_empty_payload && cmd.payload.is_empty() {
            return Err("payload must not be empty".to_string());
        }
        if self.max_payload_len > 0 && cmd.payload.len() > self.max_payload_len {
            return Err(format!(
                "payload too long: {} > {}",
                cmd.payload.len(),
                self.max_payload_len
            ));
        }
        for key in &self.required_metadata {
            if !cmd.metadata.contains_key(key) {
                return Err(format!("missing required metadata key: {key}"));
            }
        }
        Ok(())
    }
}

// ── Handler ───────────────────────────────────────────────────

/// A command handler (simulated).
#[derive(Debug, Clone)]
struct HandlerEntry {
    command_type: String,
    /// If Some, the handler will fail with this error.
    will_fail: Option<String>,
    /// Output to return on success.
    output: String,
    /// Invocation count.
    invocations: u64,
}

// ── Middleware ─────────────────────────────────────────────────

/// Middleware kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MiddlewareKind {
    /// Logs command details.
    Logging,
    /// Validates command before execution.
    Validation,
    /// Retries on failure.
    Retry { max_attempts: u32 },
    /// Custom rejection (for testing).
    Reject { reason: String },
}

/// A middleware in the pipeline.
#[derive(Debug, Clone)]
pub struct Middleware {
    pub name: String,
    pub kind: MiddlewareKind,
    /// Number of times this middleware has been invoked.
    pub invocations: u64,
}

impl Middleware {
    pub fn logging(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: MiddlewareKind::Logging,
            invocations: 0,
        }
    }

    pub fn validation(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: MiddlewareKind::Validation,
            invocations: 0,
        }
    }

    pub fn retry(name: impl Into<String>, max_attempts: u32) -> Self {
        Self {
            name: name.into(),
            kind: MiddlewareKind::Retry { max_attempts },
            invocations: 0,
        }
    }

    pub fn reject(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: MiddlewareKind::Reject {
                reason: reason.into(),
            },
            invocations: 0,
        }
    }
}

// ── Command Bus ───────────────────────────────────────────────

/// The command bus.
#[derive(Debug)]
pub struct CommandBus {
    handlers: HashMap<String, HandlerEntry>,
    middlewares: Vec<Middleware>,
    validation_rules: Vec<ValidationRule>,
    queue: VecDeque<Command>,
    results: HashMap<String, CommandResult>,
    clock_ms: u64,
    stopped: bool,
    /// Middleware execution log for the latest dispatch.
    last_middleware_log: Vec<String>,
}

impl Default for CommandBus {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandBus {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            middlewares: Vec::new(),
            validation_rules: Vec::new(),
            queue: VecDeque::new(),
            results: HashMap::new(),
            clock_ms: 0,
            stopped: false,
            last_middleware_log: Vec::new(),
        }
    }

    pub fn advance_time(&mut self, ms: u64) {
        self.clock_ms += ms;
    }

    pub fn stop(&mut self) {
        self.stopped = true;
    }

    pub fn start(&mut self) {
        self.stopped = false;
    }

    // ── Handler Registration ─────────────────────────────────

    /// Register a handler for a command type.
    pub fn register_handler(
        &mut self,
        command_type: impl Into<String>,
        output: impl Into<String>,
    ) -> Result<(), CommandBusError> {
        let ct = command_type.into();
        if self.handlers.contains_key(&ct) {
            return Err(CommandBusError::DuplicateHandler(ct));
        }
        self.handlers.insert(
            ct.clone(),
            HandlerEntry {
                command_type: ct,
                will_fail: None,
                output: output.into(),
                invocations: 0,
            },
        );
        Ok(())
    }

    /// Register a handler that will simulate failure.
    pub fn register_failing_handler(
        &mut self,
        command_type: impl Into<String>,
        error: impl Into<String>,
    ) -> Result<(), CommandBusError> {
        let ct = command_type.into();
        if self.handlers.contains_key(&ct) {
            return Err(CommandBusError::DuplicateHandler(ct));
        }
        self.handlers.insert(
            ct.clone(),
            HandlerEntry {
                command_type: ct,
                will_fail: Some(error.into()),
                output: String::new(),
                invocations: 0,
            },
        );
        Ok(())
    }

    /// Unregister a handler.
    pub fn unregister_handler(&mut self, command_type: &str) -> bool {
        self.handlers.remove(command_type).is_some()
    }

    // ── Middleware ────────────────────────────────────────────

    /// Add a middleware to the pipeline.
    pub fn add_middleware(&mut self, mw: Middleware) {
        self.middlewares.push(mw);
    }

    /// Add a validation rule.
    pub fn add_validation_rule(&mut self, rule: ValidationRule) {
        self.validation_rules.push(rule);
    }

    // ── Queue ────────────────────────────────────────────────

    /// Enqueue a command for later processing.
    pub fn enqueue(&mut self, mut cmd: Command) {
        cmd.timestamp_ms = self.clock_ms;
        self.queue.push_back(cmd);
    }

    /// Process all queued commands. Returns results.
    pub fn process_queue(&mut self) -> Vec<Result<CommandResult, CommandBusError>> {
        let commands: Vec<Command> = self.queue.drain(..).collect();
        let mut results = Vec::new();
        for cmd in commands {
            results.push(self.dispatch(cmd));
        }
        results
    }

    /// Queue length.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    // ── Dispatch ─────────────────────────────────────────────

    /// Dispatch a command immediately.
    pub fn dispatch(&mut self, mut cmd: Command) -> Result<CommandResult, CommandBusError> {
        if self.stopped {
            return Err(CommandBusError::BusStopped);
        }

        cmd.timestamp_ms = self.clock_ms;
        let cmd_id = cmd.id.clone();
        let cmd_type = cmd.command_type.clone();

        // Check handler exists.
        if !self.handlers.contains_key(&cmd_type) {
            return Err(CommandBusError::NoHandler(cmd_type));
        }

        let mut middleware_log = Vec::new();
        let mut max_retries = 1u32;

        // Run middlewares.
        for mw in &mut self.middlewares {
            mw.invocations += 1;
            match &mw.kind {
                MiddlewareKind::Logging => {
                    middleware_log.push(format!("[{}] command={}", mw.name, cmd_type));
                }
                MiddlewareKind::Validation => {
                    // Run validation rules.
                    let rules: Vec<_> = self
                        .validation_rules
                        .iter()
                        .filter(|r| r.command_type == cmd_type)
                        .cloned()
                        .collect();
                    for rule in &rules {
                        if let Err(reason) = rule.validate(&cmd) {
                            let result = CommandResult {
                                command_id: cmd_id.clone(),
                                command_type: cmd_type.clone(),
                                status: CommandStatus::Rejected,
                                output: None,
                                error: Some(reason.clone()),
                                attempts: 0,
                                middleware_log: middleware_log.clone(),
                            };
                            self.results.insert(cmd_id.clone(), result.clone());
                            return Err(CommandBusError::ValidationFailed {
                                command_type: cmd_type,
                                reason,
                            });
                        }
                    }
                    middleware_log.push(format!("[{}] validated", mw.name));
                }
                MiddlewareKind::Retry { max_attempts } => {
                    max_retries = *max_attempts;
                    middleware_log.push(format!("[{}] retry max={}", mw.name, max_attempts));
                }
                MiddlewareKind::Reject { reason } => {
                    let result = CommandResult {
                        command_id: cmd_id.clone(),
                        command_type: cmd_type.clone(),
                        status: CommandStatus::Rejected,
                        output: None,
                        error: Some(reason.clone()),
                        attempts: 0,
                        middleware_log: middleware_log.clone(),
                    };
                    self.results.insert(cmd_id.clone(), result);
                    return Err(CommandBusError::MiddlewareRejected {
                        middleware: mw.name.clone(),
                        reason: reason.clone(),
                    });
                }
            }
        }

        self.last_middleware_log = middleware_log.clone();

        // Execute handler with retries.
        let mut attempts = 0u32;
        loop {
            attempts += 1;
            let handler = self.handlers.get_mut(&cmd_type).unwrap();
            handler.invocations += 1;

            match &handler.will_fail {
                Some(err) => {
                    if attempts >= max_retries {
                        let result = CommandResult {
                            command_id: cmd_id.clone(),
                            command_type: cmd_type.clone(),
                            status: CommandStatus::Failed,
                            output: None,
                            error: Some(err.clone()),
                            attempts,
                            middleware_log: middleware_log.clone(),
                        };
                        self.results.insert(cmd_id.clone(), result.clone());
                        return Err(CommandBusError::HandlerFailed {
                            command_type: cmd_type,
                            reason: err.clone(),
                        });
                    }
                    // Retry.
                    continue;
                }
                None => {
                    let output = handler.output.clone();
                    let result = CommandResult {
                        command_id: cmd_id.clone(),
                        command_type: cmd_type.clone(),
                        status: CommandStatus::Completed,
                        output: Some(output),
                        error: None,
                        attempts,
                        middleware_log,
                    };
                    self.results.insert(cmd_id, result.clone());
                    return Ok(result);
                }
            }
        }
    }

    // ── Queries ──────────────────────────────────────────────

    /// Get result by command ID.
    pub fn get_result(&self, command_id: &str) -> Option<&CommandResult> {
        self.results.get(command_id)
    }

    /// Get handler invocation count.
    pub fn handler_invocations(&self, command_type: &str) -> Option<u64> {
        self.handlers.get(command_type).map(|h| h.invocations)
    }

    /// Total completed commands.
    pub fn completed_count(&self) -> usize {
        self.results
            .values()
            .filter(|r| r.status == CommandStatus::Completed)
            .count()
    }

    /// Total failed commands.
    pub fn failed_count(&self) -> usize {
        self.results
            .values()
            .filter(|r| r.status == CommandStatus::Failed)
            .count()
    }

    /// Last middleware log.
    pub fn last_middleware_log(&self) -> &[String] {
        &self.last_middleware_log
    }

    /// Middleware invocation count by name.
    pub fn middleware_invocations(&self, name: &str) -> Option<u64> {
        self.middlewares
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.invocations)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_dispatch() {
        let mut bus = CommandBus::new();
        bus.register_handler("create_user", "user-123").unwrap();
        let result = bus
            .dispatch(Command::new("c1", "create_user", "alice"))
            .unwrap();
        assert_eq!(result.status, CommandStatus::Completed);
        assert_eq!(result.output, Some("user-123".to_string()));
    }

    #[test]
    fn test_no_handler() {
        let mut bus = CommandBus::new();
        assert!(matches!(
            bus.dispatch(Command::new("c1", "nope", "data")),
            Err(CommandBusError::NoHandler(_))
        ));
    }

    #[test]
    fn test_duplicate_handler() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        assert!(matches!(
            bus.register_handler("cmd", "other"),
            Err(CommandBusError::DuplicateHandler(_))
        ));
    }

    #[test]
    fn test_handler_failure() {
        let mut bus = CommandBus::new();
        bus.register_failing_handler("cmd", "boom").unwrap();
        assert!(matches!(
            bus.dispatch(Command::new("c1", "cmd", "data")),
            Err(CommandBusError::HandlerFailed { .. })
        ));
    }

    #[test]
    fn test_unregister_handler() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        assert!(bus.unregister_handler("cmd"));
        assert!(!bus.unregister_handler("cmd"));
    }

    #[test]
    fn test_validation_middleware() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.add_middleware(Middleware::validation("validator"));
        bus.add_validation_rule(
            ValidationRule::new("require_payload", "cmd").require_payload(),
        );
        // Empty payload should fail.
        assert!(matches!(
            bus.dispatch(Command::new("c1", "cmd", "")),
            Err(CommandBusError::ValidationFailed { .. })
        ));
        // Non-empty should succeed.
        let result = bus.dispatch(Command::new("c2", "cmd", "data")).unwrap();
        assert_eq!(result.status, CommandStatus::Completed);
    }

    #[test]
    fn test_max_payload_validation() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.add_middleware(Middleware::validation("v"));
        bus.add_validation_rule(ValidationRule::new("size", "cmd").max_payload(5));
        assert!(matches!(
            bus.dispatch(Command::new("c1", "cmd", "too long payload")),
            Err(CommandBusError::ValidationFailed { .. })
        ));
    }

    #[test]
    fn test_required_metadata_validation() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.add_middleware(Middleware::validation("v"));
        bus.add_validation_rule(
            ValidationRule::new("meta", "cmd").require_meta("user_id"),
        );
        // Missing metadata.
        assert!(matches!(
            bus.dispatch(Command::new("c1", "cmd", "data")),
            Err(CommandBusError::ValidationFailed { .. })
        ));
        // With metadata.
        let cmd = Command::new("c2", "cmd", "data").with_metadata("user_id", "u1");
        let result = bus.dispatch(cmd).unwrap();
        assert_eq!(result.status, CommandStatus::Completed);
    }

    #[test]
    fn test_logging_middleware() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.add_middleware(Middleware::logging("logger"));
        bus.dispatch(Command::new("c1", "cmd", "data")).unwrap();
        assert!(!bus.last_middleware_log().is_empty());
        assert!(bus.last_middleware_log()[0].contains("logger"));
    }

    #[test]
    fn test_retry_middleware() {
        let mut bus = CommandBus::new();
        bus.register_failing_handler("cmd", "transient error").unwrap();
        bus.add_middleware(Middleware::retry("retrier", 3));
        let err = bus.dispatch(Command::new("c1", "cmd", "data")).unwrap_err();
        assert!(matches!(err, CommandBusError::HandlerFailed { .. }));
        // Handler was invoked 3 times (max_retries).
        assert_eq!(bus.handler_invocations("cmd"), Some(3));
    }

    #[test]
    fn test_reject_middleware() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.add_middleware(Middleware::reject("bouncer", "not allowed"));
        assert!(matches!(
            bus.dispatch(Command::new("c1", "cmd", "data")),
            Err(CommandBusError::MiddlewareRejected { .. })
        ));
    }

    #[test]
    fn test_command_queue() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.enqueue(Command::new("c1", "cmd", "a"));
        bus.enqueue(Command::new("c2", "cmd", "b"));
        assert_eq!(bus.queue_len(), 2);
        let results = bus.process_queue();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert_eq!(bus.queue_len(), 0);
    }

    #[test]
    fn test_stopped_bus() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.stop();
        assert!(matches!(
            bus.dispatch(Command::new("c1", "cmd", "data")),
            Err(CommandBusError::BusStopped)
        ));
        bus.start();
        bus.dispatch(Command::new("c2", "cmd", "data")).unwrap();
    }

    #[test]
    fn test_get_result() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.dispatch(Command::new("c1", "cmd", "data")).unwrap();
        let result = bus.get_result("c1").unwrap();
        assert_eq!(result.status, CommandStatus::Completed);
    }

    #[test]
    fn test_completed_and_failed_counts() {
        let mut bus = CommandBus::new();
        bus.register_handler("ok_cmd", "ok").unwrap();
        bus.register_failing_handler("bad_cmd", "err").unwrap();
        bus.dispatch(Command::new("c1", "ok_cmd", "a")).unwrap();
        let _ = bus.dispatch(Command::new("c2", "bad_cmd", "b"));
        assert_eq!(bus.completed_count(), 1);
        assert_eq!(bus.failed_count(), 1);
    }

    #[test]
    fn test_middleware_invocation_count() {
        let mut bus = CommandBus::new();
        bus.register_handler("cmd", "ok").unwrap();
        bus.add_middleware(Middleware::logging("log"));
        bus.dispatch(Command::new("c1", "cmd", "a")).unwrap();
        bus.dispatch(Command::new("c2", "cmd", "b")).unwrap();
        assert_eq!(bus.middleware_invocations("log"), Some(2));
    }

    #[test]
    fn test_command_metadata() {
        let cmd = Command::new("c1", "cmd", "data")
            .with_metadata("user", "alice")
            .with_metadata("role", "admin");
        assert_eq!(cmd.metadata.get("user").unwrap(), "alice");
        assert_eq!(cmd.metadata.get("role").unwrap(), "admin");
    }
}
