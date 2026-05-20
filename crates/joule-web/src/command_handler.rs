//! Command handling pipeline — command validation, aggregate loading, command
//! execution, event generation, concurrency control (retry on conflict),
//! command result, and command audit.
//!
//! Replaces JS command-handling libraries (NestJS CQRS, MediatR) with a
//! pure-Rust command pipeline for CQRS write-side processing.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Command handler errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandHandlerError {
    /// Command validation failed.
    ValidationFailed { command_type: String, reasons: Vec<String> },
    /// Aggregate not found.
    AggregateNotFound { aggregate_id: String },
    /// Version conflict during command execution.
    VersionConflict { aggregate_id: String, expected: u64, actual: u64 },
    /// Max retries exceeded.
    MaxRetriesExceeded { aggregate_id: String, retries: u32 },
    /// Command execution error.
    ExecutionError { command_type: String, reason: String },
    /// Handler not registered.
    HandlerNotRegistered(String),
    /// Aggregate already exists.
    AggregateAlreadyExists(String),
    /// Audit log error.
    AuditError(String),
}

impl std::fmt::Display for CommandHandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValidationFailed { command_type, reasons } => {
                write!(f, "validation failed for {command_type}: {}", reasons.join(", "))
            }
            Self::AggregateNotFound { aggregate_id } => {
                write!(f, "aggregate not found: {aggregate_id}")
            }
            Self::VersionConflict { aggregate_id, expected, actual } => {
                write!(
                    f,
                    "version conflict for {aggregate_id}: expected {expected}, got {actual}"
                )
            }
            Self::MaxRetriesExceeded { aggregate_id, retries } => {
                write!(f, "max retries ({retries}) exceeded for {aggregate_id}")
            }
            Self::ExecutionError { command_type, reason } => {
                write!(f, "execution error for {command_type}: {reason}")
            }
            Self::HandlerNotRegistered(cmd) => {
                write!(f, "no handler registered for command: {cmd}")
            }
            Self::AggregateAlreadyExists(id) => {
                write!(f, "aggregate already exists: {id}")
            }
            Self::AuditError(msg) => write!(f, "audit error: {msg}"),
        }
    }
}

impl std::error::Error for CommandHandlerError {}

// ── Command ─────────────────────────────────────────────────────

/// A command to be processed through the pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Command {
    pub command_id: String,
    pub command_type: String,
    pub aggregate_id: String,
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    pub expected_version: Option<u64>,
    pub timestamp: DateTime<Utc>,
}

impl Command {
    pub fn new(
        command_type: impl Into<String>,
        aggregate_id: impl Into<String>,
        data: HashMap<String, String>,
    ) -> Self {
        let ct = command_type.into();
        let aid = aggregate_id.into();
        Self {
            command_id: format!("{}-{}-{}", ct, aid, Utc::now().timestamp_nanos_opt().unwrap_or(0)),
            command_type: ct,
            aggregate_id: aid,
            data,
            metadata: HashMap::new(),
            expected_version: None,
            timestamp: Utc::now(),
        }
    }

    pub fn with_expected_version(mut self, version: u64) -> Self {
        self.expected_version = Some(version);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ── Command Result ──────────────────────────────────────────────

/// Outcome of a command execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandResult {
    pub command_id: String,
    pub aggregate_id: String,
    pub new_version: u64,
    pub events_produced: Vec<ProducedEvent>,
    pub executed_at: DateTime<Utc>,
    pub retries: u32,
}

/// An event produced by command execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducedEvent {
    pub event_type: String,
    pub data: HashMap<String, String>,
    pub version: u64,
}

// ── Validation Rule ─────────────────────────────────────────────

/// A command validation rule.
#[derive(Clone)]
pub struct ValidationRule {
    pub name: String,
    validate_fn: fn(&Command) -> Result<(), String>,
}

impl ValidationRule {
    pub fn new(
        name: impl Into<String>,
        validate_fn: fn(&Command) -> Result<(), String>,
    ) -> Self {
        Self {
            name: name.into(),
            validate_fn,
        }
    }

    pub fn validate(&self, command: &Command) -> Result<(), String> {
        (self.validate_fn)(command)
    }
}

impl std::fmt::Debug for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationRule")
            .field("name", &self.name)
            .finish()
    }
}

// ── Aggregate State ─────────────────────────────────────────────

/// In-memory aggregate state used for command handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateState {
    pub aggregate_id: String,
    pub version: u64,
    pub state: HashMap<String, String>,
    pub exists: bool,
}

impl AggregateState {
    pub fn new(aggregate_id: impl Into<String>) -> Self {
        Self {
            aggregate_id: aggregate_id.into(),
            version: 0,
            state: HashMap::new(),
            exists: false,
        }
    }

    pub fn with_state(mut self, state: HashMap<String, String>, version: u64) -> Self {
        self.state = state;
        self.version = version;
        self.exists = true;
        self
    }
}

// ── Command Handler Definition ──────────────────────────────────

/// A handler for a specific command type.
#[derive(Clone)]
pub struct HandlerDefinition {
    pub command_type: String,
    /// Handler function: takes aggregate state + command data, returns events or error.
    handler_fn: fn(&AggregateState, &HashMap<String, String>)
        -> Result<Vec<(String, HashMap<String, String>)>, String>,
    /// Validation rules specific to this handler.
    validations: Vec<ValidationRule>,
}

impl HandlerDefinition {
    pub fn new(
        command_type: impl Into<String>,
        handler_fn: fn(
            &AggregateState,
            &HashMap<String, String>,
        ) -> Result<Vec<(String, HashMap<String, String>)>, String>,
    ) -> Self {
        Self {
            command_type: command_type.into(),
            handler_fn,
            validations: Vec::new(),
        }
    }

    pub fn add_validation(&mut self, rule: ValidationRule) {
        self.validations.push(rule);
    }

    pub fn with_validation(mut self, rule: ValidationRule) -> Self {
        self.validations.push(rule);
        self
    }
}

impl std::fmt::Debug for HandlerDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandlerDefinition")
            .field("command_type", &self.command_type)
            .field("validation_count", &self.validations.len())
            .finish()
    }
}

// ── Audit Entry ─────────────────────────────────────────────────

/// An audit log entry for a command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub command_id: String,
    pub command_type: String,
    pub aggregate_id: String,
    pub outcome: AuditOutcome,
    pub error_message: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Outcome of a command for audit purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    ValidationFailed,
    ConflictRetried,
    Failed,
}

// ── Command Pipeline ────────────────────────────────────────────

/// Command handling pipeline with validation, execution, retries, and audit.
#[derive(Debug)]
pub struct CommandPipeline {
    handlers: HashMap<String, HandlerDefinition>,
    global_validations: Vec<ValidationRule>,
    /// In-memory aggregate store for testing.
    aggregates: HashMap<String, AggregateState>,
    /// Audit log.
    audit_log: Vec<AuditEntry>,
    /// Max retries on version conflict.
    max_retries: u32,
}

impl CommandPipeline {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            global_validations: Vec::new(),
            aggregates: HashMap::new(),
            audit_log: Vec::new(),
            max_retries: 3,
        }
    }

    /// Set max retries for version conflicts.
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Register a command handler.
    pub fn register_handler(&mut self, handler: HandlerDefinition) {
        self.handlers.insert(handler.command_type.clone(), handler);
    }

    /// Add a global validation rule (applies to all commands).
    pub fn add_global_validation(&mut self, rule: ValidationRule) {
        self.global_validations.push(rule);
    }

    /// Load or create an aggregate.
    pub fn load_aggregate(&self, aggregate_id: &str) -> AggregateState {
        self.aggregates
            .get(aggregate_id)
            .cloned()
            .unwrap_or_else(|| AggregateState::new(aggregate_id))
    }

    /// Store aggregate state (for testing).
    pub fn store_aggregate(&mut self, state: AggregateState) {
        self.aggregates.insert(state.aggregate_id.clone(), state);
    }

    /// Validate a command.
    fn validate(&self, command: &Command, handler: &HandlerDefinition) -> Result<(), CommandHandlerError> {
        let mut errors = Vec::new();

        // Run global validations.
        for rule in &self.global_validations {
            if let Err(msg) = rule.validate(command) {
                errors.push(msg);
            }
        }

        // Run handler-specific validations.
        for rule in &handler.validations {
            if let Err(msg) = rule.validate(command) {
                errors.push(msg);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CommandHandlerError::ValidationFailed {
                command_type: command.command_type.clone(),
                reasons: errors,
            })
        }
    }

    /// Execute a command through the pipeline.
    pub fn execute(&mut self, command: &Command) -> Result<CommandResult, CommandHandlerError> {
        let now = Utc::now();
        let cmd_type = command.command_type.clone();
        let agg_id = command.aggregate_id.clone();
        let cmd_id = command.command_id.clone();

        // Look up handler.
        let handler = self
            .handlers
            .get(&cmd_type)
            .ok_or_else(|| CommandHandlerError::HandlerNotRegistered(cmd_type.clone()))?
            .clone();

        // Validate.
        if let Err(e) = self.validate(command, &handler) {
            self.audit_log.push(AuditEntry {
                command_id: cmd_id.clone(),
                command_type: cmd_type.clone(),
                aggregate_id: agg_id.clone(),
                outcome: AuditOutcome::ValidationFailed,
                error_message: Some(e.to_string()),
                timestamp: now,
                metadata: command.metadata.clone(),
            });
            return Err(e);
        }

        // Retry loop for version conflicts.
        let mut retries = 0u32;

        loop {
            // Load aggregate.
            let aggregate = self.load_aggregate(&agg_id);

            // Check expected version.
            if let Some(expected) = command.expected_version {
                if aggregate.version != expected {
                    if retries < self.max_retries {
                        retries += 1;
                        self.audit_log.push(AuditEntry {
                            command_id: cmd_id.clone(),
                            command_type: cmd_type.clone(),
                            aggregate_id: agg_id.clone(),
                            outcome: AuditOutcome::ConflictRetried,
                            error_message: None,
                            timestamp: Utc::now(),
                            metadata: command.metadata.clone(),
                        });
                        continue;
                    } else {
                        let err = CommandHandlerError::MaxRetriesExceeded {
                            aggregate_id: agg_id.clone(),
                            retries,
                        };
                        self.audit_log.push(AuditEntry {
                            command_id: cmd_id.clone(),
                            command_type: cmd_type.clone(),
                            aggregate_id: agg_id.clone(),
                            outcome: AuditOutcome::Failed,
                            error_message: Some(err.to_string()),
                            timestamp: Utc::now(),
                            metadata: command.metadata.clone(),
                        });
                        return Err(err);
                    }
                }
            }

            // Execute handler.
            let result = (handler.handler_fn)(&aggregate, &command.data);

            match result {
                Ok(events) => {
                    let mut version = aggregate.version;
                    let mut produced = Vec::new();
                    let mut new_state = aggregate.state.clone();

                    for (event_type, event_data) in &events {
                        version += 1;
                        produced.push(ProducedEvent {
                            event_type: event_type.clone(),
                            data: event_data.clone(),
                            version,
                        });
                        // Merge event data into state.
                        for (k, v) in event_data {
                            new_state.insert(k.clone(), v.clone());
                        }
                    }

                    // Update aggregate.
                    let updated = AggregateState {
                        aggregate_id: agg_id.clone(),
                        version,
                        state: new_state,
                        exists: true,
                    };
                    self.aggregates.insert(agg_id.clone(), updated);

                    let cmd_result = CommandResult {
                        command_id: cmd_id.clone(),
                        aggregate_id: agg_id.clone(),
                        new_version: version,
                        events_produced: produced,
                        executed_at: Utc::now(),
                        retries,
                    };

                    self.audit_log.push(AuditEntry {
                        command_id: cmd_id,
                        command_type: cmd_type,
                        aggregate_id: agg_id,
                        outcome: AuditOutcome::Success,
                        error_message: None,
                        timestamp: Utc::now(),
                        metadata: command.metadata.clone(),
                    });

                    return Ok(cmd_result);
                }
                Err(reason) => {
                    let err = CommandHandlerError::ExecutionError {
                        command_type: cmd_type.clone(),
                        reason,
                    };
                    self.audit_log.push(AuditEntry {
                        command_id: cmd_id,
                        command_type: cmd_type,
                        aggregate_id: agg_id,
                        outcome: AuditOutcome::Failed,
                        error_message: Some(err.to_string()),
                        timestamp: Utc::now(),
                        metadata: command.metadata.clone(),
                    });
                    return Err(err);
                }
            }
        }
    }

    /// Get the audit log.
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Count audit entries by outcome.
    pub fn audit_count_by_outcome(&self, outcome: AuditOutcome) -> usize {
        self.audit_log.iter().filter(|e| e.outcome == outcome).count()
    }

    /// Clear the audit log.
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }

    /// Audit log size.
    pub fn audit_log_size(&self) -> usize {
        self.audit_log.len()
    }

    /// Registered handler count.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Check if a handler exists.
    pub fn has_handler(&self, command_type: &str) -> bool {
        self.handlers.contains_key(command_type)
    }

    /// Global validation count.
    pub fn global_validation_count(&self) -> usize {
        self.global_validations.len()
    }
}

impl Default for CommandPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn create_user_handler(
        agg: &AggregateState,
        data: &HashMap<String, String>,
    ) -> Result<Vec<(String, HashMap<String, String>)>, String> {
        if agg.exists {
            return Err("user already exists".to_string());
        }
        let mut event_data = HashMap::new();
        if let Some(name) = data.get("name") {
            event_data.insert("name".to_string(), name.clone());
        }
        Ok(vec![("UserCreated".to_string(), event_data)])
    }

    fn update_user_handler(
        agg: &AggregateState,
        data: &HashMap<String, String>,
    ) -> Result<Vec<(String, HashMap<String, String>)>, String> {
        if !agg.exists {
            return Err("user does not exist".to_string());
        }
        let mut event_data = HashMap::new();
        if let Some(name) = data.get("name") {
            event_data.insert("name".to_string(), name.clone());
        }
        Ok(vec![("UserUpdated".to_string(), event_data)])
    }

    fn make_command(cmd_type: &str, agg_id: &str, kv: &[(&str, &str)]) -> Command {
        let mut data = HashMap::new();
        for (k, v) in kv {
            data.insert(k.to_string(), v.to_string());
        }
        Command::new(cmd_type, agg_id, data)
    }

    fn build_pipeline() -> CommandPipeline {
        let mut pipeline = CommandPipeline::new();
        pipeline.register_handler(HandlerDefinition::new("CreateUser", create_user_handler));
        pipeline.register_handler(HandlerDefinition::new("UpdateUser", update_user_handler));
        pipeline
    }

    #[test]
    fn test_execute_creates_aggregate() {
        let mut pipeline = build_pipeline();
        let cmd = make_command("CreateUser", "user-1", &[("name", "Alice")]);
        let result = pipeline.execute(&cmd).unwrap();

        assert_eq!(result.aggregate_id, "user-1");
        assert_eq!(result.new_version, 1);
        assert_eq!(result.events_produced.len(), 1);
        assert_eq!(result.events_produced[0].event_type, "UserCreated");
    }

    #[test]
    fn test_execute_updates_aggregate() {
        let mut pipeline = build_pipeline();
        pipeline.execute(&make_command("CreateUser", "user-1", &[("name", "Alice")])).unwrap();
        let result = pipeline
            .execute(&make_command("UpdateUser", "user-1", &[("name", "Alicia")]))
            .unwrap();

        assert_eq!(result.new_version, 2);
        assert_eq!(result.events_produced[0].event_type, "UserUpdated");
    }

    #[test]
    fn test_execution_error() {
        let mut pipeline = build_pipeline();
        // UpdateUser on nonexistent aggregate.
        let err = pipeline
            .execute(&make_command("UpdateUser", "user-1", &[("name", "Alice")]))
            .unwrap_err();
        assert!(matches!(err, CommandHandlerError::ExecutionError { .. }));
    }

    #[test]
    fn test_handler_not_registered() {
        let mut pipeline = build_pipeline();
        let err = pipeline
            .execute(&make_command("DeleteUser", "user-1", &[]))
            .unwrap_err();
        assert!(matches!(err, CommandHandlerError::HandlerNotRegistered(_)));
    }

    #[test]
    fn test_validation_failure() {
        let mut pipeline = build_pipeline();

        let rule = ValidationRule::new("name_required", |cmd: &Command| {
            if cmd.data.contains_key("name") {
                Ok(())
            } else {
                Err("name is required".to_string())
            }
        });
        pipeline.add_global_validation(rule);

        let cmd = make_command("CreateUser", "user-1", &[]);
        let err = pipeline.execute(&cmd).unwrap_err();
        assert!(matches!(err, CommandHandlerError::ValidationFailed { .. }));
    }

    #[test]
    fn test_handler_specific_validation() {
        let mut pipeline = CommandPipeline::new();
        let handler = HandlerDefinition::new("CreateUser", create_user_handler).with_validation(
            ValidationRule::new("name_length", |cmd: &Command| {
                if let Some(name) = cmd.data.get("name") {
                    if name.len() >= 2 {
                        return Ok(());
                    }
                }
                Err("name must be >= 2 chars".to_string())
            }),
        );
        pipeline.register_handler(handler);

        let cmd = make_command("CreateUser", "u1", &[("name", "A")]);
        let err = pipeline.execute(&cmd).unwrap_err();
        assert!(matches!(err, CommandHandlerError::ValidationFailed { .. }));
    }

    #[test]
    fn test_version_conflict_retries() {
        let mut pipeline = build_pipeline().with_max_retries(2);

        // Store aggregate at version 5.
        let agg = AggregateState::new("user-1")
            .with_state(HashMap::new(), 5);
        pipeline.store_aggregate(agg);

        // Command expects version 3, but actual is 5.
        let cmd = make_command("UpdateUser", "user-1", &[("name", "Alice")])
            .with_expected_version(3);
        let err = pipeline.execute(&cmd).unwrap_err();
        assert!(matches!(err, CommandHandlerError::MaxRetriesExceeded { .. }));
    }

    #[test]
    fn test_expected_version_matches() {
        let mut pipeline = build_pipeline();
        pipeline
            .execute(&make_command("CreateUser", "user-1", &[("name", "Alice")]))
            .unwrap();

        let cmd = make_command("UpdateUser", "user-1", &[("name", "Bob")])
            .with_expected_version(1);
        let result = pipeline.execute(&cmd).unwrap();
        assert_eq!(result.new_version, 2);
    }

    #[test]
    fn test_audit_log_success() {
        let mut pipeline = build_pipeline();
        pipeline
            .execute(&make_command("CreateUser", "user-1", &[("name", "Alice")]))
            .unwrap();

        assert_eq!(pipeline.audit_log_size(), 1);
        assert_eq!(pipeline.audit_log()[0].outcome, AuditOutcome::Success);
        assert_eq!(pipeline.audit_log()[0].command_type, "CreateUser");
    }

    #[test]
    fn test_audit_log_failure() {
        let mut pipeline = build_pipeline();
        let _ = pipeline.execute(&make_command("UpdateUser", "ghost", &[("name", "A")]));
        assert_eq!(pipeline.audit_count_by_outcome(AuditOutcome::Failed), 1);
    }

    #[test]
    fn test_audit_log_validation_failure() {
        let mut pipeline = build_pipeline();
        pipeline.add_global_validation(ValidationRule::new("always_fail", |_| Err("nope".to_string())));
        let _ = pipeline.execute(&make_command("CreateUser", "u1", &[("name", "A")]));
        assert_eq!(pipeline.audit_count_by_outcome(AuditOutcome::ValidationFailed), 1);
    }

    #[test]
    fn test_audit_log_clear() {
        let mut pipeline = build_pipeline();
        pipeline.execute(&make_command("CreateUser", "u1", &[("name", "Alice")])).unwrap();
        assert_eq!(pipeline.audit_log_size(), 1);
        pipeline.clear_audit_log();
        assert_eq!(pipeline.audit_log_size(), 0);
    }

    #[test]
    fn test_command_metadata() {
        let cmd = make_command("CreateUser", "u1", &[("name", "Alice")])
            .with_metadata("user_id", "admin")
            .with_metadata("ip", "127.0.0.1");
        assert_eq!(cmd.metadata.get("user_id").map(|s| s.as_str()), Some("admin"));
        assert_eq!(cmd.metadata.get("ip").map(|s| s.as_str()), Some("127.0.0.1"));
    }

    #[test]
    fn test_handler_count() {
        let pipeline = build_pipeline();
        assert_eq!(pipeline.handler_count(), 2);
        assert!(pipeline.has_handler("CreateUser"));
        assert!(!pipeline.has_handler("DeleteUser"));
    }

    #[test]
    fn test_global_validation_count() {
        let mut pipeline = build_pipeline();
        assert_eq!(pipeline.global_validation_count(), 0);
        pipeline.add_global_validation(ValidationRule::new("r1", |_| Ok(())));
        assert_eq!(pipeline.global_validation_count(), 1);
    }

    #[test]
    fn test_load_aggregate_default() {
        let pipeline = build_pipeline();
        let agg = pipeline.load_aggregate("nonexistent");
        assert_eq!(agg.aggregate_id, "nonexistent");
        assert_eq!(agg.version, 0);
        assert!(!agg.exists);
    }

    #[test]
    fn test_multiple_events_from_single_command() {
        let mut pipeline = CommandPipeline::new();
        pipeline.register_handler(HandlerDefinition::new("PlaceOrder", |_agg, data| {
            let order_id = data.get("order_id").cloned().unwrap_or_default();
            Ok(vec![
                ("OrderCreated".to_string(), {
                    let mut d = HashMap::new();
                    d.insert("order_id".to_string(), order_id.clone());
                    d
                }),
                ("InventoryReserved".to_string(), {
                    let mut d = HashMap::new();
                    d.insert("order_id".to_string(), order_id);
                    d
                }),
            ])
        }));

        let result = pipeline
            .execute(&make_command("PlaceOrder", "order-1", &[("order_id", "ORD-42")]))
            .unwrap();
        assert_eq!(result.events_produced.len(), 2);
        assert_eq!(result.new_version, 2);
        assert_eq!(result.events_produced[0].version, 1);
        assert_eq!(result.events_produced[1].version, 2);
    }

    #[test]
    fn test_command_result_retries_field() {
        let mut pipeline = build_pipeline();
        let result = pipeline
            .execute(&make_command("CreateUser", "u1", &[("name", "Alice")]))
            .unwrap();
        assert_eq!(result.retries, 0);
    }

    #[test]
    fn test_aggregate_state_persists() {
        let mut pipeline = build_pipeline();
        pipeline
            .execute(&make_command("CreateUser", "user-1", &[("name", "Alice")]))
            .unwrap();

        let agg = pipeline.load_aggregate("user-1");
        assert!(agg.exists);
        assert_eq!(agg.version, 1);
        assert_eq!(agg.state.get("name").map(|s| s.as_str()), Some("Alice"));
    }

    #[test]
    fn test_audit_retried_entries() {
        let mut pipeline = build_pipeline().with_max_retries(2);
        let agg = AggregateState::new("u1").with_state(HashMap::new(), 5);
        pipeline.store_aggregate(agg);

        let cmd = make_command("UpdateUser", "u1", &[("name", "X")])
            .with_expected_version(3);
        let _ = pipeline.execute(&cmd);

        let retry_count = pipeline.audit_count_by_outcome(AuditOutcome::ConflictRetried);
        assert_eq!(retry_count, 2);
    }
}
