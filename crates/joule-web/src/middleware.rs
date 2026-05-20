//! Middleware framework for action dispatch pipelines.
//!
//! Provides a `Middleware` trait, composable `MiddlewareChain`, and built-in
//! middlewares: Logger, Thunk, Debounce, and Batch.

use std::collections::VecDeque;

// ── Action (re-used from store, but self-contained here) ──

/// An action flowing through the middleware pipeline.
#[derive(Debug, Clone)]
pub struct Action {
    pub action_type: String,
    pub payload: Option<String>,
}

impl Action {
    pub fn new(action_type: impl Into<String>) -> Self {
        Self {
            action_type: action_type.into(),
            payload: None,
        }
    }

    pub fn with_payload(action_type: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            action_type: action_type.into(),
            payload: Some(payload.into()),
        }
    }
}

// ── Middleware Trait ──

/// Result of middleware processing.
#[derive(Debug, Clone)]
pub enum MiddlewareResult {
    /// Continue with this action (possibly transformed).
    Continue(Action),
    /// Continue with multiple actions (expansion).
    Multiple(Vec<Action>),
    /// Drop the action (do not continue).
    Drop,
}

/// Trait for middleware that processes actions before/after dispatch.
pub trait Middleware {
    /// Called before the action reaches the reducer. Can transform, drop,
    /// or expand the action.
    fn before_dispatch(&mut self, action: &Action, state_json: &str) -> MiddlewareResult;

    /// Called after the reducer has produced a new state.
    fn after_dispatch(&mut self, action: &Action, state_json: &str);
}

// ── MiddlewareChain ──

/// Composes N middlewares into a pipeline.
pub struct MiddlewareChain {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a middleware to the end of the chain.
    pub fn add(&mut self, mw: impl Middleware + 'static) {
        self.middlewares.push(Box::new(mw));
    }

    /// Number of middlewares in the chain.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Run the before_dispatch phase for all middlewares. Returns the final
    /// set of actions to dispatch (may be empty if all are dropped).
    pub fn before_dispatch(&mut self, action: &Action, state_json: &str) -> Vec<Action> {
        let mut actions = vec![action.clone()];

        for mw in &mut self.middlewares {
            let mut next_actions = Vec::new();
            for act in &actions {
                match mw.before_dispatch(act, state_json) {
                    MiddlewareResult::Continue(a) => next_actions.push(a),
                    MiddlewareResult::Multiple(many) => next_actions.extend(many),
                    MiddlewareResult::Drop => {} // action dropped
                }
            }
            actions = next_actions;
        }

        actions
    }

    /// Run the after_dispatch phase for all middlewares.
    pub fn after_dispatch(&mut self, action: &Action, state_json: &str) {
        for mw in &mut self.middlewares {
            mw.after_dispatch(action, state_json);
        }
    }
}

// ── LoggerMiddleware ──

/// Logs actions and state to an internal buffer.
pub struct LoggerMiddleware {
    entries: Vec<LogEntry>,
    max_entries: usize,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub action_type: String,
    pub payload: Option<String>,
    pub state_json: String,
    pub phase: LogPhase,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogPhase {
    Before,
    After,
}

impl LoggerMiddleware {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_entries: 10000,
        }
    }

    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn log(&mut self, action: &Action, state_json: &str, phase: LogPhase) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(LogEntry {
            action_type: action.action_type.clone(),
            payload: action.payload.clone(),
            state_json: state_json.to_string(),
            phase,
        });
    }
}

impl Default for LoggerMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for LoggerMiddleware {
    fn before_dispatch(&mut self, action: &Action, state_json: &str) -> MiddlewareResult {
        self.log(action, state_json, LogPhase::Before);
        MiddlewareResult::Continue(action.clone())
    }

    fn after_dispatch(&mut self, action: &Action, state_json: &str) {
        self.log(action, state_json, LogPhase::After);
    }
}

// ── ThunkMiddleware ──

/// Allows "thunk" actions: actions whose payload contains a list of
/// sub-actions to dispatch. The thunk action type is configurable.
pub struct ThunkMiddleware {
    thunk_type: String,
}

impl ThunkMiddleware {
    pub fn new() -> Self {
        Self {
            thunk_type: "THUNK".to_string(),
        }
    }

    pub fn with_type(mut self, action_type: impl Into<String>) -> Self {
        self.thunk_type = action_type.into();
        self
    }
}

impl Default for ThunkMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for ThunkMiddleware {
    fn before_dispatch(&mut self, action: &Action, _state_json: &str) -> MiddlewareResult {
        if action.action_type == self.thunk_type {
            // Parse payload as JSON array of action objects
            if let Some(payload) = &action.payload {
                if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(payload) {
                    let actions: Vec<Action> = arr
                        .into_iter()
                        .filter_map(|v| {
                            let obj = v.as_object()?;
                            let action_type = obj.get("type")?.as_str()?.to_string();
                            let payload = obj
                                .get("payload")
                                .map(|p| p.to_string());
                            Some(Action { action_type, payload })
                        })
                        .collect();
                    if !actions.is_empty() {
                        return MiddlewareResult::Multiple(actions);
                    }
                }
            }
            MiddlewareResult::Drop
        } else {
            MiddlewareResult::Continue(action.clone())
        }
    }

    fn after_dispatch(&mut self, _action: &Action, _state_json: &str) {}
}

// ── DebounceMiddleware ──

/// Coalesces rapid dispatches of the same action type, keeping only the last.
pub struct DebounceMiddleware {
    /// Pending actions keyed by action_type.
    pending: std::collections::HashMap<String, Action>,
    /// Action types to debounce. If empty, debounces all.
    watched_types: std::collections::HashSet<String>,
}

impl DebounceMiddleware {
    pub fn new() -> Self {
        Self {
            pending: std::collections::HashMap::new(),
            watched_types: std::collections::HashSet::new(),
        }
    }

    /// Only debounce specific action types.
    pub fn watch(mut self, action_type: impl Into<String>) -> Self {
        self.watched_types.insert(action_type.into());
        self
    }

    /// Flush all pending actions, returning them.
    pub fn flush(&mut self) -> Vec<Action> {
        self.pending.drain().map(|(_, a)| a).collect()
    }

    /// Number of pending debounced actions.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn should_debounce(&self, action_type: &str) -> bool {
        self.watched_types.is_empty() || self.watched_types.contains(action_type)
    }
}

impl Default for DebounceMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for DebounceMiddleware {
    fn before_dispatch(&mut self, action: &Action, _state_json: &str) -> MiddlewareResult {
        if self.should_debounce(&action.action_type) {
            self.pending
                .insert(action.action_type.clone(), action.clone());
            MiddlewareResult::Drop
        } else {
            MiddlewareResult::Continue(action.clone())
        }
    }

    fn after_dispatch(&mut self, _action: &Action, _state_json: &str) {}
}

// ── BatchMiddleware ──

/// Groups multiple actions into a batch, dispatching them together.
pub struct BatchMiddleware {
    buffer: VecDeque<Action>,
    batch_size: usize,
}

impl BatchMiddleware {
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            batch_size,
        }
    }

    /// Flush all buffered actions.
    pub fn flush(&mut self) -> Vec<Action> {
        self.buffer.drain(..).collect()
    }

    /// Number of buffered actions.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }
}

impl Middleware for BatchMiddleware {
    fn before_dispatch(&mut self, action: &Action, _state_json: &str) -> MiddlewareResult {
        self.buffer.push_back(action.clone());

        if self.buffer.len() >= self.batch_size {
            let actions: Vec<Action> = self.buffer.drain(..).collect();
            MiddlewareResult::Multiple(actions)
        } else {
            MiddlewareResult::Drop
        }
    }

    fn after_dispatch(&mut self, _action: &Action, _state_json: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logger_records_before_and_after() {
        let mut logger = LoggerMiddleware::new();
        let action = Action::new("INC");

        let result = logger.before_dispatch(&action, r#"{"count":0}"#);
        assert!(matches!(result, MiddlewareResult::Continue(_)));

        logger.after_dispatch(&action, r#"{"count":1}"#);

        let entries = logger.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].phase, LogPhase::Before);
        assert_eq!(entries[0].action_type, "INC");
        assert_eq!(entries[1].phase, LogPhase::After);
    }

    #[test]
    fn logger_max_entries() {
        let mut logger = LoggerMiddleware::new().with_max_entries(2);
        let a = Action::new("A");
        logger.before_dispatch(&a, "{}");
        logger.before_dispatch(&a, "{}");
        logger.before_dispatch(&a, "{}");
        assert_eq!(logger.entries().len(), 2);
    }

    #[test]
    fn thunk_expands_actions() {
        let mut thunk = ThunkMiddleware::new();
        let action = Action::with_payload(
            "THUNK",
            r#"[{"type":"INC"},{"type":"INC"},{"type":"DEC"}]"#,
        );
        let result = thunk.before_dispatch(&action, "{}");
        match result {
            MiddlewareResult::Multiple(actions) => {
                assert_eq!(actions.len(), 3);
                assert_eq!(actions[0].action_type, "INC");
                assert_eq!(actions[2].action_type, "DEC");
            }
            _ => panic!("expected Multiple"),
        }
    }

    #[test]
    fn thunk_passes_non_thunk_actions() {
        let mut thunk = ThunkMiddleware::new();
        let action = Action::new("INC");
        let result = thunk.before_dispatch(&action, "{}");
        assert!(matches!(result, MiddlewareResult::Continue(_)));
    }

    #[test]
    fn thunk_drops_invalid_payload() {
        let mut thunk = ThunkMiddleware::new();
        let action = Action::new("THUNK"); // no payload
        let result = thunk.before_dispatch(&action, "{}");
        assert!(matches!(result, MiddlewareResult::Drop));
    }

    #[test]
    fn debounce_coalesces_actions() {
        let mut debounce = DebounceMiddleware::new().watch("TYPING");

        let a1 = Action::with_payload("TYPING", "a");
        let a2 = Action::with_payload("TYPING", "ab");
        let a3 = Action::with_payload("TYPING", "abc");

        assert!(matches!(debounce.before_dispatch(&a1, "{}"), MiddlewareResult::Drop));
        assert!(matches!(debounce.before_dispatch(&a2, "{}"), MiddlewareResult::Drop));
        assert!(matches!(debounce.before_dispatch(&a3, "{}"), MiddlewareResult::Drop));

        assert_eq!(debounce.pending_count(), 1);
        let flushed = debounce.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].payload.as_deref(), Some("abc"));
    }

    #[test]
    fn debounce_passes_unwatched_types() {
        let mut debounce = DebounceMiddleware::new().watch("TYPING");
        let action = Action::new("SUBMIT");
        let result = debounce.before_dispatch(&action, "{}");
        assert!(matches!(result, MiddlewareResult::Continue(_)));
    }

    #[test]
    fn batch_groups_actions() {
        let mut batch = BatchMiddleware::new(3);

        let a = Action::new("A");
        assert!(matches!(batch.before_dispatch(&a, "{}"), MiddlewareResult::Drop));
        assert!(matches!(batch.before_dispatch(&a, "{}"), MiddlewareResult::Drop));

        // Third action triggers the batch
        let result = batch.before_dispatch(&a, "{}");
        match result {
            MiddlewareResult::Multiple(actions) => {
                assert_eq!(actions.len(), 3);
            }
            _ => panic!("expected Multiple"),
        }
        assert_eq!(batch.buffered_count(), 0);
    }

    #[test]
    fn batch_flush() {
        let mut batch = BatchMiddleware::new(10);
        batch.before_dispatch(&Action::new("A"), "{}");
        batch.before_dispatch(&Action::new("B"), "{}");
        assert_eq!(batch.buffered_count(), 2);

        let flushed = batch.flush();
        assert_eq!(flushed.len(), 2);
        assert_eq!(batch.buffered_count(), 0);
    }

    #[test]
    fn middleware_chain_composition() {
        let mut chain = MiddlewareChain::new();
        chain.add(LoggerMiddleware::new());
        assert_eq!(chain.len(), 1);

        let action = Action::new("TEST");
        let result = chain.before_dispatch(&action, "{}");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].action_type, "TEST");
    }

    #[test]
    fn middleware_chain_drop_propagation() {
        let mut chain = MiddlewareChain::new();
        // Add a middleware that drops all actions
        struct DropAll;
        impl Middleware for DropAll {
            fn before_dispatch(&mut self, _: &Action, _: &str) -> MiddlewareResult {
                MiddlewareResult::Drop
            }
            fn after_dispatch(&mut self, _: &Action, _: &str) {}
        }
        chain.add(DropAll);
        chain.add(LoggerMiddleware::new());

        let result = chain.before_dispatch(&Action::new("TEST"), "{}");
        assert!(result.is_empty());
    }

    #[test]
    fn middleware_chain_after_dispatch() {
        let mut chain = MiddlewareChain::new();
        chain.add(LoggerMiddleware::new());
        // This should not panic
        chain.after_dispatch(&Action::new("TEST"), r#"{"done":true}"#);
    }

    #[test]
    fn empty_chain() {
        let mut chain = MiddlewareChain::new();
        assert!(chain.is_empty());
        let result = chain.before_dispatch(&Action::new("X"), "{}");
        assert_eq!(result.len(), 1);
    }
}
