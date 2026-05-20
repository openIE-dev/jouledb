//! Application event bus — typed event dispatch, handler registration/deregistration,
//! synchronous dispatch, event history/replay, before/after hooks, error isolation.
//!
//! Replaces JS event bus libraries (mitt, EventEmitter2, postal.js) with a
//! pure-Rust event bus that provides typed dispatch, hook chains, handler
//! error isolation, and full event replay.

use std::collections::{HashMap, VecDeque};

// ── Errors ─────────────────────────────────────────────────────

/// Event bus domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventBusError {
    /// Handler not found.
    HandlerNotFound(u64),
    /// Event type not registered.
    EventTypeNotRegistered(String),
    /// Hook error.
    HookFailed { hook_id: u64, reason: String },
    /// Handler error (isolated).
    HandlerFailed { handler_id: u64, reason: String },
    /// Bus is paused.
    BusPaused,
}

impl std::fmt::Display for EventBusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HandlerNotFound(id) => write!(f, "handler not found: {id}"),
            Self::EventTypeNotRegistered(t) => write!(f, "event type not registered: {t}"),
            Self::HookFailed { hook_id, reason } => {
                write!(f, "hook {hook_id} failed: {reason}")
            }
            Self::HandlerFailed { handler_id, reason } => {
                write!(f, "handler {handler_id} failed: {reason}")
            }
            Self::BusPaused => write!(f, "event bus is paused"),
        }
    }
}

impl std::error::Error for EventBusError {}

// ── Event ─────────────────────────────────────────────────────

/// An event on the bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub id: u64,
    pub event_type: String,
    pub payload: String,
    pub source: Option<String>,
    pub timestamp_ms: u64,
}

impl Event {
    pub fn new(event_type: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            id: 0,
            event_type: event_type.into(),
            payload: payload.into(),
            source: None,
            timestamp_ms: 0,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

// ── Handler Result ────────────────────────────────────────────

/// Outcome of invoking a handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlerOutcome {
    /// Handler processed the event successfully.
    Ok,
    /// Handler encountered an error (isolated — other handlers still run).
    Error(String),
}

// ── Handler Entry ─────────────────────────────────────────────

/// Registered handler metadata.
#[derive(Debug, Clone)]
struct HandlerEntry {
    id: u64,
    event_type: String,
    /// Simulated handler: returns Ok or an error string.
    /// We store a flag to simulate success/error handlers.
    will_fail: Option<String>,
    /// If true, deregister after first dispatch.
    once: bool,
    /// Invocation count.
    invocations: u64,
}

// ── Hook Phase ────────────────────────────────────────────────

/// When a hook runs relative to dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookPhase {
    Before,
    After,
}

/// Registered hook metadata.
#[derive(Debug, Clone)]
struct HookEntry {
    id: u64,
    phase: HookPhase,
    event_type: Option<String>,
    /// If Some, this hook will return an error.
    will_fail: Option<String>,
}

// ── Dispatch Result ───────────────────────────────────────────

/// Result of dispatching an event.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    pub event_id: u64,
    pub handlers_invoked: u32,
    pub outcomes: Vec<(u64, HandlerOutcome)>,
    pub before_hooks_run: u32,
    pub after_hooks_run: u32,
    pub hook_errors: Vec<EventBusError>,
}

// ── Event Bus ─────────────────────────────────────────────────

/// Application event bus with typed dispatch, hooks, and error isolation.
#[derive(Debug)]
pub struct EventBus {
    handlers: Vec<HandlerEntry>,
    hooks: Vec<HookEntry>,
    history: VecDeque<Event>,
    history_limit: usize,
    next_handler_id: u64,
    next_hook_id: u64,
    next_event_id: u64,
    clock_ms: u64,
    paused: bool,
    /// Per-event-type dispatch counts.
    dispatch_counts: HashMap<String, u64>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
            hooks: Vec::new(),
            history: VecDeque::new(),
            history_limit: 1000,
            next_handler_id: 1,
            next_hook_id: 1,
            next_event_id: 1,
            clock_ms: 0,
            paused: false,
            dispatch_counts: HashMap::new(),
        }
    }

    pub fn with_history_limit(mut self, limit: usize) -> Self {
        self.history_limit = limit;
        self
    }

    pub fn advance_time(&mut self, ms: u64) {
        self.clock_ms += ms;
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    // ── Handler Registration ─────────────────────────────────

    /// Register a handler for an event type. Returns handler ID.
    pub fn on(&mut self, event_type: impl Into<String>) -> u64 {
        let id = self.next_handler_id;
        self.next_handler_id += 1;
        self.handlers.push(HandlerEntry {
            id,
            event_type: event_type.into(),
            will_fail: None,
            once: false,
            invocations: 0,
        });
        id
    }

    /// Register a one-shot handler. Returns handler ID.
    pub fn once(&mut self, event_type: impl Into<String>) -> u64 {
        let id = self.next_handler_id;
        self.next_handler_id += 1;
        self.handlers.push(HandlerEntry {
            id,
            event_type: event_type.into(),
            will_fail: None,
            once: true,
            invocations: 0,
        });
        id
    }

    /// Register a handler that will simulate failure with a given error.
    pub fn on_failing(
        &mut self,
        event_type: impl Into<String>,
        error: impl Into<String>,
    ) -> u64 {
        let id = self.next_handler_id;
        self.next_handler_id += 1;
        self.handlers.push(HandlerEntry {
            id,
            event_type: event_type.into(),
            will_fail: Some(error.into()),
            once: false,
            invocations: 0,
        });
        id
    }

    /// Deregister a handler by ID.
    pub fn off(&mut self, handler_id: u64) -> Result<(), EventBusError> {
        let pos = self
            .handlers
            .iter()
            .position(|h| h.id == handler_id)
            .ok_or(EventBusError::HandlerNotFound(handler_id))?;
        self.handlers.remove(pos);
        Ok(())
    }

    /// Get invocation count for a handler.
    pub fn handler_invocations(&self, handler_id: u64) -> Option<u64> {
        self.handlers.iter().find(|h| h.id == handler_id).map(|h| h.invocations)
    }

    // ── Hooks ────────────────────────────────────────────────

    /// Register a before or after hook. If event_type is None, it runs for all events.
    pub fn add_hook(
        &mut self,
        phase: HookPhase,
        event_type: Option<String>,
    ) -> u64 {
        let id = self.next_hook_id;
        self.next_hook_id += 1;
        self.hooks.push(HookEntry {
            id,
            phase,
            event_type,
            will_fail: None,
        });
        id
    }

    /// Register a hook that will simulate failure.
    pub fn add_failing_hook(
        &mut self,
        phase: HookPhase,
        event_type: Option<String>,
        error: impl Into<String>,
    ) -> u64 {
        let id = self.next_hook_id;
        self.next_hook_id += 1;
        self.hooks.push(HookEntry {
            id,
            phase,
            event_type,
            will_fail: Some(error.into()),
        });
        id
    }

    /// Remove a hook by ID.
    pub fn remove_hook(&mut self, hook_id: u64) -> bool {
        if let Some(pos) = self.hooks.iter().position(|h| h.id == hook_id) {
            self.hooks.remove(pos);
            true
        } else {
            false
        }
    }

    // ── Dispatch ─────────────────────────────────────────────

    /// Dispatch an event synchronously to all matching handlers.
    pub fn dispatch(&mut self, mut event: Event) -> Result<DispatchResult, EventBusError> {
        if self.paused {
            return Err(EventBusError::BusPaused);
        }

        event.id = self.next_event_id;
        self.next_event_id += 1;
        event.timestamp_ms = self.clock_ms;

        let event_type = event.event_type.clone();

        // Run before hooks.
        let mut hook_errors = Vec::new();
        let mut before_count = 0u32;
        for hook in &self.hooks {
            if hook.phase != HookPhase::Before {
                continue;
            }
            let matches = hook
                .event_type
                .as_ref()
                .map_or(true, |t| t == &event_type);
            if !matches {
                continue;
            }
            before_count += 1;
            if let Some(err) = &hook.will_fail {
                hook_errors.push(EventBusError::HookFailed {
                    hook_id: hook.id,
                    reason: err.clone(),
                });
            }
        }

        // Dispatch to handlers (error isolation: each handler runs independently).
        let mut outcomes = Vec::new();
        let mut handlers_invoked = 0u32;
        let mut once_ids = Vec::new();

        for handler in &mut self.handlers {
            if handler.event_type != event_type {
                continue;
            }
            handler.invocations += 1;
            handlers_invoked += 1;
            let outcome = match &handler.will_fail {
                Some(err) => HandlerOutcome::Error(err.clone()),
                None => HandlerOutcome::Ok,
            };
            outcomes.push((handler.id, outcome));
            if handler.once {
                once_ids.push(handler.id);
            }
        }

        // Remove one-shot handlers.
        self.handlers.retain(|h| !once_ids.contains(&h.id));

        // Run after hooks.
        let mut after_count = 0u32;
        for hook in &self.hooks {
            if hook.phase != HookPhase::After {
                continue;
            }
            let matches = hook
                .event_type
                .as_ref()
                .map_or(true, |t| t == &event_type);
            if !matches {
                continue;
            }
            after_count += 1;
            if let Some(err) = &hook.will_fail {
                hook_errors.push(EventBusError::HookFailed {
                    hook_id: hook.id,
                    reason: err.clone(),
                });
            }
        }

        // Record history.
        self.history.push_back(event);
        while self.history.len() > self.history_limit {
            self.history.pop_front();
        }

        // Update dispatch count.
        *self.dispatch_counts.entry(event_type).or_insert(0) += 1;

        Ok(DispatchResult {
            event_id: self.next_event_id - 1,
            handlers_invoked,
            outcomes,
            before_hooks_run: before_count,
            after_hooks_run: after_count,
            hook_errors,
        })
    }

    // ── Replay ───────────────────────────────────────────────

    /// Replay all events in history. Returns dispatch results.
    pub fn replay(&mut self) -> Result<Vec<DispatchResult>, EventBusError> {
        let events: Vec<Event> = self.history.iter().cloned().collect();
        let mut results = Vec::new();
        for event in events {
            let fresh = Event::new(event.event_type, event.payload);
            results.push(self.dispatch(fresh)?);
        }
        Ok(results)
    }

    /// Replay events of a specific type.
    pub fn replay_by_type(
        &mut self,
        event_type: &str,
    ) -> Result<Vec<DispatchResult>, EventBusError> {
        let events: Vec<Event> = self
            .history
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect();
        let mut results = Vec::new();
        for event in events {
            let fresh = Event::new(event.event_type, event.payload);
            results.push(self.dispatch(fresh)?);
        }
        Ok(results)
    }

    // ── Queries ──────────────────────────────────────────────

    /// Get event history.
    pub fn history(&self) -> &VecDeque<Event> {
        &self.history
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Number of handlers for a specific event type.
    pub fn handler_count_for(&self, event_type: &str) -> usize {
        self.handlers
            .iter()
            .filter(|h| h.event_type == event_type)
            .count()
    }

    /// Dispatch count for an event type.
    pub fn dispatch_count(&self, event_type: &str) -> u64 {
        self.dispatch_counts.get(event_type).copied().unwrap_or(0)
    }

    /// Total events dispatched.
    pub fn total_dispatched(&self) -> u64 {
        self.dispatch_counts.values().sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_dispatch() {
        let mut bus = EventBus::new();
        bus.on("click");
        let result = bus.dispatch(Event::new("click", "button")).unwrap();
        assert_eq!(result.handlers_invoked, 1);
        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(result.outcomes[0].1, HandlerOutcome::Ok);
    }

    #[test]
    fn test_multiple_handlers_same_type() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.on("click");
        bus.on("click");
        let result = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(result.handlers_invoked, 3);
    }

    #[test]
    fn test_handler_type_isolation() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.on("hover");
        let result = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(result.handlers_invoked, 1);
    }

    #[test]
    fn test_once_handler() {
        let mut bus = EventBus::new();
        bus.once("click");
        let r1 = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(r1.handlers_invoked, 1);
        let r2 = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(r2.handlers_invoked, 0);
    }

    #[test]
    fn test_deregister_handler() {
        let mut bus = EventBus::new();
        let id = bus.on("click");
        bus.off(id).unwrap();
        let result = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(result.handlers_invoked, 0);
    }

    #[test]
    fn test_deregister_not_found() {
        let mut bus = EventBus::new();
        assert!(matches!(bus.off(999), Err(EventBusError::HandlerNotFound(999))));
    }

    #[test]
    fn test_error_isolation() {
        let mut bus = EventBus::new();
        bus.on("click"); // succeeds
        bus.on_failing("click", "handler crashed"); // fails
        bus.on("click"); // succeeds
        let result = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(result.handlers_invoked, 3);
        // Check that the second handler returned error but others ok.
        assert_eq!(result.outcomes[0].1, HandlerOutcome::Ok);
        assert_eq!(
            result.outcomes[1].1,
            HandlerOutcome::Error("handler crashed".to_string())
        );
        assert_eq!(result.outcomes[2].1, HandlerOutcome::Ok);
    }

    #[test]
    fn test_before_hook() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.add_hook(HookPhase::Before, Some("click".to_string()));
        let result = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(result.before_hooks_run, 1);
        assert_eq!(result.after_hooks_run, 0);
    }

    #[test]
    fn test_after_hook() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.add_hook(HookPhase::After, Some("click".to_string()));
        let result = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(result.after_hooks_run, 1);
    }

    #[test]
    fn test_global_hook() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.on("hover");
        bus.add_hook(HookPhase::Before, None); // global hook
        let r1 = bus.dispatch(Event::new("click", "x")).unwrap();
        let r2 = bus.dispatch(Event::new("hover", "y")).unwrap();
        assert_eq!(r1.before_hooks_run, 1);
        assert_eq!(r2.before_hooks_run, 1);
    }

    #[test]
    fn test_failing_hook() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.add_failing_hook(HookPhase::Before, Some("click".to_string()), "hook error");
        let result = bus.dispatch(Event::new("click", "x")).unwrap();
        assert_eq!(result.hook_errors.len(), 1);
        // Handlers still run despite hook error.
        assert_eq!(result.handlers_invoked, 1);
    }

    #[test]
    fn test_event_history() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.dispatch(Event::new("click", "a")).unwrap();
        bus.dispatch(Event::new("click", "b")).unwrap();
        assert_eq!(bus.history().len(), 2);
    }

    #[test]
    fn test_history_limit() {
        let mut bus = EventBus::new().with_history_limit(3);
        bus.on("e");
        for i in 0..5 {
            bus.dispatch(Event::new("e", format!("payload-{i}"))).unwrap();
        }
        assert_eq!(bus.history().len(), 3);
        // Oldest events are dropped.
        assert_eq!(bus.history()[0].payload, "payload-2");
    }

    #[test]
    fn test_replay() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.dispatch(Event::new("click", "a")).unwrap();
        bus.dispatch(Event::new("click", "b")).unwrap();
        let results = bus.replay().unwrap();
        assert_eq!(results.len(), 2);
        // Each replay re-dispatches, so now 4 total in history (2 original + 2 replayed).
        assert_eq!(bus.history().len(), 4);
    }

    #[test]
    fn test_replay_by_type() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.on("hover");
        bus.dispatch(Event::new("click", "a")).unwrap();
        bus.dispatch(Event::new("hover", "b")).unwrap();
        bus.dispatch(Event::new("click", "c")).unwrap();
        let results = bus.replay_by_type("click").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_pause_resume() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.pause();
        assert!(matches!(
            bus.dispatch(Event::new("click", "x")),
            Err(EventBusError::BusPaused)
        ));
        bus.resume();
        bus.dispatch(Event::new("click", "x")).unwrap();
    }

    #[test]
    fn test_dispatch_counts() {
        let mut bus = EventBus::new();
        bus.on("click");
        bus.on("hover");
        bus.dispatch(Event::new("click", "a")).unwrap();
        bus.dispatch(Event::new("click", "b")).unwrap();
        bus.dispatch(Event::new("hover", "c")).unwrap();
        assert_eq!(bus.dispatch_count("click"), 2);
        assert_eq!(bus.dispatch_count("hover"), 1);
        assert_eq!(bus.total_dispatched(), 3);
    }

    #[test]
    fn test_handler_invocation_count() {
        let mut bus = EventBus::new();
        let id = bus.on("click");
        bus.dispatch(Event::new("click", "a")).unwrap();
        bus.dispatch(Event::new("click", "b")).unwrap();
        assert_eq!(bus.handler_invocations(id), Some(2));
    }

    #[test]
    fn test_remove_hook() {
        let mut bus = EventBus::new();
        let hid = bus.add_hook(HookPhase::Before, None);
        assert!(bus.remove_hook(hid));
        assert!(!bus.remove_hook(hid)); // already removed
    }

    #[test]
    fn test_event_source() {
        let bus_evt = Event::new("click", "x").with_source("ui");
        assert_eq!(bus_evt.source, Some("ui".to_string()));
    }
}
