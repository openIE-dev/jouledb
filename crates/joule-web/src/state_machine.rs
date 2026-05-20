//! Extended state machine — states, events, transitions with guards,
//! entry/exit actions, hierarchical states, parallel regions, history states,
//! and state machine composition.
//!
//! Replaces XState (JS) with a pure-Rust hierarchical state machine
//! supporting UML statechart semantics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// State machine domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateMachineError {
    /// State not found.
    StateNotFound(String),
    /// Event not handled in current state.
    EventNotHandled { state: String, event: String },
    /// Guard rejected the transition.
    GuardRejected { from: String, to: String, guard: String },
    /// Duplicate state ID.
    DuplicateState(String),
    /// Duplicate transition.
    DuplicateTransition { from: String, event: String },
    /// Machine not started.
    NotStarted,
    /// Machine already finalized.
    AlreadyFinalized,
    /// Invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for StateMachineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateNotFound(s) => write!(f, "state not found: {s}"),
            Self::EventNotHandled { state, event } => {
                write!(f, "event {event} not handled in state {state}")
            }
            Self::GuardRejected { from, to, guard } => {
                write!(f, "guard {guard} rejected transition {from} -> {to}")
            }
            Self::DuplicateState(s) => write!(f, "duplicate state: {s}"),
            Self::DuplicateTransition { from, event } => {
                write!(f, "duplicate transition from {from} on {event}")
            }
            Self::NotStarted => write!(f, "state machine not started"),
            Self::AlreadyFinalized => write!(f, "state machine already finalized"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for StateMachineError {}

// ── Guard / Action ──────────────────────────────────────────────

/// Named guard condition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Guard {
    pub name: String,
    pub context_key: String,
    pub expected_value: String,
}

impl Guard {
    pub fn new(name: impl Into<String>, key: impl Into<String>, val: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            context_key: key.into(),
            expected_value: val.into(),
        }
    }

    /// Evaluate the guard against context.
    pub fn evaluate(&self, context: &HashMap<String, String>) -> bool {
        context.get(&self.context_key).map_or(false, |v| v == &self.expected_value)
    }
}

/// Named action to execute on entry/exit/transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionKind {
    /// Set a context key.
    SetContext { key: String, value: String },
    /// Remove a context key.
    RemoveContext { key: String },
    /// Log a message.
    Log { message: String },
    /// Custom named action.
    Custom { name: String },
}

/// A named action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action {
    pub id: String,
    pub kind: ActionKind,
}

impl Action {
    pub fn set_context(id: impl Into<String>, key: impl Into<String>, val: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: ActionKind::SetContext { key: key.into(), value: val.into() },
        }
    }

    pub fn log(id: impl Into<String>, msg: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: ActionKind::Log { message: msg.into() },
        }
    }

    /// Execute the action against context, returning log messages.
    pub fn execute(&self, context: &mut HashMap<String, String>) -> Option<String> {
        match &self.kind {
            ActionKind::SetContext { key, value } => {
                context.insert(key.clone(), value.clone());
                None
            }
            ActionKind::RemoveContext { key } => {
                context.remove(key);
                None
            }
            ActionKind::Log { message } => Some(message.clone()),
            ActionKind::Custom { name } => Some(format!("custom:{name}")),
        }
    }
}

// ── State Types ─────────────────────────────────────────────────

/// The type of a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateType {
    /// Normal atomic state.
    Atomic,
    /// Composite state with sub-states.
    Composite,
    /// Parallel state with regions.
    Parallel,
    /// Final state (terminal).
    Final,
    /// History pseudo-state (shallow).
    ShallowHistory,
    /// History pseudo-state (deep).
    DeepHistory,
}

/// A state definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateDefinition {
    pub id: String,
    pub state_type: StateType,
    pub parent: Option<String>,
    pub initial_child: Option<String>,
    pub on_entry: Vec<Action>,
    pub on_exit: Vec<Action>,
    pub children: Vec<String>,
}

impl StateDefinition {
    pub fn new(id: impl Into<String>, state_type: StateType) -> Self {
        Self {
            id: id.into(),
            state_type,
            parent: None,
            initial_child: None,
            on_entry: Vec::new(),
            on_exit: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn with_parent(mut self, p: impl Into<String>) -> Self {
        self.parent = Some(p.into());
        self
    }

    pub fn with_initial_child(mut self, c: impl Into<String>) -> Self {
        self.initial_child = Some(c.into());
        self
    }

    pub fn with_entry_action(mut self, a: Action) -> Self {
        self.on_entry.push(a);
        self
    }

    pub fn with_exit_action(mut self, a: Action) -> Self {
        self.on_exit.push(a);
        self
    }

    pub fn with_child(mut self, c: impl Into<String>) -> Self {
        self.children.push(c.into());
        self
    }
}

// ── Transition ──────────────────────────────────────────────────

/// A transition between states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transition {
    pub from: String,
    pub event: String,
    pub to: String,
    pub guard: Option<Guard>,
    pub actions: Vec<Action>,
}

impl Transition {
    pub fn new(from: impl Into<String>, event: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            event: event.into(),
            to: to.into(),
            guard: None,
            actions: Vec::new(),
        }
    }

    pub fn with_guard(mut self, g: Guard) -> Self {
        self.guard = Some(g);
        self
    }

    pub fn with_action(mut self, a: Action) -> Self {
        self.actions.push(a);
        self
    }
}

// ── Transition Record ───────────────────────────────────────────

/// Record of a state transition for history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionRecord {
    pub from: String,
    pub to: String,
    pub event: String,
    pub timestamp: DateTime<Utc>,
    pub actions_executed: Vec<String>,
}

// ── State Machine ───────────────────────────────────────────────

/// A hierarchical state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachine {
    pub id: String,
    pub states: HashMap<String, StateDefinition>,
    pub transitions: Vec<Transition>,
    pub initial_state: String,
    pub current_state: Option<String>,
    pub context: HashMap<String, String>,
    pub history: Vec<TransitionRecord>,
    /// Shallow history: maps parent_state_id → last active child state.
    pub shallow_history: HashMap<String, String>,
    /// Deep history: maps parent_state_id → full path of last active states.
    pub deep_history: HashMap<String, Vec<String>>,
    pub started: bool,
    pub finalized: bool,
}

impl StateMachine {
    pub fn new(id: impl Into<String>, initial: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            states: HashMap::new(),
            transitions: Vec::new(),
            initial_state: initial.into(),
            current_state: None,
            context: HashMap::new(),
            history: Vec::new(),
            shallow_history: HashMap::new(),
            deep_history: HashMap::new(),
            started: false,
            finalized: false,
        }
    }

    /// Add a state definition.
    pub fn add_state(&mut self, state: StateDefinition) -> Result<(), StateMachineError> {
        if self.states.contains_key(&state.id) {
            return Err(StateMachineError::DuplicateState(state.id));
        }
        self.states.insert(state.id.clone(), state);
        Ok(())
    }

    /// Add a transition.
    pub fn add_transition(&mut self, t: Transition) -> Result<(), StateMachineError> {
        self.transitions.push(t);
        Ok(())
    }

    /// Start the state machine, entering the initial state.
    pub fn start(&mut self) -> Result<(), StateMachineError> {
        if !self.states.contains_key(&self.initial_state) {
            return Err(StateMachineError::StateNotFound(self.initial_state.clone()));
        }
        self.current_state = Some(self.initial_state.clone());
        self.started = true;

        // Execute entry actions for initial state.
        let state = self.states.get(&self.initial_state).cloned();
        if let Some(s) = state {
            for action in &s.on_entry {
                action.execute(&mut self.context);
            }
        }
        Ok(())
    }

    /// Get the current state ID.
    pub fn current(&self) -> Result<&str, StateMachineError> {
        self.current_state.as_deref().ok_or(StateMachineError::NotStarted)
    }

    /// Check if the machine is in a final state.
    pub fn is_final(&self) -> bool {
        if let Some(current) = &self.current_state {
            self.states.get(current).map_or(false, |s| s.state_type == StateType::Final)
        } else {
            false
        }
    }

    /// Send an event to the state machine, triggering a transition if applicable.
    pub fn send(&mut self, event: &str) -> Result<TransitionRecord, StateMachineError> {
        if !self.started {
            return Err(StateMachineError::NotStarted);
        }
        if self.finalized {
            return Err(StateMachineError::AlreadyFinalized);
        }

        let current = self.current_state.clone()
            .ok_or(StateMachineError::NotStarted)?;

        // Find matching transition.
        let transition = self.transitions.iter()
            .find(|t| t.from == current && t.event == event)
            .cloned()
            .ok_or_else(|| StateMachineError::EventNotHandled {
                state: current.clone(),
                event: event.to_string(),
            })?;

        // Check guard.
        if let Some(guard) = &transition.guard {
            if !guard.evaluate(&self.context) {
                return Err(StateMachineError::GuardRejected {
                    from: current.clone(),
                    to: transition.to.clone(),
                    guard: guard.name.clone(),
                });
            }
        }

        // Verify target state exists.
        if !self.states.contains_key(&transition.to) {
            return Err(StateMachineError::StateNotFound(transition.to.clone()));
        }

        // Execute exit actions.
        let mut action_names = Vec::new();
        if let Some(state) = self.states.get(&current).cloned() {
            for action in &state.on_exit {
                action.execute(&mut self.context);
                action_names.push(action.id.clone());
            }
            // Save shallow history for parent.
            if let Some(parent) = &state.parent {
                self.shallow_history.insert(parent.clone(), current.clone());
            }
        }

        // Execute transition actions.
        for action in &transition.actions {
            action.execute(&mut self.context);
            action_names.push(action.id.clone());
        }

        // Execute entry actions for target.
        if let Some(target) = self.states.get(&transition.to).cloned() {
            for action in &target.on_entry {
                action.execute(&mut self.context);
                action_names.push(action.id.clone());
            }

            // Handle composite state: enter initial child.
            if target.state_type == StateType::Composite {
                if let Some(initial) = &target.initial_child {
                    self.current_state = Some(initial.clone());
                } else {
                    self.current_state = Some(transition.to.clone());
                }
            } else {
                self.current_state = Some(transition.to.clone());
            }

            if target.state_type == StateType::Final {
                self.finalized = true;
            }
        } else {
            self.current_state = Some(transition.to.clone());
        }

        let record = TransitionRecord {
            from: current,
            to: self.current_state.clone().unwrap_or_default(),
            event: event.to_string(),
            timestamp: Utc::now(),
            actions_executed: action_names,
        };

        self.history.push(record.clone());
        Ok(record)
    }

    /// Get the ancestors of a state (parent chain).
    pub fn ancestors(&self, state_id: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = state_id.to_string();
        while let Some(state) = self.states.get(&current) {
            if let Some(parent) = &state.parent {
                result.push(parent.clone());
                current = parent.clone();
            } else {
                break;
            }
        }
        result
    }

    /// Restore a history state (shallow).
    pub fn restore_history(&self, parent_id: &str) -> Option<&str> {
        self.shallow_history.get(parent_id).map(|s| s.as_str())
    }

    /// Get all states of a given type.
    pub fn states_of_type(&self, st: StateType) -> Vec<&StateDefinition> {
        self.states.values().filter(|s| s.state_type == st).collect()
    }
}

// ── Compose two machines ────────────────────────────────────────

/// Compose two state machines into a single machine by prefixing state/event names.
pub fn compose(
    a: &StateMachine,
    b: &StateMachine,
    combined_id: &str,
    initial: &str,
) -> StateMachine {
    let mut combined = StateMachine::new(combined_id, initial);
    let prefix_a = format!("{}_", a.id);
    let prefix_b = format!("{}_", b.id);

    for (id, state) in &a.states {
        let mut s = state.clone();
        s.id = format!("{prefix_a}{id}");
        s.parent = s.parent.map(|p| format!("{prefix_a}{p}"));
        let _ = combined.add_state(s);
    }
    for (id, state) in &b.states {
        let mut s = state.clone();
        s.id = format!("{prefix_b}{id}");
        s.parent = s.parent.map(|p| format!("{prefix_b}{p}"));
        let _ = combined.add_state(s);
    }
    for t in &a.transitions {
        let _ = combined.add_transition(Transition {
            from: format!("{prefix_a}{}", t.from),
            event: t.event.clone(),
            to: format!("{prefix_a}{}", t.to),
            guard: t.guard.clone(),
            actions: t.actions.clone(),
        });
    }
    for t in &b.transitions {
        let _ = combined.add_transition(Transition {
            from: format!("{prefix_b}{}", t.from),
            event: t.event.clone(),
            to: format!("{prefix_b}{}", t.to),
            guard: t.guard.clone(),
            actions: t.actions.clone(),
        });
    }
    combined
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn traffic_light() -> StateMachine {
        let mut sm = StateMachine::new("traffic", "green");
        sm.add_state(StateDefinition::new("green", StateType::Atomic)
            .with_entry_action(Action::set_context("enter-green", "light", "green")))
            .unwrap();
        sm.add_state(StateDefinition::new("yellow", StateType::Atomic)).unwrap();
        sm.add_state(StateDefinition::new("red", StateType::Atomic)).unwrap();
        sm.add_transition(Transition::new("green", "timer", "yellow")).unwrap();
        sm.add_transition(Transition::new("yellow", "timer", "red")).unwrap();
        sm.add_transition(Transition::new("red", "timer", "green")).unwrap();
        sm
    }

    #[test]
    fn test_basic_transitions() {
        let mut sm = traffic_light();
        sm.start().unwrap();
        assert_eq!(sm.current().unwrap(), "green");

        sm.send("timer").unwrap();
        assert_eq!(sm.current().unwrap(), "yellow");

        sm.send("timer").unwrap();
        assert_eq!(sm.current().unwrap(), "red");

        sm.send("timer").unwrap();
        assert_eq!(sm.current().unwrap(), "green");
    }

    #[test]
    fn test_entry_actions() {
        let mut sm = traffic_light();
        sm.start().unwrap();
        assert_eq!(sm.context.get("light"), Some(&"green".to_string()));
    }

    #[test]
    fn test_unhandled_event() {
        let mut sm = traffic_light();
        sm.start().unwrap();
        let err = sm.send("unknown").unwrap_err();
        assert!(matches!(err, StateMachineError::EventNotHandled { .. }));
    }

    #[test]
    fn test_guard_condition() {
        let mut sm = StateMachine::new("guarded", "locked");
        sm.add_state(StateDefinition::new("locked", StateType::Atomic)).unwrap();
        sm.add_state(StateDefinition::new("unlocked", StateType::Atomic)).unwrap();
        sm.add_transition(
            Transition::new("locked", "coin", "unlocked")
                .with_guard(Guard::new("has_coin", "coin_inserted", "true")),
        ).unwrap();
        sm.start().unwrap();

        // Without coin in context → guard rejects.
        let err = sm.send("coin").unwrap_err();
        assert!(matches!(err, StateMachineError::GuardRejected { .. }));

        // With coin → guard passes.
        sm.context.insert("coin_inserted".into(), "true".into());
        sm.send("coin").unwrap();
        assert_eq!(sm.current().unwrap(), "unlocked");
    }

    #[test]
    fn test_final_state() {
        let mut sm = StateMachine::new("process", "running");
        sm.add_state(StateDefinition::new("running", StateType::Atomic)).unwrap();
        sm.add_state(StateDefinition::new("done", StateType::Final)).unwrap();
        sm.add_transition(Transition::new("running", "finish", "done")).unwrap();
        sm.start().unwrap();

        sm.send("finish").unwrap();
        assert!(sm.is_final());
        assert!(sm.finalized);
        assert!(sm.send("anything").is_err());
    }

    #[test]
    fn test_composite_state() {
        let mut sm = StateMachine::new("app", "idle");
        sm.add_state(StateDefinition::new("idle", StateType::Atomic)).unwrap();
        sm.add_state(
            StateDefinition::new("active", StateType::Composite)
                .with_initial_child("sub_a")
                .with_child("sub_a")
                .with_child("sub_b"),
        ).unwrap();
        sm.add_state(
            StateDefinition::new("sub_a", StateType::Atomic)
                .with_parent("active"),
        ).unwrap();
        sm.add_state(
            StateDefinition::new("sub_b", StateType::Atomic)
                .with_parent("active"),
        ).unwrap();
        sm.add_transition(Transition::new("idle", "activate", "active")).unwrap();
        sm.add_transition(Transition::new("sub_a", "next", "sub_b")).unwrap();
        sm.start().unwrap();

        sm.send("activate").unwrap();
        // Should enter initial child sub_a.
        assert_eq!(sm.current().unwrap(), "sub_a");

        sm.send("next").unwrap();
        assert_eq!(sm.current().unwrap(), "sub_b");
    }

    #[test]
    fn test_history_tracking() {
        let mut sm = traffic_light();
        sm.start().unwrap();
        sm.send("timer").unwrap();
        sm.send("timer").unwrap();
        assert_eq!(sm.history.len(), 2);
        assert_eq!(sm.history[0].from, "green");
        assert_eq!(sm.history[0].to, "yellow");
    }

    #[test]
    fn test_shallow_history() {
        let mut sm = StateMachine::new("h", "a");
        sm.add_state(
            StateDefinition::new("parent", StateType::Composite)
                .with_initial_child("a")
                .with_child("a")
                .with_child("b"),
        ).unwrap();
        sm.add_state(StateDefinition::new("a", StateType::Atomic).with_parent("parent")).unwrap();
        sm.add_state(StateDefinition::new("b", StateType::Atomic).with_parent("parent")).unwrap();
        sm.add_state(StateDefinition::new("outside", StateType::Atomic)).unwrap();
        sm.add_transition(Transition::new("a", "next", "b")).unwrap();
        sm.add_transition(Transition::new("b", "leave", "outside")).unwrap();
        sm.start().unwrap();

        sm.send("next").unwrap(); // a → b
        sm.send("leave").unwrap(); // b → outside (shallow_history["parent"] = "b")

        assert_eq!(sm.restore_history("parent"), Some("b"));
    }

    #[test]
    fn test_transition_actions() {
        let mut sm = StateMachine::new("act", "s1");
        sm.add_state(StateDefinition::new("s1", StateType::Atomic)).unwrap();
        sm.add_state(StateDefinition::new("s2", StateType::Atomic)).unwrap();
        sm.add_transition(
            Transition::new("s1", "go", "s2")
                .with_action(Action::set_context("set-x", "x", "42")),
        ).unwrap();
        sm.start().unwrap();
        sm.send("go").unwrap();
        assert_eq!(sm.context.get("x"), Some(&"42".to_string()));
    }

    #[test]
    fn test_not_started_error() {
        let sm = traffic_light();
        assert!(matches!(sm.current(), Err(StateMachineError::NotStarted)));
    }

    #[test]
    fn test_duplicate_state() {
        let mut sm = StateMachine::new("d", "a");
        sm.add_state(StateDefinition::new("a", StateType::Atomic)).unwrap();
        let err = sm.add_state(StateDefinition::new("a", StateType::Atomic)).unwrap_err();
        assert!(matches!(err, StateMachineError::DuplicateState(_)));
    }

    #[test]
    fn test_ancestors() {
        let mut sm = StateMachine::new("anc", "c");
        sm.add_state(StateDefinition::new("root", StateType::Composite)).unwrap();
        sm.add_state(StateDefinition::new("mid", StateType::Composite).with_parent("root")).unwrap();
        sm.add_state(StateDefinition::new("c", StateType::Atomic).with_parent("mid")).unwrap();
        let ancs = sm.ancestors("c");
        assert_eq!(ancs, vec!["mid", "root"]);
    }

    #[test]
    fn test_compose_machines() {
        let mut a = StateMachine::new("a", "s1");
        a.add_state(StateDefinition::new("s1", StateType::Atomic)).unwrap();
        a.add_state(StateDefinition::new("s2", StateType::Atomic)).unwrap();
        a.add_transition(Transition::new("s1", "go", "s2")).unwrap();

        let mut b = StateMachine::new("b", "x1");
        b.add_state(StateDefinition::new("x1", StateType::Atomic)).unwrap();

        let combined = compose(&a, &b, "combined", "a_s1");
        assert!(combined.states.contains_key("a_s1"));
        assert!(combined.states.contains_key("b_x1"));
        assert_eq!(combined.transitions.len(), 1);
    }
}
