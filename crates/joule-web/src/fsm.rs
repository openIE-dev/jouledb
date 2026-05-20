//! Finite state machine with guards, actions, context, and history.
//!
//! Replaces XState with a pure-Rust flat FSM that covers the 90% case:
//! named states, event-driven transitions, guard predicates, action IDs,
//! enter/exit hooks, and arbitrary JSON context.

use serde_json::Value;
use std::collections::HashMap;

// ── State ──

/// A named state with optional enter/exit action IDs and arbitrary data.
#[derive(Debug, Clone)]
pub struct State {
    pub name: String,
    pub on_enter: Option<u64>,
    pub on_exit: Option<u64>,
    pub data: HashMap<String, Value>,
}

impl State {
    /// Create a new state with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            on_enter: None,
            on_exit: None,
            data: HashMap::new(),
        }
    }

    /// Set the on-enter action ID.
    pub fn with_on_enter(mut self, id: u64) -> Self {
        self.on_enter = Some(id);
        self
    }

    /// Set the on-exit action ID.
    pub fn with_on_exit(mut self, id: u64) -> Self {
        self.on_exit = Some(id);
        self
    }
}

// ── Transition ──

/// An event-driven transition with an optional guard and action list.
pub struct Transition {
    pub event: String,
    pub target: String,
    pub guard: Option<Box<dyn Fn(&HashMap<String, Value>) -> bool>>,
    pub actions: Vec<u64>,
}

impl Transition {
    /// Create a transition that fires on `event` and moves to `target`.
    pub fn new(event: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            event: event.into(),
            target: target.into(),
            guard: None,
            actions: Vec::new(),
        }
    }

    /// Attach a guard predicate.
    pub fn with_guard(
        mut self,
        f: impl Fn(&HashMap<String, Value>) -> bool + 'static,
    ) -> Self {
        self.guard = Some(Box::new(f));
        self
    }

    /// Attach action IDs to fire when the transition executes.
    pub fn with_actions(mut self, actions: Vec<u64>) -> Self {
        self.actions = actions;
        self
    }
}

// ── StateMachine ──

/// Flat finite state machine with guards, actions, context, and history.
pub struct StateMachine {
    states: HashMap<String, State>,
    transitions: HashMap<String, Vec<Transition>>,
    current: String,
    context: HashMap<String, Value>,
    history: Vec<String>,
}

impl StateMachine {
    /// Create a new machine starting in `initial_state`.
    pub fn new(initial_state: impl Into<String>) -> Self {
        let initial = initial_state.into();
        Self {
            states: HashMap::new(),
            transitions: HashMap::new(),
            current: initial.clone(),
            context: HashMap::new(),
            history: vec![initial],
        }
    }

    /// Register a state.
    pub fn add_state(&mut self, state: State) {
        self.states.insert(state.name.clone(), state);
    }

    /// Register a transition from a given source state.
    pub fn add_transition(&mut self, from_state: impl Into<String>, transition: Transition) {
        self.transitions
            .entry(from_state.into())
            .or_default()
            .push(transition);
    }

    /// Send an event. Returns `Some(action_ids)` on successful transition,
    /// `None` if no matching transition was found or all guards blocked.
    pub fn send(&mut self, event: &str) -> Option<Vec<u64>> {
        let transitions = self.transitions.get(&self.current)?;
        let idx = transitions.iter().position(|t| {
            t.event == event
                && t.guard
                    .as_ref()
                    .map_or(true, |g| g(&self.context))
        })?;

        let target = transitions[idx].target.clone();
        let actions = transitions[idx].actions.clone();

        // Collect on_exit / on_enter
        let mut all_actions = Vec::new();
        if let Some(st) = self.states.get(&self.current) {
            if let Some(id) = st.on_exit {
                all_actions.push(id);
            }
        }
        all_actions.extend(&actions);
        if let Some(st) = self.states.get(&target) {
            if let Some(id) = st.on_enter {
                all_actions.push(id);
            }
        }

        self.current = target.clone();
        self.history.push(target);

        Some(all_actions)
    }

    /// Current state name.
    pub fn current_state(&self) -> &str {
        &self.current
    }

    /// Check whether `event` can fire from the current state (at least one
    /// transition exists whose guard passes).
    pub fn can_send(&self, event: &str) -> bool {
        self.transitions.get(&self.current).is_some_and(|ts| {
            ts.iter().any(|t| {
                t.event == event
                    && t.guard
                        .as_ref()
                        .map_or(true, |g| g(&self.context))
            })
        })
    }

    /// List all events that can fire from the current state.
    pub fn available_events(&self) -> Vec<&str> {
        self.transitions
            .get(&self.current)
            .map(|ts| {
                ts.iter()
                    .filter(|t| {
                        t.guard
                            .as_ref()
                            .map_or(true, |g| g(&self.context))
                    })
                    .map(|t| t.event.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Set a context value.
    pub fn set_context(&mut self, key: impl Into<String>, value: Value) {
        self.context.insert(key.into(), value);
    }

    /// Read a context value.
    pub fn context_value(&self, key: &str) -> Option<&Value> {
        self.context.get(key)
    }

    /// Check if the machine is in the named state.
    pub fn matches(&self, state_name: &str) -> bool {
        self.current == state_name
    }

    /// Full transition history (including the initial state).
    pub fn history(&self) -> &[String] {
        &self.history
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn traffic_light() -> StateMachine {
        let mut m = StateMachine::new("green");
        m.add_state(State::new("green"));
        m.add_state(State::new("yellow"));
        m.add_state(State::new("red"));
        m.add_transition("green", Transition::new("TIMER", "yellow"));
        m.add_transition("yellow", Transition::new("TIMER", "red"));
        m.add_transition("red", Transition::new("TIMER", "green"));
        m
    }

    #[test]
    fn basic_transition() {
        let mut m = traffic_light();
        let actions = m.send("TIMER");
        assert!(actions.is_some());
        assert_eq!(m.current_state(), "yellow");
    }

    #[test]
    fn guard_blocks() {
        let mut m = StateMachine::new("locked");
        m.add_state(State::new("locked"));
        m.add_state(State::new("unlocked"));
        m.add_transition(
            "locked",
            Transition::new("UNLOCK", "unlocked")
                .with_guard(|ctx| ctx.get("has_key").and_then(|v| v.as_bool()).unwrap_or(false)),
        );
        // No key in context -> guard blocks
        assert_eq!(m.send("UNLOCK"), None);
        assert_eq!(m.current_state(), "locked");
    }

    #[test]
    fn guard_allows() {
        let mut m = StateMachine::new("locked");
        m.add_state(State::new("locked"));
        m.add_state(State::new("unlocked"));
        m.add_transition(
            "locked",
            Transition::new("UNLOCK", "unlocked")
                .with_guard(|ctx| ctx.get("has_key").and_then(|v| v.as_bool()).unwrap_or(false)),
        );
        m.set_context("has_key", json!(true));
        assert!(m.send("UNLOCK").is_some());
        assert_eq!(m.current_state(), "unlocked");
    }

    #[test]
    fn actions_returned() {
        let mut m = StateMachine::new("a");
        m.add_state(State::new("a"));
        m.add_state(State::new("b"));
        m.add_transition(
            "a",
            Transition::new("GO", "b").with_actions(vec![10, 20, 30]),
        );
        let actions = m.send("GO").unwrap();
        assert_eq!(actions, vec![10, 20, 30]);
    }

    #[test]
    fn on_enter_on_exit_ids() {
        let mut m = StateMachine::new("a");
        m.add_state(State::new("a").with_on_exit(1));
        m.add_state(State::new("b").with_on_enter(2));
        m.add_transition(
            "a",
            Transition::new("GO", "b").with_actions(vec![99]),
        );
        let actions = m.send("GO").unwrap();
        // on_exit(1), transition action(99), on_enter(2)
        assert_eq!(actions, vec![1, 99, 2]);
    }

    #[test]
    fn context_updates() {
        let mut m = StateMachine::new("s");
        m.set_context("count", json!(42));
        assert_eq!(m.context_value("count"), Some(&json!(42)));
        m.set_context("count", json!(43));
        assert_eq!(m.context_value("count"), Some(&json!(43)));
    }

    #[test]
    fn history_tracked() {
        let mut m = traffic_light();
        m.send("TIMER");
        m.send("TIMER");
        assert_eq!(
            m.history(),
            &["green", "yellow", "red"]
        );
    }

    #[test]
    fn can_send_checks() {
        let m = traffic_light();
        assert!(m.can_send("TIMER"));
        assert!(!m.can_send("NONEXISTENT"));
    }

    #[test]
    fn unavailable_event_returns_none() {
        let mut m = traffic_light();
        assert_eq!(m.send("BOGUS"), None);
        assert_eq!(m.current_state(), "green");
    }

    #[test]
    fn multiple_transitions_first_match_wins() {
        let mut m = StateMachine::new("s");
        m.add_state(State::new("s"));
        m.add_state(State::new("a"));
        m.add_state(State::new("b"));
        // First transition to "a", second to "b" — first wins.
        m.add_transition("s", Transition::new("GO", "a").with_actions(vec![1]));
        m.add_transition("s", Transition::new("GO", "b").with_actions(vec![2]));
        let actions = m.send("GO").unwrap();
        assert_eq!(actions, vec![1]);
        assert_eq!(m.current_state(), "a");
    }

    #[test]
    fn matches_helper() {
        let m = traffic_light();
        assert!(m.matches("green"));
        assert!(!m.matches("red"));
    }

    #[test]
    fn available_events_lists_valid() {
        let m = traffic_light();
        assert_eq!(m.available_events(), vec!["TIMER"]);
    }
}
