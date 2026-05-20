//! Finite state machine inspired by XState.
//!
//! Provides states, transitions, guards, actions, nested/parallel states,
//! history states, event processing, and DOT export for visualization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Action Record ────────────────────────────────────────────

/// A record of an action that was fired during a transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub action_id: String,
    pub from_state: String,
    pub to_state: String,
    pub event: String,
}

// ── Guard ────────────────────────────────────────────────────

/// A guard predicate that can block a transition.
pub struct Guard {
    pub name: String,
    pub predicate: Box<dyn Fn(&Context) -> bool>,
}

impl Guard {
    pub fn new(name: impl Into<String>, predicate: impl Fn(&Context) -> bool + 'static) -> Self {
        Self {
            name: name.into(),
            predicate: Box::new(predicate),
        }
    }

    pub fn evaluate(&self, ctx: &Context) -> bool {
        (self.predicate)(ctx)
    }
}

// ── Context ──────────────────────────────────────────────────

/// Arbitrary key-value context for the state machine.
pub type Context = HashMap<String, serde_json::Value>;

// ── Transition ───────────────────────────────────────────────

/// A transition from one state to another on an event.
pub struct Transition {
    pub event: String,
    pub target: String,
    pub guard: Option<Guard>,
    pub actions: Vec<String>,
}

impl Transition {
    pub fn new(event: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            event: event.into(),
            target: target.into(),
            guard: None,
            actions: Vec::new(),
        }
    }

    pub fn with_guard(mut self, guard: Guard) -> Self {
        self.guard = Some(guard);
        self
    }

    pub fn with_actions(mut self, actions: Vec<String>) -> Self {
        self.actions = actions;
        self
    }

    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.actions.push(action.into());
        self
    }
}

// ── State ────────────────────────────────────────────────────

/// A state in the machine.
pub struct State {
    pub name: String,
    pub on_enter: Vec<String>,
    pub on_exit: Vec<String>,
    pub transitions: Vec<Transition>,
    /// Nested child states (for hierarchical/compound states).
    pub children: Vec<String>,
    /// Initial child state (for compound states).
    pub initial_child: Option<String>,
    /// Whether this is a final state.
    pub is_final: bool,
    /// History type for this state.
    pub history: HistoryType,
}

/// History state type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryType {
    /// No history tracking.
    None,
    /// Shallow: remembers the last direct child state.
    Shallow,
    /// Deep: remembers the full nested state path.
    Deep,
}

impl State {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            on_enter: Vec::new(),
            on_exit: Vec::new(),
            transitions: Vec::new(),
            children: Vec::new(),
            initial_child: None,
            is_final: false,
            history: HistoryType::None,
        }
    }

    pub fn with_on_enter(mut self, action: impl Into<String>) -> Self {
        self.on_enter.push(action.into());
        self
    }

    pub fn with_on_exit(mut self, action: impl Into<String>) -> Self {
        self.on_exit.push(action.into());
        self
    }

    pub fn with_transition(mut self, transition: Transition) -> Self {
        self.transitions.push(transition);
        self
    }

    pub fn with_child(mut self, child: impl Into<String>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn with_initial_child(mut self, child: impl Into<String>) -> Self {
        self.initial_child = Some(child.into());
        self
    }

    pub fn as_final(mut self) -> Self {
        self.is_final = true;
        self
    }

    pub fn with_history(mut self, history: HistoryType) -> Self {
        self.history = history;
        self
    }
}

// ── Machine ──────────────────────────────────────────────────

/// A finite state machine.
pub struct Machine {
    states: HashMap<String, State>,
    current: String,
    context: Context,
    action_log: Vec<ActionRecord>,
    /// History memory: state_name -> last active child.
    history_memory: HashMap<String, String>,
    /// For parallel states: set of currently active states.
    parallel_states: Vec<String>,
    /// Whether this machine uses parallel regions.
    is_parallel: bool,
}

impl Machine {
    /// Create a new machine with an initial state.
    pub fn new(initial: impl Into<String>) -> Self {
        let initial = initial.into();
        Self {
            states: HashMap::new(),
            current: initial,
            context: HashMap::new(),
            action_log: Vec::new(),
            history_memory: HashMap::new(),
            parallel_states: Vec::new(),
            is_parallel: false,
        }
    }

    /// Add a state to the machine.
    pub fn add_state(&mut self, state: State) {
        self.states.insert(state.name.clone(), state);
    }

    /// Set the machine to use parallel regions.
    pub fn set_parallel(&mut self, regions: Vec<String>) {
        self.is_parallel = true;
        self.parallel_states = regions;
    }

    /// Get the current state name.
    pub fn current_state(&self) -> &str {
        &self.current
    }

    /// Get the context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get a mutable reference to the context.
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Set a context value.
    pub fn set_context(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.context.insert(key.into(), value);
    }

    /// Get the action log.
    pub fn action_log(&self) -> &[ActionRecord] {
        &self.action_log
    }

    /// Check if the machine is in a final state.
    pub fn is_done(&self) -> bool {
        self.states
            .get(&self.current)
            .is_some_and(|s| s.is_final)
    }

    /// Get all currently active states (for parallel machines).
    pub fn active_states(&self) -> Vec<String> {
        if self.is_parallel {
            self.parallel_states.clone()
        } else {
            vec![self.current.clone()]
        }
    }

    /// Send an event to the machine. Returns true if a transition occurred.
    pub fn send(&mut self, event: &str) -> bool {
        if self.is_parallel {
            return self.send_parallel(event);
        }

        let current_name = self.current.clone();
        let state = match self.states.get(&current_name) {
            Some(s) => s,
            None => return false,
        };

        // Find matching transition
        let mut target = None;
        let mut actions_to_fire = Vec::new();

        for transition in &state.transitions {
            if transition.event == event {
                // Check guard
                let guard_passes = transition
                    .guard
                    .as_ref()
                    .map_or(true, |g| g.evaluate(&self.context));

                if guard_passes {
                    target = Some(transition.target.clone());
                    actions_to_fire = transition.actions.clone();
                    break;
                }
            }
        }

        if let Some(target_name) = target {
            // Fire exit actions
            if let Some(state) = self.states.get(&current_name) {
                for action in &state.on_exit {
                    self.action_log.push(ActionRecord {
                        action_id: action.clone(),
                        from_state: current_name.clone(),
                        to_state: target_name.clone(),
                        event: event.to_string(),
                    });
                }
            }

            // Fire transition actions
            for action in &actions_to_fire {
                self.action_log.push(ActionRecord {
                    action_id: action.clone(),
                    from_state: current_name.clone(),
                    to_state: target_name.clone(),
                    event: event.to_string(),
                });
            }

            // Save history for the parent of the current state
            self.save_history(&current_name);

            // Resolve target (handle history states)
            let resolved_target = self.resolve_target(&target_name);

            // Fire enter actions
            if let Some(state) = self.states.get(&resolved_target) {
                for action in &state.on_enter {
                    self.action_log.push(ActionRecord {
                        action_id: action.clone(),
                        from_state: current_name.clone(),
                        to_state: resolved_target.clone(),
                        event: event.to_string(),
                    });
                }
            }

            self.current = resolved_target;
            true
        } else {
            false
        }
    }

    fn send_parallel(&mut self, event: &str) -> bool {
        let mut any_transitioned = false;
        let active = self.parallel_states.clone();

        for state_name in &active {
            let state = match self.states.get(state_name) {
                Some(s) => s,
                None => continue,
            };

            for transition in &state.transitions {
                if transition.event == event {
                    let guard_passes = transition
                        .guard
                        .as_ref()
                        .map_or(true, |g| g.evaluate(&self.context));

                    if guard_passes {
                        // Update the parallel state
                        if let Some(pos) = self
                            .parallel_states
                            .iter()
                            .position(|s| s == state_name)
                        {
                            for action in &transition.actions {
                                self.action_log.push(ActionRecord {
                                    action_id: action.clone(),
                                    from_state: state_name.clone(),
                                    to_state: transition.target.clone(),
                                    event: event.to_string(),
                                });
                            }
                            self.parallel_states[pos] = transition.target.clone();
                            any_transitioned = true;
                        }
                        break;
                    }
                }
            }
        }

        any_transitioned
    }

    fn save_history(&mut self, state_name: &str) {
        // Find parent states that have history enabled
        for (name, state) in &self.states {
            if state.children.contains(&state_name.to_string())
                && state.history != HistoryType::None
            {
                self.history_memory
                    .insert(name.clone(), state_name.to_string());
            }
        }
    }

    fn resolve_target(&self, target: &str) -> String {
        // Check if target state has an initial child
        if let Some(state) = self.states.get(target) {
            // If this state has history and a remembered child, go there
            if state.history != HistoryType::None {
                if let Some(remembered) = self.history_memory.get(target) {
                    return remembered.clone();
                }
            }

            // Otherwise, enter the initial child if it exists
            if let Some(initial) = &state.initial_child {
                return initial.clone();
            }
        }
        target.to_string()
    }

    /// Get all registered state names.
    pub fn state_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.states.keys().cloned().collect();
        names.sort();
        names
    }

    /// Check if an event can trigger a transition from the current state.
    pub fn can_handle(&self, event: &str) -> bool {
        if let Some(state) = self.states.get(&self.current) {
            state.transitions.iter().any(|t| {
                t.event == event
                    && t.guard
                        .as_ref()
                        .map_or(true, |g| g.evaluate(&self.context))
            })
        } else {
            false
        }
    }

    /// Get all events that the current state can handle.
    pub fn available_events(&self) -> Vec<String> {
        if let Some(state) = self.states.get(&self.current) {
            let mut events: Vec<String> = state
                .transitions
                .iter()
                .map(|t| t.event.clone())
                .collect();
            events.sort();
            events.dedup();
            events
        } else {
            Vec::new()
        }
    }

    /// Reset the machine to a given state.
    pub fn reset(&mut self, state: impl Into<String>) {
        self.current = state.into();
        self.action_log.clear();
        self.history_memory.clear();
    }

    /// Export the state chart as a DOT graph.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph fsm {\n  rankdir=LR;\n");

        // Mark current state
        dot.push_str(&format!(
            "  \"{}\" [style=filled fillcolor=lightblue];\n",
            self.current
        ));

        // Sorted state names for deterministic output
        let mut state_names: Vec<&String> = self.states.keys().collect();
        state_names.sort();

        for name in &state_names {
            let state = &self.states[*name];

            if state.is_final {
                dot.push_str(&format!(
                    "  \"{}\" [shape=doublecircle];\n",
                    name
                ));
            }

            // Sort transitions by event for deterministic output
            let mut transitions: Vec<(&str, &str)> = state
                .transitions
                .iter()
                .map(|t| (t.event.as_str(), t.target.as_str()))
                .collect();
            transitions.sort();

            for (event, target) in transitions {
                dot.push_str(&format!(
                    "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                    name, target, event
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }
}

// ── Builder ──────────────────────────────────────────────────

/// Builder for constructing a state machine fluently.
pub struct MachineBuilder {
    machine: Machine,
}

impl MachineBuilder {
    pub fn new(initial: impl Into<String>) -> Self {
        Self {
            machine: Machine::new(initial),
        }
    }

    pub fn state(mut self, state: State) -> Self {
        self.machine.add_state(state);
        self
    }

    pub fn context(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.machine.set_context(key, value);
        self
    }

    pub fn parallel(mut self, regions: Vec<String>) -> Self {
        self.machine.set_parallel(regions);
        self
    }

    pub fn build(self) -> Machine {
        self.machine
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn traffic_light() -> Machine {
        MachineBuilder::new("green")
            .state(
                State::new("green")
                    .with_transition(Transition::new("TIMER", "yellow"))
                    .with_on_enter("start_green_timer"),
            )
            .state(
                State::new("yellow")
                    .with_transition(Transition::new("TIMER", "red"))
                    .with_on_enter("start_yellow_timer"),
            )
            .state(
                State::new("red")
                    .with_transition(Transition::new("TIMER", "green"))
                    .with_on_enter("start_red_timer"),
            )
            .build()
    }

    #[test]
    fn basic_transitions() {
        let mut machine = traffic_light();
        assert_eq!(machine.current_state(), "green");
        assert!(machine.send("TIMER"));
        assert_eq!(machine.current_state(), "yellow");
        assert!(machine.send("TIMER"));
        assert_eq!(machine.current_state(), "red");
        assert!(machine.send("TIMER"));
        assert_eq!(machine.current_state(), "green");
    }

    #[test]
    fn unknown_event_returns_false() {
        let mut machine = traffic_light();
        assert!(!machine.send("UNKNOWN"));
        assert_eq!(machine.current_state(), "green");
    }

    #[test]
    fn guard_blocks_transition() {
        let mut machine = MachineBuilder::new("locked")
            .state(State::new("locked").with_transition(
                Transition::new("UNLOCK", "unlocked").with_guard(Guard::new(
                    "has_key",
                    |ctx| {
                        ctx.get("has_key")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                    },
                )),
            ))
            .state(State::new("unlocked"))
            .build();

        // Without key, transition should fail
        assert!(!machine.send("UNLOCK"));
        assert_eq!(machine.current_state(), "locked");

        // With key, transition should succeed
        machine.set_context("has_key", serde_json::Value::Bool(true));
        assert!(machine.send("UNLOCK"));
        assert_eq!(machine.current_state(), "unlocked");
    }

    #[test]
    fn transition_actions_logged() {
        let mut machine = MachineBuilder::new("idle")
            .state(State::new("idle").with_transition(
                Transition::new("START", "running").with_action("log_start"),
            ))
            .state(State::new("running"))
            .build();

        machine.send("START");
        let log = machine.action_log();
        assert!(log.iter().any(|a| a.action_id == "log_start"));
    }

    #[test]
    fn on_enter_on_exit_actions() {
        let mut machine = MachineBuilder::new("a")
            .state(
                State::new("a")
                    .with_on_exit("exit_a")
                    .with_transition(Transition::new("GO", "b")),
            )
            .state(State::new("b").with_on_enter("enter_b"))
            .build();

        machine.send("GO");
        let log = machine.action_log();
        assert!(log.iter().any(|a| a.action_id == "exit_a"));
        assert!(log.iter().any(|a| a.action_id == "enter_b"));
    }

    #[test]
    fn final_state() {
        let mut machine = MachineBuilder::new("running")
            .state(
                State::new("running")
                    .with_transition(Transition::new("FINISH", "done")),
            )
            .state(State::new("done").as_final())
            .build();

        assert!(!machine.is_done());
        machine.send("FINISH");
        assert!(machine.is_done());
    }

    #[test]
    fn nested_states_initial_child() {
        let mut machine = MachineBuilder::new("active")
            .state(
                State::new("active")
                    .with_child("editing")
                    .with_child("saving")
                    .with_initial_child("editing")
                    .with_transition(Transition::new("DEACTIVATE", "inactive")),
            )
            .state(
                State::new("editing")
                    .with_transition(Transition::new("SAVE", "saving")),
            )
            .state(State::new("saving"))
            .state(
                State::new("inactive")
                    .with_transition(Transition::new("ACTIVATE", "active")),
            )
            .build();

        // Entering "active" should resolve to "editing" (initial child)
        assert_eq!(machine.current_state(), "active");
        // Sending ACTIVATE from inactive -> active -> editing
        machine.current = "inactive".to_string();
        machine.send("ACTIVATE");
        assert_eq!(machine.current_state(), "editing");
    }

    #[test]
    fn history_state_shallow() {
        let mut machine = Machine::new("idle");
        machine.add_state(
            State::new("active")
                .with_child("editing")
                .with_child("previewing")
                .with_initial_child("editing")
                .with_history(HistoryType::Shallow),
        );
        machine.add_state(
            State::new("editing")
                .with_transition(Transition::new("PREVIEW", "previewing"))
                .with_transition(Transition::new("CLOSE", "idle")),
        );
        machine.add_state(
            State::new("previewing")
                .with_transition(Transition::new("EDIT", "editing"))
                .with_transition(Transition::new("CLOSE", "idle")),
        );
        machine.add_state(
            State::new("idle")
                .with_transition(Transition::new("OPEN", "active")),
        );

        machine.current = "editing".to_string();
        machine.send("PREVIEW"); // now in previewing
        assert_eq!(machine.current_state(), "previewing");
        machine.send("CLOSE"); // back to idle, history saves "previewing"
        assert_eq!(machine.current_state(), "idle");

        machine.send("OPEN"); // should restore to "previewing" via history
        assert_eq!(machine.current_state(), "previewing");
    }

    #[test]
    fn parallel_states() {
        let mut machine = MachineBuilder::new("idle")
            .state(
                State::new("bold_off")
                    .with_transition(Transition::new("TOGGLE_BOLD", "bold_on")),
            )
            .state(
                State::new("bold_on")
                    .with_transition(Transition::new("TOGGLE_BOLD", "bold_off")),
            )
            .state(
                State::new("italic_off")
                    .with_transition(Transition::new("TOGGLE_ITALIC", "italic_on")),
            )
            .state(
                State::new("italic_on")
                    .with_transition(Transition::new("TOGGLE_ITALIC", "italic_off")),
            )
            .parallel(vec!["bold_off".into(), "italic_off".into()])
            .build();

        machine.send("TOGGLE_BOLD");
        let active = machine.active_states();
        assert!(active.contains(&"bold_on".to_string()));
        assert!(active.contains(&"italic_off".to_string()));

        machine.send("TOGGLE_ITALIC");
        let active = machine.active_states();
        assert!(active.contains(&"bold_on".to_string()));
        assert!(active.contains(&"italic_on".to_string()));
    }

    #[test]
    fn can_handle_event() {
        let machine = traffic_light();
        assert!(machine.can_handle("TIMER"));
        assert!(!machine.can_handle("UNKNOWN"));
    }

    #[test]
    fn available_events() {
        let machine = traffic_light();
        let events = machine.available_events();
        assert_eq!(events, vec!["TIMER"]);
    }

    #[test]
    fn dot_export() {
        let machine = traffic_light();
        let dot = machine.to_dot();
        assert!(dot.contains("digraph fsm"));
        assert!(dot.contains("green"));
        assert!(dot.contains("yellow"));
        assert!(dot.contains("red"));
        assert!(dot.contains("TIMER"));
    }

    #[test]
    fn reset_machine() {
        let mut machine = traffic_light();
        machine.send("TIMER");
        machine.send("TIMER");
        assert_eq!(machine.current_state(), "red");
        machine.reset("green");
        assert_eq!(machine.current_state(), "green");
        assert!(machine.action_log().is_empty());
    }

    #[test]
    fn state_names_sorted() {
        let machine = traffic_light();
        let names = machine.state_names();
        assert_eq!(names, vec!["green", "red", "yellow"]);
    }

    #[test]
    fn context_operations() {
        let mut machine = Machine::new("idle");
        machine.set_context("count", serde_json::json!(0));
        assert_eq!(machine.context().get("count"), Some(&serde_json::json!(0)));
        machine.context_mut().insert("name".into(), serde_json::json!("test"));
        assert_eq!(
            machine.context().get("name"),
            Some(&serde_json::json!("test"))
        );
    }
}
