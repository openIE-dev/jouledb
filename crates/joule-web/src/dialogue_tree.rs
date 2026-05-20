//! Branching dialogue system — nodes, choices, conditions, effects, interpolation.
//!
//! Replaces Ink.js / Yarn Spinner / Twine-web with pure Rust.
//! Dialogue graph with NPC text, player choices, conditions, effects,
//! variable interpolation, history tracking, and graph validation.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DialogueError {
    NodeNotFound(String),
    NoChoicesAvailable(String),
    ChoiceIndexOutOfBounds { node: String, index: usize, count: usize },
    InvalidGraph(Vec<String>),
    EmptyGraph,
    NoEntryPoint,
    CycleDetected(String),
}

impl fmt::Display for DialogueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "dialogue node not found: {id}"),
            Self::NoChoicesAvailable(id) => write!(f, "no available choices at node: {id}"),
            Self::ChoiceIndexOutOfBounds { node, index, count } => {
                write!(f, "choice {index} out of bounds at node {node} ({count} choices)")
            }
            Self::InvalidGraph(errors) => {
                write!(f, "invalid dialogue graph: {}", errors.join("; "))
            }
            Self::EmptyGraph => write!(f, "dialogue graph is empty"),
            Self::NoEntryPoint => write!(f, "no entry point defined"),
            Self::CycleDetected(id) => write!(f, "cycle detected at node: {id}"),
        }
    }
}

impl std::error::Error for DialogueError {}

// ── Conditions ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    HasItem(u64),
    StatAbove(String, i64),
    FlagSet(String),
    FlagNotSet(String),
    And(Vec<Condition>),
    Or(Vec<Condition>),
}

impl Condition {
    pub fn evaluate(&self, state: &DialogueState) -> bool {
        match self {
            Self::HasItem(id) => state.items.contains(id),
            Self::StatAbove(name, threshold) => {
                state.stats.get(name).copied().unwrap_or(0) > *threshold
            }
            Self::FlagSet(name) => state.flags.contains_key(name) && state.flags[name],
            Self::FlagNotSet(name) => !state.flags.get(name).copied().unwrap_or(false),
            Self::And(conditions) => conditions.iter().all(|c| c.evaluate(state)),
            Self::Or(conditions) => conditions.iter().any(|c| c.evaluate(state)),
        }
    }
}

// ── Effects ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    SetFlag(String, bool),
    GiveItem(u64),
    RemoveItem(u64),
    ModifyStat(String, i64),
    SetStat(String, i64),
}

impl Effect {
    pub fn apply(&self, state: &mut DialogueState) {
        match self {
            Self::SetFlag(name, val) => { state.flags.insert(name.clone(), *val); }
            Self::GiveItem(id) => { state.items.push(*id); }
            Self::RemoveItem(id) => { state.items.retain(|i| i != id); }
            Self::ModifyStat(name, delta) => {
                let entry = state.stats.entry(name.clone()).or_insert(0);
                *entry += delta;
            }
            Self::SetStat(name, val) => { state.stats.insert(name.clone(), *val); }
        }
    }
}

// ── Dialogue State ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct DialogueState {
    pub flags: HashMap<String, bool>,
    pub stats: HashMap<String, i64>,
    pub items: Vec<u64>,
    pub variables: HashMap<String, String>,
}

impl DialogueState {
    pub fn new() -> Self {
        Self {
            flags: HashMap::new(),
            stats: HashMap::new(),
            items: Vec::new(),
            variables: HashMap::new(),
        }
    }

    pub fn with_variable(mut self, key: &str, value: &str) -> Self {
        self.variables.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_flag(mut self, key: &str, val: bool) -> Self {
        self.flags.insert(key.to_string(), val);
        self
    }

    pub fn with_stat(mut self, key: &str, val: i64) -> Self {
        self.stats.insert(key.to_string(), val);
        self
    }

    pub fn with_item(mut self, id: u64) -> Self {
        self.items.push(id);
        self
    }
}

// ── Choice ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub text: String,
    pub next_node: String,
    pub conditions: Vec<Condition>,
    pub effects: Vec<Effect>,
}

impl Choice {
    pub fn new(text: &str, next_node: &str) -> Self {
        Self {
            text: text.to_string(),
            next_node: next_node.to_string(),
            conditions: Vec::new(),
            effects: Vec::new(),
        }
    }

    pub fn with_condition(mut self, cond: Condition) -> Self {
        self.conditions.push(cond);
        self
    }

    pub fn with_effect(mut self, effect: Effect) -> Self {
        self.effects.push(effect);
        self
    }

    pub fn is_available(&self, state: &DialogueState) -> bool {
        self.conditions.iter().all(|c| c.evaluate(state))
    }
}

// ── Dialogue Node ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct DialogueNode {
    pub id: String,
    pub speaker: String,
    pub text: String,
    pub choices: Vec<Choice>,
    pub on_enter_effects: Vec<Effect>,
    pub is_terminal: bool,
}

impl DialogueNode {
    pub fn new(id: &str, speaker: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            speaker: speaker.to_string(),
            text: text.to_string(),
            choices: Vec::new(),
            on_enter_effects: Vec::new(),
            is_terminal: false,
        }
    }

    pub fn terminal(id: &str, speaker: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            speaker: speaker.to_string(),
            text: text.to_string(),
            choices: Vec::new(),
            on_enter_effects: Vec::new(),
            is_terminal: true,
        }
    }

    pub fn with_choice(mut self, choice: Choice) -> Self {
        self.choices.push(choice);
        self
    }

    pub fn with_enter_effect(mut self, effect: Effect) -> Self {
        self.on_enter_effects.push(effect);
        self
    }

    /// Interpolate variables in text: {variable_name} → value
    pub fn interpolated_text(&self, state: &DialogueState) -> String {
        let mut result = self.text.clone();
        for (key, val) in &state.variables {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, val);
        }
        // Also interpolate stats
        for (key, val) in &state.stats {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, &val.to_string());
        }
        result
    }

    pub fn available_choices(&self, state: &DialogueState) -> Vec<(usize, &Choice)> {
        self.choices.iter().enumerate()
            .filter(|(_, c)| c.is_available(state))
            .collect()
    }
}

// ── Entry Point ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct EntryPoint {
    pub node_id: String,
    pub conditions: Vec<Condition>,
    pub priority: i32,
}

impl EntryPoint {
    pub fn new(node_id: &str, priority: i32) -> Self {
        Self {
            node_id: node_id.to_string(),
            conditions: Vec::new(),
            priority,
        }
    }

    pub fn with_condition(mut self, cond: Condition) -> Self {
        self.conditions.push(cond);
        self
    }
}

// ── Dialogue Graph ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DialogueGraph {
    nodes: HashMap<String, DialogueNode>,
    entry_points: Vec<EntryPoint>,
    history: Vec<String>,
}

impl DialogueGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry_points: Vec::new(),
            history: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: DialogueNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn add_entry_point(&mut self, entry: EntryPoint) {
        self.entry_points.push(entry);
        self.entry_points.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    pub fn get_node(&self, id: &str) -> Option<&DialogueNode> {
        self.nodes.get(id)
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Select the best entry point based on state conditions.
    pub fn select_entry(&self, state: &DialogueState) -> Result<&str, DialogueError> {
        if self.entry_points.is_empty() {
            return Err(DialogueError::NoEntryPoint);
        }
        for ep in &self.entry_points {
            if ep.conditions.iter().all(|c| c.evaluate(state)) {
                return Ok(&ep.node_id);
            }
        }
        // Fall through to lowest priority unconditional
        Err(DialogueError::NoEntryPoint)
    }

    /// Enter a node: apply on_enter effects, record history, return the node.
    pub fn enter_node(&mut self, node_id: &str, state: &mut DialogueState) -> Result<&DialogueNode, DialogueError> {
        if !self.nodes.contains_key(node_id) {
            return Err(DialogueError::NodeNotFound(node_id.to_string()));
        }
        // Apply on_enter effects (clone to avoid borrow conflict)
        let effects: Vec<Effect> = self.nodes[node_id].on_enter_effects.clone();
        for effect in &effects {
            effect.apply(state);
        }
        self.history.push(node_id.to_string());
        Ok(&self.nodes[node_id])
    }

    /// Choose a choice at a node, apply its effects, return the next node ID.
    pub fn choose(&self, node_id: &str, choice_idx: usize, state: &mut DialogueState) -> Result<String, DialogueError> {
        let node = self.nodes.get(node_id)
            .ok_or_else(|| DialogueError::NodeNotFound(node_id.to_string()))?;
        let available: Vec<(usize, &Choice)> = node.available_choices(state);
        if available.is_empty() {
            return Err(DialogueError::NoChoicesAvailable(node_id.to_string()));
        }
        if choice_idx >= available.len() {
            return Err(DialogueError::ChoiceIndexOutOfBounds {
                node: node_id.to_string(),
                index: choice_idx,
                count: available.len(),
            });
        }
        let (_, choice) = available[choice_idx];
        let next = choice.next_node.clone();
        let effects = choice.effects.clone();
        for effect in &effects {
            effect.apply(state);
        }
        Ok(next)
    }

    pub fn history(&self) -> &[String] { &self.history }
    pub fn clear_history(&mut self) { self.history.clear(); }

    pub fn was_visited(&self, node_id: &str) -> bool {
        self.history.iter().any(|h| h == node_id)
    }

    pub fn visit_count(&self, node_id: &str) -> usize {
        self.history.iter().filter(|h| h.as_str() == node_id).count()
    }

    /// Validate graph: check all references are valid, no non-terminal dead ends.
    pub fn validate(&self) -> Result<(), DialogueError> {
        if self.nodes.is_empty() {
            return Err(DialogueError::EmptyGraph);
        }
        let mut errors = Vec::new();
        // Collect all node IDs first
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        for nid in &node_ids {
            let node = &self.nodes[nid];
            if !node.is_terminal && node.choices.is_empty() {
                errors.push(format!("node '{nid}' has no choices and is not terminal"));
            }
            for (ci, choice) in node.choices.iter().enumerate() {
                if !self.nodes.contains_key(&choice.next_node) {
                    errors.push(format!("node '{nid}' choice {ci} references missing node '{}'", choice.next_node));
                }
            }
        }
        for ep in &self.entry_points {
            if !self.nodes.contains_key(&ep.node_id) {
                errors.push(format!("entry point references missing node '{}'", ep.node_id));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(DialogueError::InvalidGraph(errors))
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_graph() -> DialogueGraph {
        let mut g = DialogueGraph::new();
        g.add_node(
            DialogueNode::new("start", "NPC", "Hello, {player_name}! What brings you here?")
                .with_choice(Choice::new("I need a quest.", "quest_offer"))
                .with_choice(
                    Choice::new("I have the artifact.", "artifact_return")
                        .with_condition(Condition::HasItem(42))
                )
        );
        g.add_node(
            DialogueNode::new("quest_offer", "NPC", "Find the lost artifact in the cave.")
                .with_enter_effect(Effect::SetFlag("quest_given".to_string(), true))
                .with_choice(Choice::new("I'll do it!", "accept"))
                .with_choice(Choice::new("No thanks.", "decline"))
        );
        g.add_node(
            DialogueNode::terminal("accept", "NPC", "Good luck, {player_name}!")
                .with_enter_effect(Effect::ModifyStat("reputation".to_string(), 5))
        );
        g.add_node(DialogueNode::terminal("decline", "NPC", "Perhaps another time."));
        g.add_node(
            DialogueNode::terminal("artifact_return", "NPC", "You found it! Here's your reward.")
                .with_enter_effect(Effect::GiveItem(100))
                .with_enter_effect(Effect::SetFlag("quest_complete".to_string(), true))
        );
        g.add_entry_point(EntryPoint::new("start", 0));
        g
    }

    #[test]
    fn create_graph() {
        let g = build_graph();
        assert_eq!(g.node_count(), 5);
    }

    #[test]
    fn validate_ok() {
        let g = build_graph();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn validate_empty() {
        let g = DialogueGraph::new();
        let err = g.validate().unwrap_err();
        assert!(matches!(err, DialogueError::EmptyGraph));
    }

    #[test]
    fn validate_dead_end() {
        let mut g = DialogueGraph::new();
        g.add_node(DialogueNode::new("dead", "NPC", "No way out."));
        let err = g.validate().unwrap_err();
        assert!(matches!(err, DialogueError::InvalidGraph(_)));
    }

    #[test]
    fn validate_missing_reference() {
        let mut g = DialogueGraph::new();
        g.add_node(
            DialogueNode::new("a", "NPC", "Hi")
                .with_choice(Choice::new("Go", "nonexistent"))
        );
        let err = g.validate().unwrap_err();
        if let DialogueError::InvalidGraph(errors) = &err {
            assert!(errors.iter().any(|e| e.contains("nonexistent")));
        }
    }

    #[test]
    fn select_entry_default() {
        let g = build_graph();
        let state = DialogueState::new();
        let entry = g.select_entry(&state).unwrap();
        assert_eq!(entry, "start");
    }

    #[test]
    fn select_entry_conditional() {
        let mut g = build_graph();
        g.add_entry_point(
            EntryPoint::new("artifact_return", 10)
                .with_condition(Condition::HasItem(42))
        );
        let state = DialogueState::new().with_item(42);
        let entry = g.select_entry(&state).unwrap();
        assert_eq!(entry, "artifact_return"); // higher priority
    }

    #[test]
    fn enter_node_records_history() {
        let mut g = build_graph();
        let mut state = DialogueState::new().with_variable("player_name", "Hero");
        g.enter_node("start", &mut state).unwrap();
        assert!(g.was_visited("start"));
        assert_eq!(g.visit_count("start"), 1);
    }

    #[test]
    fn enter_node_applies_effects() {
        let mut g = build_graph();
        let mut state = DialogueState::new();
        g.enter_node("quest_offer", &mut state).unwrap();
        assert_eq!(state.flags.get("quest_given"), Some(&true));
    }

    #[test]
    fn interpolation() {
        let g = build_graph();
        let state = DialogueState::new().with_variable("player_name", "Aria");
        let node = g.get_node("start").unwrap();
        let text = node.interpolated_text(&state);
        assert_eq!(text, "Hello, Aria! What brings you here?");
    }

    #[test]
    fn interpolation_with_stats() {
        let mut g = DialogueGraph::new();
        g.add_node(DialogueNode::terminal("s", "NPC", "Your rep is {reputation}."));
        let state = DialogueState::new().with_stat("reputation", 42);
        let node = g.get_node("s").unwrap();
        assert_eq!(node.interpolated_text(&state), "Your rep is 42.");
    }

    #[test]
    fn available_choices_filtered() {
        let g = build_graph();
        let state = DialogueState::new(); // no item 42
        let node = g.get_node("start").unwrap();
        let available = node.available_choices(&state);
        assert_eq!(available.len(), 1); // only "I need a quest"
    }

    #[test]
    fn available_choices_with_item() {
        let g = build_graph();
        let state = DialogueState::new().with_item(42);
        let node = g.get_node("start").unwrap();
        let available = node.available_choices(&state);
        assert_eq!(available.len(), 2);
    }

    #[test]
    fn choose_valid() {
        let g = build_graph();
        let mut state = DialogueState::new();
        let next = g.choose("start", 0, &mut state).unwrap();
        assert_eq!(next, "quest_offer");
    }

    #[test]
    fn choose_out_of_bounds() {
        let g = build_graph();
        let mut state = DialogueState::new();
        let err = g.choose("start", 5, &mut state).unwrap_err();
        assert!(matches!(err, DialogueError::ChoiceIndexOutOfBounds { .. }));
    }

    #[test]
    fn choose_applies_effects() {
        let mut g = DialogueGraph::new();
        g.add_node(
            DialogueNode::new("n", "NPC", "Hi")
                .with_choice(
                    Choice::new("Give gold", "end")
                        .with_effect(Effect::ModifyStat("gold".to_string(), 100))
                )
        );
        g.add_node(DialogueNode::terminal("end", "NPC", "Thanks!"));
        let mut state = DialogueState::new();
        g.choose("n", 0, &mut state).unwrap();
        assert_eq!(*state.stats.get("gold").unwrap(), 100);
    }

    #[test]
    fn node_not_found() {
        let mut g = build_graph();
        let mut state = DialogueState::new();
        let err = g.enter_node("missing", &mut state).unwrap_err();
        assert!(matches!(err, DialogueError::NodeNotFound(_)));
    }

    #[test]
    fn full_dialogue_walkthrough() {
        let mut g = build_graph();
        let mut state = DialogueState::new()
            .with_variable("player_name", "Hero");

        let entry = g.select_entry(&state).unwrap().to_string();
        let node = g.enter_node(&entry, &mut state).unwrap();
        assert!(node.interpolated_text(&state).contains("Hero"));

        let next = g.choose(&entry, 0, &mut state).unwrap();
        assert_eq!(next, "quest_offer");

        g.enter_node(&next, &mut state).unwrap();
        assert!(state.flags.get("quest_given") == Some(&true));

        let final_id = g.choose(&next, 0, &mut state).unwrap();
        assert_eq!(final_id, "accept");

        g.enter_node(&final_id, &mut state).unwrap();
        assert_eq!(*state.stats.get("reputation").unwrap(), 5);
        assert_eq!(g.history().len(), 3);
    }

    #[test]
    fn condition_and_or() {
        let state = DialogueState::new()
            .with_flag("a", true)
            .with_flag("b", false);
        let cond_and = Condition::And(vec![
            Condition::FlagSet("a".to_string()),
            Condition::FlagSet("b".to_string()),
        ]);
        assert!(!cond_and.evaluate(&state));

        let cond_or = Condition::Or(vec![
            Condition::FlagSet("a".to_string()),
            Condition::FlagSet("b".to_string()),
        ]);
        assert!(cond_or.evaluate(&state));
    }

    #[test]
    fn condition_stat_above() {
        let state = DialogueState::new().with_stat("str", 10);
        assert!(Condition::StatAbove("str".to_string(), 5).evaluate(&state));
        assert!(!Condition::StatAbove("str".to_string(), 10).evaluate(&state));
        assert!(!Condition::StatAbove("str".to_string(), 15).evaluate(&state));
    }

    #[test]
    fn condition_flag_not_set() {
        let state = DialogueState::new();
        assert!(Condition::FlagNotSet("unset".to_string()).evaluate(&state));
        let state2 = DialogueState::new().with_flag("set", true);
        assert!(!Condition::FlagNotSet("set".to_string()).evaluate(&state2));
    }

    #[test]
    fn effect_remove_item() {
        let mut state = DialogueState::new().with_item(42).with_item(43);
        Effect::RemoveItem(42).apply(&mut state);
        assert!(!state.items.contains(&42));
        assert!(state.items.contains(&43));
    }

    #[test]
    fn effect_set_stat() {
        let mut state = DialogueState::new().with_stat("hp", 50);
        Effect::SetStat("hp".to_string(), 100).apply(&mut state);
        assert_eq!(*state.stats.get("hp").unwrap(), 100);
    }

    #[test]
    fn clear_history() {
        let mut g = build_graph();
        let mut state = DialogueState::new();
        g.enter_node("start", &mut state).unwrap();
        assert!(!g.history().is_empty());
        g.clear_history();
        assert!(g.history().is_empty());
    }
}
