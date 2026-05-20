//! Petri nets — places, transitions, tokens, firing rules, analysis.
//!
//! Replaces PetriNet.js / PIPE / pnml-js with pure Rust.
//! Supports places with tokens, transitions with arcs, firing rules,
//! reachability analysis, deadlock detection, liveness checking,
//! inhibitor arcs, and Petri net visualization (DOT export).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for Petri nets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PetriError {
    /// Place not found.
    PlaceNotFound(String),
    /// Transition not found.
    TransitionNotFound(String),
    /// Duplicate place.
    DuplicatePlace(String),
    /// Duplicate transition.
    DuplicateTransition(String),
    /// Arc already exists.
    DuplicateArc { from: String, to: String },
    /// Transition not enabled.
    NotEnabled(String),
    /// Insufficient tokens.
    InsufficientTokens { place: String, required: u32, available: u32 },
}

impl fmt::Display for PetriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlaceNotFound(p) => write!(f, "place not found: {p}"),
            Self::TransitionNotFound(t) => write!(f, "transition not found: {t}"),
            Self::DuplicatePlace(p) => write!(f, "duplicate place: {p}"),
            Self::DuplicateTransition(t) => write!(f, "duplicate transition: {t}"),
            Self::DuplicateArc { from, to } => write!(f, "duplicate arc: {from} -> {to}"),
            Self::NotEnabled(t) => write!(f, "transition not enabled: {t}"),
            Self::InsufficientTokens { place, required, available } => {
                write!(f, "insufficient tokens in {place}: need {required}, have {available}")
            }
        }
    }
}

impl std::error::Error for PetriError {}

// ── Arc types ───────────────────────────────────────────────────

/// Type of arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcKind {
    /// Normal arc with a weight.
    Normal,
    /// Inhibitor arc: transition fires only if place has fewer tokens than the weight.
    Inhibitor,
}

/// An arc connecting a place and a transition.
#[derive(Debug, Clone)]
pub struct Arc {
    pub place: String,
    pub transition: String,
    pub weight: u32,
    pub kind: ArcKind,
    /// Direction: true = place -> transition (input), false = transition -> place (output).
    pub is_input: bool,
}

// ── Place ───────────────────────────────────────────────────────

/// A place in the Petri net.
#[derive(Debug, Clone)]
pub struct Place {
    pub name: String,
    pub tokens: u32,
    pub capacity: Option<u32>,
}

impl Place {
    /// Create a place with initial tokens.
    pub fn new(name: impl Into<String>, tokens: u32) -> Self {
        Self { name: name.into(), tokens, capacity: None }
    }

    /// Set a capacity limit.
    pub fn with_capacity(mut self, cap: u32) -> Self {
        self.capacity = Some(cap);
        self
    }
}

// ── Transition ──────────────────────────────────────────────────

/// A transition in the Petri net.
#[derive(Debug, Clone)]
pub struct Transition {
    pub name: String,
    pub fire_count: u64,
}

impl Transition {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), fire_count: 0 }
    }
}

// ── Petri Net ───────────────────────────────────────────────────

/// A Petri net.
#[derive(Debug, Clone)]
pub struct PetriNet {
    places: HashMap<String, Place>,
    transitions: HashMap<String, Transition>,
    arcs: Vec<Arc>,
    firing_history: Vec<String>,
}

impl PetriNet {
    /// Create an empty Petri net.
    pub fn new() -> Self {
        Self {
            places: HashMap::new(),
            transitions: HashMap::new(),
            arcs: Vec::new(),
            firing_history: Vec::new(),
        }
    }

    /// Add a place.
    pub fn add_place(&mut self, place: Place) -> Result<(), PetriError> {
        if self.places.contains_key(&place.name) {
            return Err(PetriError::DuplicatePlace(place.name));
        }
        self.places.insert(place.name.clone(), place);
        Ok(())
    }

    /// Add a transition.
    pub fn add_transition(&mut self, transition: Transition) -> Result<(), PetriError> {
        if self.transitions.contains_key(&transition.name) {
            return Err(PetriError::DuplicateTransition(transition.name));
        }
        self.transitions.insert(transition.name.clone(), transition);
        Ok(())
    }

    /// Add an input arc (place -> transition).
    pub fn add_input_arc(&mut self, place: &str, transition: &str, weight: u32) -> Result<(), PetriError> {
        if !self.places.contains_key(place) {
            return Err(PetriError::PlaceNotFound(place.to_string()));
        }
        if !self.transitions.contains_key(transition) {
            return Err(PetriError::TransitionNotFound(transition.to_string()));
        }
        self.arcs.push(Arc {
            place: place.to_string(),
            transition: transition.to_string(),
            weight,
            kind: ArcKind::Normal,
            is_input: true,
        });
        Ok(())
    }

    /// Add an output arc (transition -> place).
    pub fn add_output_arc(&mut self, transition: &str, place: &str, weight: u32) -> Result<(), PetriError> {
        if !self.places.contains_key(place) {
            return Err(PetriError::PlaceNotFound(place.to_string()));
        }
        if !self.transitions.contains_key(transition) {
            return Err(PetriError::TransitionNotFound(transition.to_string()));
        }
        self.arcs.push(Arc {
            place: place.to_string(),
            transition: transition.to_string(),
            weight,
            kind: ArcKind::Normal,
            is_input: false,
        });
        Ok(())
    }

    /// Add an inhibitor arc (place -> transition). Transition fires only if
    /// place has fewer tokens than weight.
    pub fn add_inhibitor_arc(&mut self, place: &str, transition: &str, weight: u32) -> Result<(), PetriError> {
        if !self.places.contains_key(place) {
            return Err(PetriError::PlaceNotFound(place.to_string()));
        }
        if !self.transitions.contains_key(transition) {
            return Err(PetriError::TransitionNotFound(transition.to_string()));
        }
        self.arcs.push(Arc {
            place: place.to_string(),
            transition: transition.to_string(),
            weight,
            kind: ArcKind::Inhibitor,
            is_input: true,
        });
        Ok(())
    }

    /// Get a place.
    pub fn place(&self, name: &str) -> Option<&Place> {
        self.places.get(name)
    }

    /// Get tokens in a place.
    pub fn tokens(&self, place: &str) -> Option<u32> {
        self.places.get(place).map(|p| p.tokens)
    }

    /// Set tokens in a place.
    pub fn set_tokens(&mut self, place: &str, tokens: u32) -> Result<(), PetriError> {
        let p = self.places.get_mut(place)
            .ok_or_else(|| PetriError::PlaceNotFound(place.to_string()))?;
        p.tokens = tokens;
        Ok(())
    }

    /// Number of places.
    pub fn place_count(&self) -> usize { self.places.len() }

    /// Number of transitions.
    pub fn transition_count(&self) -> usize { self.transitions.len() }

    /// Number of arcs.
    pub fn arc_count(&self) -> usize { self.arcs.len() }

    /// Get the current marking (token assignment) as a sorted vector.
    pub fn marking(&self) -> Vec<(String, u32)> {
        let mut m: Vec<(String, u32)> = self.places.iter()
            .map(|(n, p)| (n.clone(), p.tokens))
            .collect();
        m.sort_by(|a, b| a.0.cmp(&b.0));
        m
    }

    /// Get the marking as a hashable key (sorted by place name).
    fn marking_key(&self) -> Vec<u32> {
        self.marking().iter().map(|(_, t)| *t).collect()
    }

    /// Check if a transition is enabled (can fire).
    pub fn is_enabled(&self, transition: &str) -> bool {
        if !self.transitions.contains_key(transition) {
            return false;
        }
        for arc in &self.arcs {
            if arc.transition != transition || !arc.is_input {
                continue;
            }
            let tokens = self.places.get(&arc.place).map_or(0, |p| p.tokens);
            match arc.kind {
                ArcKind::Normal => {
                    if tokens < arc.weight {
                        return false;
                    }
                }
                ArcKind::Inhibitor => {
                    if tokens >= arc.weight {
                        return false;
                    }
                }
            }
        }
        // Check output place capacity constraints.
        for arc in &self.arcs {
            if arc.transition != transition || arc.is_input {
                continue;
            }
            if let Some(place) = self.places.get(&arc.place) {
                if let Some(cap) = place.capacity {
                    if place.tokens + arc.weight > cap {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// All enabled transitions.
    pub fn enabled_transitions(&self) -> Vec<String> {
        let mut result: Vec<String> = self.transitions.keys()
            .filter(|t| self.is_enabled(t))
            .cloned()
            .collect();
        result.sort();
        result
    }

    /// Fire a transition.
    pub fn fire(&mut self, transition: &str) -> Result<(), PetriError> {
        if !self.transitions.contains_key(transition) {
            return Err(PetriError::TransitionNotFound(transition.to_string()));
        }
        if !self.is_enabled(transition) {
            return Err(PetriError::NotEnabled(transition.to_string()));
        }

        // Consume tokens from input places (normal arcs only).
        for arc in &self.arcs {
            if arc.transition == transition && arc.is_input && arc.kind == ArcKind::Normal {
                let place = self.places.get_mut(&arc.place).unwrap();
                place.tokens -= arc.weight;
            }
        }

        // Produce tokens to output places.
        for arc in &self.arcs {
            if arc.transition == transition && !arc.is_input {
                let place = self.places.get_mut(&arc.place).unwrap();
                place.tokens += arc.weight;
            }
        }

        self.transitions.get_mut(transition).unwrap().fire_count += 1;
        self.firing_history.push(transition.to_string());
        Ok(())
    }

    /// Fire count for a transition.
    pub fn fire_count(&self, transition: &str) -> Option<u64> {
        self.transitions.get(transition).map(|t| t.fire_count)
    }

    /// Firing history.
    pub fn firing_history(&self) -> &[String] {
        &self.firing_history
    }

    /// Detect deadlock: no transitions are enabled.
    pub fn is_deadlocked(&self) -> bool {
        self.enabled_transitions().is_empty()
    }

    /// Check if a transition is live (can always eventually fire from any reachable marking).
    /// Approximated: checks if the transition fires in a bounded BFS from current marking.
    pub fn is_live_approx(&self, transition: &str, max_markings: usize) -> bool {
        let reachable = self.reachability_graph(max_markings);
        // Check that from every reachable marking, we can reach a marking where
        // the transition is enabled.
        for (marking, enabled) in &reachable {
            if enabled.contains(&transition.to_string()) {
                continue;
            }
            // Check if any reachable marking from here enables the transition.
            let mut found = false;
            for (other_marking, other_enabled) in &reachable {
                if other_marking != marking && other_enabled.contains(&transition.to_string()) {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        true
    }

    /// Compute the reachability graph (bounded BFS).
    /// Returns a list of (marking, enabled transitions) pairs.
    pub fn reachability_graph(&self, max_markings: usize) -> Vec<(Vec<u32>, Vec<String>)> {
        let mut visited: HashSet<Vec<u32>> = HashSet::new();
        let mut queue: VecDeque<PetriNet> = VecDeque::new();
        let mut result = Vec::new();

        let initial_key = self.marking_key();
        visited.insert(initial_key);
        queue.push_back(self.clone());

        while let Some(net) = queue.pop_front() {
            if result.len() >= max_markings {
                break;
            }
            let key = net.marking_key();
            let enabled = net.enabled_transitions();
            result.push((key, enabled.clone()));

            for t in &enabled {
                let mut next = net.clone();
                if next.fire(t).is_ok() {
                    let next_key = next.marking_key();
                    if !visited.contains(&next_key) {
                        visited.insert(next_key);
                        queue.push_back(next);
                    }
                }
            }
        }
        result
    }

    /// Total tokens across all places.
    pub fn total_tokens(&self) -> u32 {
        self.places.values().map(|p| p.tokens).sum()
    }

    /// Export to DOT (Graphviz) format.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph PetriNet {\n");
        out.push_str("  rankdir=LR;\n");
        out.push_str("  node [shape=circle];\n");

        // Places.
        let mut place_names: Vec<&String> = self.places.keys().collect();
        place_names.sort();
        for name in &place_names {
            let p = &self.places[*name];
            let label = format!("{}\\n[{}]", name, p.tokens);
            out.push_str(&format!("  \"{}\" [label=\"{}\"];\n", name, label));
        }

        // Transitions.
        let mut trans_names: Vec<&String> = self.transitions.keys().collect();
        trans_names.sort();
        for name in &trans_names {
            out.push_str(&format!("  \"{}\" [shape=box, style=filled, fillcolor=lightgray];\n", name));
        }

        // Arcs.
        for arc in &self.arcs {
            let style = match arc.kind {
                ArcKind::Normal => "",
                ArcKind::Inhibitor => ", style=dashed, arrowhead=odot",
            };
            let label = if arc.weight > 1 {
                format!(", label=\"{}\"", arc.weight)
            } else {
                String::new()
            };
            if arc.is_input {
                out.push_str(&format!("  \"{}\" -> \"{}\" [{}{}];\n",
                    arc.place, arc.transition, label.trim_start_matches(", "), style));
            } else {
                out.push_str(&format!("  \"{}\" -> \"{}\" [{}];\n",
                    arc.transition, arc.place, label.trim_start_matches(", ")));
            }
        }

        out.push_str("}\n");
        out
    }
}

impl Default for PetriNet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_net() -> PetriNet {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 1)).unwrap();
        net.add_place(Place::new("p2", 0)).unwrap();
        net.add_transition(Transition::new("t1")).unwrap();
        net.add_input_arc("p1", "t1", 1).unwrap();
        net.add_output_arc("t1", "p2", 1).unwrap();
        net
    }

    #[test]
    fn test_creation() {
        let net = PetriNet::new();
        assert_eq!(net.place_count(), 0);
        assert_eq!(net.transition_count(), 0);
    }

    #[test]
    fn test_add_place() {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 3)).unwrap();
        assert_eq!(net.tokens("p1"), Some(3));
    }

    #[test]
    fn test_duplicate_place() {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 0)).unwrap();
        assert!(net.add_place(Place::new("p1", 0)).is_err());
    }

    #[test]
    fn test_add_transition() {
        let mut net = PetriNet::new();
        net.add_transition(Transition::new("t1")).unwrap();
        assert_eq!(net.transition_count(), 1);
    }

    #[test]
    fn test_duplicate_transition() {
        let mut net = PetriNet::new();
        net.add_transition(Transition::new("t1")).unwrap();
        assert!(net.add_transition(Transition::new("t1")).is_err());
    }

    #[test]
    fn test_arc_missing_place() {
        let mut net = PetriNet::new();
        net.add_transition(Transition::new("t1")).unwrap();
        assert!(net.add_input_arc("missing", "t1", 1).is_err());
    }

    #[test]
    fn test_arc_missing_transition() {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 0)).unwrap();
        assert!(net.add_input_arc("p1", "missing", 1).is_err());
    }

    #[test]
    fn test_is_enabled() {
        let net = simple_net();
        assert!(net.is_enabled("t1"));
    }

    #[test]
    fn test_not_enabled() {
        let mut net = simple_net();
        net.set_tokens("p1", 0).unwrap();
        assert!(!net.is_enabled("t1"));
    }

    #[test]
    fn test_fire() {
        let mut net = simple_net();
        net.fire("t1").unwrap();
        assert_eq!(net.tokens("p1"), Some(0));
        assert_eq!(net.tokens("p2"), Some(1));
    }

    #[test]
    fn test_fire_not_enabled() {
        let mut net = simple_net();
        net.set_tokens("p1", 0).unwrap();
        assert!(net.fire("t1").is_err());
    }

    #[test]
    fn test_fire_count() {
        let mut net = simple_net();
        net.fire("t1").unwrap();
        assert_eq!(net.fire_count("t1"), Some(1));
    }

    #[test]
    fn test_firing_history() {
        let mut net = simple_net();
        net.fire("t1").unwrap();
        assert_eq!(net.firing_history(), &["t1"]);
    }

    #[test]
    fn test_deadlock() {
        let mut net = simple_net();
        net.fire("t1").unwrap();
        // After firing, p1 is empty, no transitions enabled.
        assert!(net.is_deadlocked());
    }

    #[test]
    fn test_not_deadlocked() {
        let net = simple_net();
        assert!(!net.is_deadlocked());
    }

    #[test]
    fn test_inhibitor_arc() {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 0)).unwrap();
        net.add_place(Place::new("p2", 0)).unwrap();
        net.add_transition(Transition::new("t1")).unwrap();
        net.add_inhibitor_arc("p1", "t1", 1).unwrap();
        net.add_output_arc("t1", "p2", 1).unwrap();

        // p1 has 0 tokens, inhibitor weight is 1: 0 < 1 so enabled.
        assert!(net.is_enabled("t1"));
        net.fire("t1").unwrap();
        assert_eq!(net.tokens("p2"), Some(1));

        // Now put a token in p1 — inhibitor should block.
        net.set_tokens("p1", 1).unwrap();
        assert!(!net.is_enabled("t1"));
    }

    #[test]
    fn test_capacity() {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 1)).unwrap();
        net.add_place(Place::new("p2", 2).with_capacity(2)).unwrap();
        net.add_transition(Transition::new("t1")).unwrap();
        net.add_input_arc("p1", "t1", 1).unwrap();
        net.add_output_arc("t1", "p2", 1).unwrap();

        // p2 is at capacity 2 — transition should not be enabled.
        assert!(!net.is_enabled("t1"));
    }

    #[test]
    fn test_marking() {
        let net = simple_net();
        let marking = net.marking();
        assert_eq!(marking.len(), 2);
        // Sorted by name.
        assert_eq!(marking[0], ("p1".to_string(), 1));
        assert_eq!(marking[1], ("p2".to_string(), 0));
    }

    #[test]
    fn test_total_tokens() {
        let net = simple_net();
        assert_eq!(net.total_tokens(), 1);
    }

    #[test]
    fn test_total_tokens_conserved() {
        let mut net = simple_net();
        let before = net.total_tokens();
        net.fire("t1").unwrap();
        assert_eq!(net.total_tokens(), before);
    }

    #[test]
    fn test_reachability_graph() {
        let net = simple_net();
        let graph = net.reachability_graph(10);
        assert!(graph.len() >= 2); // At least initial + one firing.
    }

    #[test]
    fn test_set_tokens_missing_place() {
        let mut net = PetriNet::new();
        assert!(net.set_tokens("missing", 5).is_err());
    }

    #[test]
    fn test_enabled_transitions_list() {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 2)).unwrap();
        net.add_transition(Transition::new("t1")).unwrap();
        net.add_transition(Transition::new("t2")).unwrap();
        net.add_input_arc("p1", "t1", 1).unwrap();
        net.add_input_arc("p1", "t2", 1).unwrap();
        let enabled = net.enabled_transitions();
        assert_eq!(enabled.len(), 2);
    }

    #[test]
    fn test_dot_export() {
        let net = simple_net();
        let dot = net.to_dot();
        assert!(dot.contains("digraph PetriNet"));
        assert!(dot.contains("p1"));
        assert!(dot.contains("t1"));
        assert!(dot.contains("p2"));
    }

    #[test]
    fn test_weighted_arcs() {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 3)).unwrap();
        net.add_place(Place::new("p2", 0)).unwrap();
        net.add_transition(Transition::new("t1")).unwrap();
        net.add_input_arc("p1", "t1", 2).unwrap();
        net.add_output_arc("t1", "p2", 3).unwrap();
        net.fire("t1").unwrap();
        assert_eq!(net.tokens("p1"), Some(1));
        assert_eq!(net.tokens("p2"), Some(3));
    }

    #[test]
    fn test_multiple_firings() {
        let mut net = PetriNet::new();
        net.add_place(Place::new("p1", 5)).unwrap();
        net.add_place(Place::new("p2", 0)).unwrap();
        net.add_transition(Transition::new("t1")).unwrap();
        net.add_input_arc("p1", "t1", 1).unwrap();
        net.add_output_arc("t1", "p2", 1).unwrap();
        for _ in 0..5 {
            net.fire("t1").unwrap();
        }
        assert_eq!(net.tokens("p1"), Some(0));
        assert_eq!(net.tokens("p2"), Some(5));
        assert_eq!(net.fire_count("t1"), Some(5));
    }
}
