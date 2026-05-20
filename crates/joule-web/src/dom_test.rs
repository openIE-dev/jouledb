//! Virtual DOM testing utilities.
//!
//! Replaces `@testing-library/react` and `enzyme` with pure-Rust helpers:
//!
//! - [`TestRenderer`] — renders components to a virtual tree
//! - [`Screen`] — query interface with `by_text`, `by_role`, `by_test_id`, `by_tag`
//! - [`SimulatedEvent`] — fire synthetic events and assert state changes
//! - [`wait_for`] — poll until a condition holds

use crate::vdom::VNode;
use std::collections::HashMap;
use std::fmt;

// ── Screen ──────────────────────────────────────────────────────

/// A screen object that wraps a virtual tree and provides query methods.
pub struct Screen {
    root: VNode,
}

impl Screen {
    /// Create a new screen from a root VNode.
    pub fn new(root: VNode) -> Self {
        Self { root }
    }

    /// Replace the current tree.
    pub fn update(&mut self, root: VNode) {
        self.root = root;
    }

    /// Return a reference to the root node.
    pub fn root(&self) -> &VNode {
        &self.root
    }

    /// Find all nodes whose text content contains `needle`.
    pub fn by_text(&self, needle: &str) -> Vec<&VNode> {
        let mut results = Vec::new();
        Self::collect_by_text(&self.root, needle, &mut results);
        results
    }

    /// Find all elements with the given `role` attribute.
    pub fn by_role(&self, role: &str) -> Vec<&VNode> {
        let mut results = Vec::new();
        Self::collect_by_attr(&self.root, "role", role, &mut results);
        results
    }

    /// Find all elements with the given `data-testid` attribute.
    pub fn by_test_id(&self, test_id: &str) -> Vec<&VNode> {
        let mut results = Vec::new();
        Self::collect_by_attr(&self.root, "data-testid", test_id, &mut results);
        results
    }

    /// Find all elements with the given tag name.
    pub fn by_tag(&self, tag_name: &str) -> Vec<&VNode> {
        let mut results = Vec::new();
        Self::collect_by_tag(&self.root, tag_name, &mut results);
        results
    }

    /// Assert that at least one node matching the text query exists.
    pub fn assert_text_present(&self, needle: &str) {
        assert!(
            !self.by_text(needle).is_empty(),
            "Expected to find text \"{needle}\" in the tree, but it was not found.\n{}",
            self.debug()
        );
    }

    /// Assert that no node matching the text query exists.
    pub fn assert_text_absent(&self, needle: &str) {
        assert!(
            self.by_text(needle).is_empty(),
            "Expected text \"{needle}\" to be absent, but it was found.\n{}",
            self.debug()
        );
    }

    /// Assert presence of an element with the given test id.
    pub fn assert_test_id_present(&self, test_id: &str) {
        assert!(
            !self.by_test_id(test_id).is_empty(),
            "Expected data-testid=\"{test_id}\" to be present.\n{}",
            self.debug()
        );
    }

    /// Assert absence of an element with the given test id.
    pub fn assert_test_id_absent(&self, test_id: &str) {
        assert!(
            self.by_test_id(test_id).is_empty(),
            "Expected data-testid=\"{test_id}\" to be absent.\n{}",
            self.debug()
        );
    }

    /// Pretty-print the tree for debugging.
    pub fn debug(&self) -> String {
        let mut buf = String::new();
        Self::debug_node(&self.root, 0, &mut buf);
        buf
    }

    // ── Private helpers ─────────────────────────────────────────

    fn collect_by_text<'a>(node: &'a VNode, needle: &str, out: &mut Vec<&'a VNode>) {
        match node {
            VNode::Text(t) if t.contains(needle) => {
                out.push(node);
            }
            VNode::Element { children, .. } => {
                // Check if any direct child text contains needle
                let full_text = Self::extract_text(node);
                if full_text.contains(needle) {
                    out.push(node);
                }
                for child in children {
                    if !matches!(child, VNode::Text(_)) {
                        Self::collect_by_text(child, needle, out);
                    }
                }
            }
            VNode::Fragment(children) => {
                for child in children {
                    Self::collect_by_text(child, needle, out);
                }
            }
            _ => {}
        }
    }

    fn collect_by_attr<'a>(node: &'a VNode, attr_key: &str, attr_val: &str, out: &mut Vec<&'a VNode>) {
        match node {
            VNode::Element { attrs, children, .. } => {
                if attrs.get(attr_key).map(|v| v.as_str()) == Some(attr_val) {
                    out.push(node);
                }
                for child in children {
                    Self::collect_by_attr(child, attr_key, attr_val, out);
                }
            }
            VNode::Fragment(children) => {
                for child in children {
                    Self::collect_by_attr(child, attr_key, attr_val, out);
                }
            }
            _ => {}
        }
    }

    fn collect_by_tag<'a>(node: &'a VNode, tag_name: &str, out: &mut Vec<&'a VNode>) {
        match node {
            VNode::Element { tag, children, .. } => {
                if tag == tag_name {
                    out.push(node);
                }
                for child in children {
                    Self::collect_by_tag(child, tag_name, out);
                }
            }
            VNode::Fragment(children) => {
                for child in children {
                    Self::collect_by_tag(child, tag_name, out);
                }
            }
            _ => {}
        }
    }

    /// Extract all text content from a node recursively.
    fn extract_text(node: &VNode) -> String {
        match node {
            VNode::Text(t) => t.clone(),
            VNode::Element { children, .. } | VNode::Fragment(children) => {
                children.iter().map(Self::extract_text).collect::<Vec<_>>().join("")
            }
            VNode::Empty => String::new(),
        }
    }

    fn debug_node(node: &VNode, indent: usize, buf: &mut String) {
        let pad = "  ".repeat(indent);
        match node {
            VNode::Text(t) => {
                let _ = fmt::write(buf, format_args!("{pad}\"{t}\"\n"));
            }
            VNode::Empty => {
                let _ = fmt::write(buf, format_args!("{pad}<empty />\n"));
            }
            VNode::Element { tag, attrs, children, .. } => {
                let _ = fmt::write(buf, format_args!("{pad}<{tag}"));
                let mut sorted_attrs: Vec<_> = attrs.iter().collect();
                sorted_attrs.sort_by_key(|(k, _)| k.as_str());
                for (k, v) in &sorted_attrs {
                    let _ = fmt::write(buf, format_args!(" {k}=\"{v}\""));
                }
                if children.is_empty() {
                    let _ = fmt::write(buf, format_args!(" />\n"));
                } else {
                    let _ = fmt::write(buf, format_args!(">\n"));
                    for child in children {
                        Self::debug_node(child, indent + 1, buf);
                    }
                    let _ = fmt::write(buf, format_args!("{pad}</{tag}>\n"));
                }
            }
            VNode::Fragment(children) => {
                let _ = fmt::write(buf, format_args!("{pad}<Fragment>\n"));
                for child in children {
                    Self::debug_node(child, indent + 1, buf);
                }
                let _ = fmt::write(buf, format_args!("{pad}</Fragment>\n"));
            }
        }
    }
}

// ── TestRenderer ────────────────────────────────────────────────

/// A test renderer that holds component state and produces a Screen.
pub struct TestRenderer {
    state: HashMap<String, String>,
    render_fn: Option<Box<dyn Fn(&HashMap<String, String>) -> VNode>>,
}

impl TestRenderer {
    /// Create a test renderer with a render function.
    pub fn new(render_fn: impl Fn(&HashMap<String, String>) -> VNode + 'static) -> Self {
        Self {
            state: HashMap::new(),
            render_fn: Some(Box::new(render_fn)),
        }
    }

    /// Create a test renderer from a static VNode tree.
    pub fn from_tree(tree: VNode) -> Self {
        Self {
            state: HashMap::new(),
            render_fn: Some(Box::new(move |_| tree.clone())),
        }
    }

    /// Set a state value.
    pub fn set_state(&mut self, key: &str, value: &str) {
        self.state.insert(key.to_string(), value.to_string());
    }

    /// Get a state value.
    pub fn get_state(&self, key: &str) -> Option<&str> {
        self.state.get(key).map(|s| s.as_str())
    }

    /// Render the component into a Screen.
    pub fn render(&self) -> Screen {
        let tree = match &self.render_fn {
            Some(f) => f(&self.state),
            None => VNode::Empty,
        };
        Screen::new(tree)
    }
}

// ── Simulated Events ────────────────────────────────────────────

/// A simulated event type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimulatedEvent {
    Click,
    Input(String),
    Submit,
    Focus,
    Blur,
    KeyDown(String),
    KeyUp(String),
    Change(String),
}

/// Record of events fired during simulation.
#[derive(Debug, Clone, Default)]
pub struct EventLog {
    entries: Vec<(String, SimulatedEvent)>,
}

impl EventLog {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Record an event against a target identifier.
    pub fn record(&mut self, target: &str, event: SimulatedEvent) {
        self.entries.push((target.to_string(), event));
    }

    /// Get all recorded events.
    pub fn entries(&self) -> &[(String, SimulatedEvent)] {
        &self.entries
    }

    /// Count events of a specific type.
    pub fn count(&self, event_type: &SimulatedEvent) -> usize {
        self.entries.iter().filter(|(_, e)| e == event_type).count()
    }

    /// Count events targeting a specific element.
    pub fn count_for_target(&self, target: &str) -> usize {
        self.entries.iter().filter(|(t, _)| t == target).count()
    }

    /// Clear the log.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Simulate a click by modifying renderer state and logging.
pub fn fire_click(
    renderer: &mut TestRenderer,
    log: &mut EventLog,
    target: &str,
    state_updates: &[(&str, &str)],
) {
    log.record(target, SimulatedEvent::Click);
    for (k, v) in state_updates {
        renderer.set_state(k, v);
    }
}

/// Simulate text input by modifying renderer state and logging.
pub fn fire_input(
    renderer: &mut TestRenderer,
    log: &mut EventLog,
    target: &str,
    value: &str,
    state_key: &str,
) {
    log.record(target, SimulatedEvent::Input(value.to_string()));
    renderer.set_state(state_key, value);
}

/// Simulate form submission.
pub fn fire_submit(
    renderer: &mut TestRenderer,
    log: &mut EventLog,
    target: &str,
    state_updates: &[(&str, &str)],
) {
    log.record(target, SimulatedEvent::Submit);
    for (k, v) in state_updates {
        renderer.set_state(k, v);
    }
}

// ── Wait-for ────────────────────────────────────────────────────

/// Poll a condition up to `max_attempts` times. Returns true if the condition
/// is met within the limit. This is a simulated poll — no real sleeping.
pub fn wait_for(mut condition: impl FnMut() -> bool, max_attempts: usize) -> bool {
    for _ in 0..max_attempts {
        if condition() {
            return true;
        }
    }
    false
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vdom::VNode;

    fn sample_tree() -> VNode {
        VNode::element("div")
            .attr("data-testid", "root")
            .child(
                VNode::element("h1")
                    .attr("role", "heading")
                    .child(VNode::text("Hello World")),
            )
            .child(
                VNode::element("button")
                    .attr("data-testid", "submit-btn")
                    .attr("role", "button")
                    .child(VNode::text("Click Me")),
            )
            .child(
                VNode::element("input")
                    .attr("data-testid", "name-input")
                    .attr("role", "textbox"),
            )
    }

    #[test]
    fn screen_by_text_finds_nodes() {
        let screen = Screen::new(sample_tree());
        let results = screen.by_text("Hello World");
        assert!(!results.is_empty());
    }

    #[test]
    fn screen_by_text_partial_match() {
        let screen = Screen::new(sample_tree());
        let results = screen.by_text("Hello");
        assert!(!results.is_empty());
    }

    #[test]
    fn screen_by_text_no_match() {
        let screen = Screen::new(sample_tree());
        let results = screen.by_text("Nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn screen_by_role() {
        let screen = Screen::new(sample_tree());
        let headings = screen.by_role("heading");
        assert_eq!(headings.len(), 1);
        let buttons = screen.by_role("button");
        assert_eq!(buttons.len(), 1);
    }

    #[test]
    fn screen_by_test_id() {
        let screen = Screen::new(sample_tree());
        let root = screen.by_test_id("root");
        assert_eq!(root.len(), 1);
        let btn = screen.by_test_id("submit-btn");
        assert_eq!(btn.len(), 1);
    }

    #[test]
    fn screen_by_tag() {
        let screen = Screen::new(sample_tree());
        let inputs = screen.by_tag("input");
        assert_eq!(inputs.len(), 1);
        let divs = screen.by_tag("div");
        assert_eq!(divs.len(), 1);
        let h1s = screen.by_tag("h1");
        assert_eq!(h1s.len(), 1);
    }

    #[test]
    fn screen_assert_text_present() {
        let screen = Screen::new(sample_tree());
        screen.assert_text_present("Click Me");
    }

    #[test]
    fn screen_assert_text_absent() {
        let screen = Screen::new(sample_tree());
        screen.assert_text_absent("Nonexistent");
    }

    #[test]
    fn screen_debug_output() {
        let screen = Screen::new(sample_tree());
        let debug = screen.debug();
        assert!(debug.contains("<div"));
        assert!(debug.contains("<h1"));
        assert!(debug.contains("Hello World"));
        assert!(debug.contains("<button"));
    }

    #[test]
    fn test_renderer_state_updates_tree() {
        let mut renderer = TestRenderer::new(|state| {
            let text = state.get("name").map(|s| s.as_str()).unwrap_or("Anonymous");
            VNode::element("div")
                .child(VNode::element("span").child(VNode::text(text)))
        });

        let screen = renderer.render();
        screen.assert_text_present("Anonymous");

        renderer.set_state("name", "Alice");
        let screen = renderer.render();
        screen.assert_text_present("Alice");
        screen.assert_text_absent("Anonymous");
    }

    #[test]
    fn fire_click_updates_state_and_logs() {
        let mut renderer = TestRenderer::new(|state| {
            let count = state.get("count").map(|s| s.as_str()).unwrap_or("0");
            VNode::element("div")
                .child(VNode::element("span").child(VNode::text(count)))
        });

        let mut log = EventLog::new();
        fire_click(&mut renderer, &mut log, "button", &[("count", "1")]);

        assert_eq!(log.count(&SimulatedEvent::Click), 1);
        assert_eq!(log.count_for_target("button"), 1);

        let screen = renderer.render();
        screen.assert_text_present("1");
    }

    #[test]
    fn fire_input_updates_state() {
        let mut renderer = TestRenderer::new(|state| {
            let val = state.get("email").map(|s| s.as_str()).unwrap_or("");
            VNode::element("input").attr("value", val)
        });

        let mut log = EventLog::new();
        fire_input(&mut renderer, &mut log, "email-field", "test@example.com", "email");

        assert_eq!(renderer.get_state("email"), Some("test@example.com"));
        assert_eq!(log.entries().len(), 1);
    }

    #[test]
    fn fire_submit_logs_and_updates() {
        let mut renderer = TestRenderer::new(|state| {
            let submitted = state.get("submitted").map(|s| s.as_str()).unwrap_or("false");
            VNode::element("div")
                .child(VNode::text(submitted))
        });

        let mut log = EventLog::new();
        fire_submit(&mut renderer, &mut log, "form", &[("submitted", "true")]);

        let screen = renderer.render();
        screen.assert_text_present("true");
        assert_eq!(log.count(&SimulatedEvent::Submit), 1);
    }

    #[test]
    fn wait_for_succeeds() {
        let mut counter = 0u32;
        let result = wait_for(|| { counter += 1; counter >= 3 }, 10);
        assert!(result);
        assert_eq!(counter, 3);
    }

    #[test]
    fn wait_for_fails() {
        let result = wait_for(|| false, 5);
        assert!(!result);
    }

    #[test]
    fn event_log_clear() {
        let mut log = EventLog::new();
        log.record("a", SimulatedEvent::Click);
        log.record("b", SimulatedEvent::Focus);
        assert_eq!(log.entries().len(), 2);
        log.clear();
        assert_eq!(log.entries().len(), 0);
    }

    #[test]
    fn screen_from_fragment() {
        let tree = VNode::fragment(vec![
            VNode::element("p").child(VNode::text("First")),
            VNode::element("p").child(VNode::text("Second")),
        ]);
        let screen = Screen::new(tree);
        let ps = screen.by_tag("p");
        assert_eq!(ps.len(), 2);
        screen.assert_text_present("First");
        screen.assert_text_present("Second");
    }
}
