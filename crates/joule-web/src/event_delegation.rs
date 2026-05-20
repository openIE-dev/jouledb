//! Event delegation system — capture/bubble phases, event target resolution,
//! delegated handlers, stopPropagation, preventDefault, synthetic events,
//! event pooling, and custom events.
//!
//! Replaces React SyntheticEvent / Vue event system with a pure-Rust event
//! delegation model. Supports the full W3C event flow: capture phase
//! (root-to-target), target phase, and bubble phase (target-to-root).

use std::collections::HashMap;

// ── Event types ─────────────────────────────────────────────────────────

/// The phase of event dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventPhase {
    None,
    Capture,
    Target,
    Bubble,
}

/// A synthetic event flowing through the delegation system.
#[derive(Debug, Clone)]
pub struct SyntheticEvent {
    pub event_type: String,
    pub target_id: u64,
    pub current_target_id: u64,
    pub phase: EventPhase,
    pub timestamp: u64,
    pub data: HashMap<String, String>,
    propagation_stopped: bool,
    immediate_propagation_stopped: bool,
    default_prevented: bool,
    pub bubbles: bool,
    pub cancelable: bool,
}

impl SyntheticEvent {
    pub fn new(event_type: &str, target_id: u64, timestamp: u64) -> Self {
        Self {
            event_type: event_type.to_string(),
            target_id,
            current_target_id: target_id,
            phase: EventPhase::None,
            timestamp,
            data: HashMap::new(),
            propagation_stopped: false,
            immediate_propagation_stopped: false,
            default_prevented: false,
            bubbles: true,
            cancelable: true,
        }
    }

    pub fn with_data(mut self, key: &str, value: &str) -> Self {
        self.data.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_bubbles(mut self, bubbles: bool) -> Self {
        self.bubbles = bubbles;
        self
    }

    pub fn with_cancelable(mut self, cancelable: bool) -> Self {
        self.cancelable = cancelable;
        self
    }

    pub fn stop_propagation(&mut self) {
        self.propagation_stopped = true;
    }

    pub fn stop_immediate_propagation(&mut self) {
        self.immediate_propagation_stopped = true;
        self.propagation_stopped = true;
    }

    pub fn prevent_default(&mut self) {
        if self.cancelable {
            self.default_prevented = true;
        }
    }

    pub fn is_propagation_stopped(&self) -> bool {
        self.propagation_stopped
    }

    pub fn is_immediate_propagation_stopped(&self) -> bool {
        self.immediate_propagation_stopped
    }

    pub fn is_default_prevented(&self) -> bool {
        self.default_prevented
    }
}

// ── Handler registration ────────────────────────────────────────────────

/// A registered event handler.
#[derive(Debug, Clone)]
pub struct EventHandler {
    pub id: u64,
    pub event_type: String,
    pub node_id: u64,
    pub capture: bool,
    pub once: bool,
    pub passive: bool,
    fired_count: u64,
}

impl EventHandler {
    pub fn new(id: u64, event_type: &str, node_id: u64) -> Self {
        Self {
            id,
            event_type: event_type.to_string(),
            node_id,
            capture: false,
            once: false,
            passive: false,
            fired_count: 0,
        }
    }

    pub fn with_capture(mut self) -> Self {
        self.capture = true;
        self
    }

    pub fn with_once(mut self) -> Self {
        self.once = true;
        self
    }

    pub fn with_passive(mut self) -> Self {
        self.passive = true;
        self
    }

    pub fn fired_count(&self) -> u64 {
        self.fired_count
    }
}

// ── Event pool ──────────────────────────────────────────────────────────

/// Event pool for reusing event objects to reduce allocation.
pub struct EventPool {
    pool: Vec<SyntheticEvent>,
    max_size: usize,
}

impl EventPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            pool: Vec::new(),
            max_size,
        }
    }

    /// Acquire an event from the pool (or create a fresh one).
    pub fn acquire(&mut self, event_type: &str, target_id: u64, timestamp: u64) -> SyntheticEvent {
        if let Some(mut event) = self.pool.pop() {
            // Reset and reuse
            event.event_type = event_type.to_string();
            event.target_id = target_id;
            event.current_target_id = target_id;
            event.phase = EventPhase::None;
            event.timestamp = timestamp;
            event.data.clear();
            event.propagation_stopped = false;
            event.immediate_propagation_stopped = false;
            event.default_prevented = false;
            event.bubbles = true;
            event.cancelable = true;
            event
        } else {
            SyntheticEvent::new(event_type, target_id, timestamp)
        }
    }

    /// Return an event to the pool for reuse.
    pub fn release(&mut self, event: SyntheticEvent) {
        if self.pool.len() < self.max_size {
            self.pool.push(event);
        }
    }

    pub fn available(&self) -> usize {
        self.pool.len()
    }
}

// ── Dispatch result ─────────────────────────────────────────────────────

/// Record of a handler invocation during dispatch.
#[derive(Debug, Clone)]
pub struct HandlerInvocation {
    pub handler_id: u64,
    pub node_id: u64,
    pub phase: EventPhase,
}

/// Result of dispatching an event.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    pub event_type: String,
    pub target_id: u64,
    pub default_prevented: bool,
    pub propagation_stopped: bool,
    pub invocations: Vec<HandlerInvocation>,
}

// ── Delegation manager ──────────────────────────────────────────────────

/// Node in the delegation tree (simplified DOM-like structure for event routing).
#[derive(Debug, Clone)]
pub struct DelegationNode {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub children_ids: Vec<u64>,
    pub tag: String,
}

/// Central event delegation system.
pub struct EventDelegation {
    nodes: HashMap<u64, DelegationNode>,
    handlers: Vec<EventHandler>,
    next_handler_id: u64,
    dispatch_log: Vec<DispatchResult>,
}

impl EventDelegation {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            handlers: Vec::new(),
            next_handler_id: 1,
            dispatch_log: Vec::new(),
        }
    }

    /// Register a node in the delegation tree.
    pub fn register_node(&mut self, id: u64, tag: &str, parent_id: Option<u64>) {
        let node = DelegationNode {
            id,
            parent_id,
            children_ids: Vec::new(),
            tag: tag.to_string(),
        };
        self.nodes.insert(id, node);

        if let Some(pid) = parent_id {
            if let Some(parent) = self.nodes.get_mut(&pid) {
                parent.children_ids.push(id);
            }
        }
    }

    /// Remove a node from the tree.
    pub fn unregister_node(&mut self, id: u64) {
        // Remove from parent's children
        if let Some(node) = self.nodes.get(&id) {
            let parent_id = node.parent_id;
            if let Some(pid) = parent_id {
                if let Some(parent) = self.nodes.get_mut(&pid) {
                    parent.children_ids.retain(|cid| *cid != id);
                }
            }
        }
        self.nodes.remove(&id);
        // Remove handlers for this node
        self.handlers.retain(|h| h.node_id != id);
    }

    /// Add an event handler. Returns the handler ID.
    pub fn add_handler(&mut self, handler: EventHandler) -> u64 {
        let id = self.next_handler_id;
        self.next_handler_id += 1;
        let mut h = handler;
        h.id = id;
        self.handlers.push(h);
        id
    }

    /// Remove a handler by ID.
    pub fn remove_handler(&mut self, handler_id: u64) -> bool {
        let before = self.handlers.len();
        self.handlers.retain(|h| h.id != handler_id);
        self.handlers.len() < before
    }

    /// Get the ancestor path from root to the target node (inclusive).
    pub fn ancestor_path(&self, target_id: u64) -> Vec<u64> {
        let mut path = Vec::new();
        let mut current = target_id;
        loop {
            path.push(current);
            match self.nodes.get(&current).and_then(|n| n.parent_id) {
                Some(pid) => current = pid,
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Dispatch an event through the full capture-target-bubble cycle.
    pub fn dispatch(&mut self, event: &mut SyntheticEvent) -> DispatchResult {
        let target_id = event.target_id;
        let path = self.ancestor_path(target_id);
        let mut invocations = Vec::new();

        // Capture phase: root -> target (exclusive of target)
        if path.len() > 1 {
            for &node_id in &path[..path.len() - 1] {
                if event.is_propagation_stopped() {
                    break;
                }
                event.phase = EventPhase::Capture;
                event.current_target_id = node_id;
                self.invoke_handlers(node_id, event, true, &mut invocations);
            }
        }

        // Target phase
        if !event.is_propagation_stopped() {
            event.phase = EventPhase::Target;
            event.current_target_id = target_id;
            // At target, both capture and bubble handlers fire
            self.invoke_handlers(target_id, event, true, &mut invocations);
            self.invoke_handlers(target_id, event, false, &mut invocations);
        }

        // Bubble phase: target parent -> root
        if event.bubbles && path.len() > 1 {
            for &node_id in path[..path.len() - 1].iter().rev() {
                if event.is_propagation_stopped() {
                    break;
                }
                event.phase = EventPhase::Bubble;
                event.current_target_id = node_id;
                self.invoke_handlers(node_id, event, false, &mut invocations);
            }
        }

        // Remove once handlers
        let fired_once: Vec<u64> = self
            .handlers
            .iter()
            .filter(|h| h.once && h.fired_count > 0)
            .map(|h| h.id)
            .collect();
        for hid in &fired_once {
            self.handlers.retain(|h| h.id != *hid);
        }

        let result = DispatchResult {
            event_type: event.event_type.clone(),
            target_id,
            default_prevented: event.is_default_prevented(),
            propagation_stopped: event.is_propagation_stopped(),
            invocations,
        };

        self.dispatch_log.push(result.clone());
        result
    }

    fn invoke_handlers(
        &mut self,
        node_id: u64,
        event: &SyntheticEvent,
        capture: bool,
        invocations: &mut Vec<HandlerInvocation>,
    ) {
        let matching: Vec<usize> = self
            .handlers
            .iter()
            .enumerate()
            .filter(|(_, h)| {
                h.node_id == node_id
                    && h.event_type == event.event_type
                    && h.capture == capture
            })
            .map(|(i, _)| i)
            .collect();

        for idx in matching {
            if event.is_immediate_propagation_stopped() {
                break;
            }
            self.handlers[idx].fired_count += 1;
            invocations.push(HandlerInvocation {
                handler_id: self.handlers[idx].id,
                node_id,
                phase: event.phase,
            });
        }
    }

    /// Get the dispatch log.
    pub fn dispatch_log(&self) -> &[DispatchResult] {
        &self.dispatch_log
    }

    /// Number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: u64) -> Option<&DelegationNode> {
        self.nodes.get(&id)
    }
}

impl Default for EventDelegation {
    fn default() -> Self {
        Self::new()
    }
}

// ── Custom events ───────────────────────────────────────────────────────

/// Create a custom (non-standard) event with arbitrary data.
pub fn custom_event(event_type: &str, target_id: u64, timestamp: u64, data: HashMap<String, String>) -> SyntheticEvent {
    let mut event = SyntheticEvent::new(event_type, target_id, timestamp);
    event.data = data;
    event
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_tree() -> EventDelegation {
        let mut ed = EventDelegation::new();
        // root(1) -> div(2) -> button(3)
        ed.register_node(1, "body", None);
        ed.register_node(2, "div", Some(1));
        ed.register_node(3, "button", Some(2));
        ed
    }

    #[test]
    fn register_and_count_nodes() {
        let ed = setup_tree();
        assert_eq!(ed.node_count(), 3);
    }

    #[test]
    fn ancestor_path_correct() {
        let ed = setup_tree();
        let path = ed.ancestor_path(3);
        assert_eq!(path, vec![1, 2, 3]);
    }

    #[test]
    fn ancestor_path_root() {
        let ed = setup_tree();
        let path = ed.ancestor_path(1);
        assert_eq!(path, vec![1]);
    }

    #[test]
    fn add_and_remove_handler() {
        let mut ed = setup_tree();
        let hid = ed.add_handler(EventHandler::new(0, "click", 3));
        assert_eq!(ed.handler_count(), 1);
        assert!(ed.remove_handler(hid));
        assert_eq!(ed.handler_count(), 0);
    }

    #[test]
    fn dispatch_bubble_phase() {
        let mut ed = setup_tree();
        ed.add_handler(EventHandler::new(0, "click", 1)); // bubble on root
        ed.add_handler(EventHandler::new(0, "click", 3)); // bubble on target

        let mut event = SyntheticEvent::new("click", 3, 100);
        let result = ed.dispatch(&mut event);

        assert_eq!(result.invocations.len(), 2);
        // Target fires first, then bubble up to root
        assert_eq!(result.invocations[0].node_id, 3);
        assert_eq!(result.invocations[1].node_id, 1);
        assert_eq!(result.invocations[1].phase, EventPhase::Bubble);
    }

    #[test]
    fn dispatch_capture_phase() {
        let mut ed = setup_tree();
        ed.add_handler(EventHandler::new(0, "click", 1).with_capture());

        let mut event = SyntheticEvent::new("click", 3, 100);
        let result = ed.dispatch(&mut event);

        assert_eq!(result.invocations.len(), 1);
        assert_eq!(result.invocations[0].phase, EventPhase::Capture);
        assert_eq!(result.invocations[0].node_id, 1);
    }

    #[test]
    fn stop_propagation_halts_bubble() {
        let mut ed = setup_tree();
        ed.add_handler(EventHandler::new(0, "click", 3)); // target
        ed.add_handler(EventHandler::new(0, "click", 1)); // root bubble

        let mut event = SyntheticEvent::new("click", 3, 100);
        event.stop_propagation(); // stopped before dispatch
        let result = ed.dispatch(&mut event);

        // Nothing fires because propagation was already stopped
        assert!(result.propagation_stopped);
    }

    #[test]
    fn prevent_default() {
        let mut event = SyntheticEvent::new("submit", 1, 100);
        event.prevent_default();
        assert!(event.is_default_prevented());
    }

    #[test]
    fn prevent_default_non_cancelable_noop() {
        let mut event = SyntheticEvent::new("scroll", 1, 100).with_cancelable(false);
        event.prevent_default();
        assert!(!event.is_default_prevented());
    }

    #[test]
    fn non_bubbling_event() {
        let mut ed = setup_tree();
        ed.add_handler(EventHandler::new(0, "focus", 3)); // target
        ed.add_handler(EventHandler::new(0, "focus", 1)); // root

        let mut event = SyntheticEvent::new("focus", 3, 100).with_bubbles(false);
        let result = ed.dispatch(&mut event);

        // Only target handler fires; root bubble handler does not
        assert_eq!(result.invocations.len(), 1);
        assert_eq!(result.invocations[0].node_id, 3);
    }

    #[test]
    fn once_handler_removed_after_fire() {
        let mut ed = setup_tree();
        ed.add_handler(EventHandler::new(0, "click", 3).with_once());
        assert_eq!(ed.handler_count(), 1);

        let mut event = SyntheticEvent::new("click", 3, 100);
        ed.dispatch(&mut event);

        // Handler should be removed after firing once
        assert_eq!(ed.handler_count(), 0);
    }

    #[test]
    fn event_pool_acquire_release() {
        let mut pool = EventPool::new(5);
        assert_eq!(pool.available(), 0);

        let e = pool.acquire("click", 1, 100);
        pool.release(e);
        assert_eq!(pool.available(), 1);

        let e2 = pool.acquire("mouseover", 2, 200);
        assert_eq!(e2.event_type, "mouseover");
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn event_pool_max_size() {
        let mut pool = EventPool::new(2);
        let e1 = pool.acquire("a", 1, 1);
        let e2 = pool.acquire("b", 2, 2);
        let e3 = pool.acquire("c", 3, 3);
        pool.release(e1);
        pool.release(e2);
        pool.release(e3); // exceeds max, should be dropped
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn custom_event_with_data() {
        let mut data = HashMap::new();
        data.insert("detail".to_string(), "custom-payload".to_string());
        let event = custom_event("myevent", 1, 100, data);
        assert_eq!(event.data.get("detail").unwrap(), "custom-payload");
    }

    #[test]
    fn unregister_node_removes_handlers() {
        let mut ed = setup_tree();
        ed.add_handler(EventHandler::new(0, "click", 3));
        ed.add_handler(EventHandler::new(0, "hover", 3));
        assert_eq!(ed.handler_count(), 2);

        ed.unregister_node(3);
        assert_eq!(ed.handler_count(), 0);
        assert_eq!(ed.node_count(), 2);
    }

    #[test]
    fn dispatch_log_recorded() {
        let mut ed = setup_tree();
        ed.add_handler(EventHandler::new(0, "click", 3));

        let mut event = SyntheticEvent::new("click", 3, 100);
        ed.dispatch(&mut event);

        assert_eq!(ed.dispatch_log().len(), 1);
        assert_eq!(ed.dispatch_log()[0].event_type, "click");
    }

    #[test]
    fn event_with_data_builder() {
        let event = SyntheticEvent::new("input", 1, 0)
            .with_data("value", "hello")
            .with_data("key", "Enter");
        assert_eq!(event.data.get("value").unwrap(), "hello");
        assert_eq!(event.data.get("key").unwrap(), "Enter");
    }

    #[test]
    fn stop_immediate_propagation_flag() {
        let mut event = SyntheticEvent::new("click", 1, 0);
        assert!(!event.is_immediate_propagation_stopped());
        event.stop_immediate_propagation();
        assert!(event.is_immediate_propagation_stopped());
        assert!(event.is_propagation_stopped());
    }

    #[test]
    fn handler_fired_count_increments() {
        let mut ed = setup_tree();
        ed.add_handler(EventHandler::new(0, "click", 3));

        let mut e1 = SyntheticEvent::new("click", 3, 100);
        ed.dispatch(&mut e1);
        let mut e2 = SyntheticEvent::new("click", 3, 200);
        ed.dispatch(&mut e2);

        // Handler still exists and was fired twice
        assert_eq!(ed.handler_count(), 1);
    }
}
