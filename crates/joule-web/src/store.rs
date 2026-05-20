//! Redux-like typed store with middleware, subscriptions, and time-travel debugging.
//!
//! Unlike the simpler `state::Store`, `TypedStore` features a full middleware
//! chain, action logging, and a history of states for time-travel debugging.

use std::cell::RefCell;
use std::rc::Rc;

// ── Action ──

/// An action dispatched to the store.
#[derive(Debug, Clone)]
pub struct Action {
    /// A string identifying the action type.
    pub action_type: String,
    /// Optional JSON payload.
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

// ── Middleware ──

/// A middleware function receives the current state, the action, and a `next`
/// callback. It may transform the action before passing it on, or skip it.
pub type MiddlewareFn<S> = Box<dyn Fn(&S, &Action, &dyn Fn(Action))>;

// ── TypedStore ──

/// Redux-like store with middleware chain, subscriptions, and time-travel.
pub struct TypedStore<S> {
    state: S,
    reducer: Box<dyn Fn(&S, &Action) -> S>,
    middleware: Vec<MiddlewareFn<S>>,
    listeners: Vec<Option<Box<dyn Fn(&S)>>>,
    next_listener_id: usize,
    /// Full history of states for time-travel debugging.
    history: Vec<S>,
    /// Log of all dispatched actions.
    action_log: Vec<Action>,
    /// Whether time-travel recording is enabled.
    recording: bool,
}

impl<S> TypedStore<S>
where
    S: Clone,
{
    /// Create a new store with initial state and reducer.
    pub fn new(initial_state: S, reducer: impl Fn(&S, &Action) -> S + 'static) -> Self {
        let history = vec![initial_state.clone()];
        Self {
            state: initial_state,
            reducer: Box::new(reducer),
            middleware: Vec::new(),
            listeners: Vec::new(),
            next_listener_id: 0,
            history,
            action_log: Vec::new(),
            recording: true,
        }
    }

    /// Add a middleware to the chain.
    pub fn add_middleware(&mut self, mw: impl Fn(&S, &Action, &dyn Fn(Action)) + 'static) {
        self.middleware.push(Box::new(mw));
    }

    /// Dispatch an action through the middleware chain then the reducer.
    pub fn dispatch(&mut self, action: Action) {
        let final_action = if self.middleware.is_empty() {
            action
        } else {
            let collected: Rc<RefCell<Option<Action>>> = Rc::new(RefCell::new(None));
            let coll = collected.clone();
            let base_next = move |a: Action| {
                coll.borrow_mut().replace(a);
            };

            for mw in &self.middleware {
                mw(&self.state, &action, &base_next);
            }

            let result = collected.borrow_mut().take();
            result.unwrap_or(action)
        };

        // Log action
        self.action_log.push(final_action.clone());

        // Run reducer
        self.state = (self.reducer)(&self.state, &final_action);

        // Record state in history
        if self.recording {
            self.history.push(self.state.clone());
        }

        // Notify listeners
        for listener in &self.listeners {
            if let Some(f) = listener {
                f(&self.state);
            }
        }
    }

    /// Get a reference to the current state.
    pub fn state(&self) -> &S {
        &self.state
    }

    /// Subscribe to state changes. Returns a listener ID for unsubscribing.
    pub fn subscribe(&mut self, listener: impl Fn(&S) + 'static) -> usize {
        let id = self.next_listener_id;
        self.next_listener_id += 1;
        while self.listeners.len() <= id {
            self.listeners.push(None);
        }
        self.listeners[id] = Some(Box::new(listener));
        id
    }

    /// Unsubscribe by listener ID.
    pub fn unsubscribe(&mut self, id: usize) {
        if id < self.listeners.len() {
            self.listeners[id] = None;
        }
    }

    // ── Time-travel debugging ──

    /// Get the full state history.
    pub fn history(&self) -> &[S] {
        &self.history
    }

    /// Jump to a specific index in the history, setting the current state.
    pub fn jump_to(&mut self, index: usize) -> bool {
        if index < self.history.len() {
            self.state = self.history[index].clone();
            true
        } else {
            false
        }
    }

    /// Number of recorded states.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Enable or disable time-travel recording.
    pub fn set_recording(&mut self, enabled: bool) {
        self.recording = enabled;
    }

    /// Get the action log.
    pub fn action_log(&self) -> &[Action] {
        &self.action_log
    }

    /// Clear the action log.
    pub fn clear_action_log(&mut self) {
        self.action_log.clear();
    }

    /// Clear history, keeping only the current state.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history.push(self.state.clone());
    }

    /// Replace the current state directly (bypass reducer).
    pub fn replace_state(&mut self, new_state: S) {
        self.state = new_state;
        if self.recording {
            self.history.push(self.state.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    fn counter_reducer(state: &i32, action: &Action) -> i32 {
        match action.action_type.as_str() {
            "INCREMENT" => state + 1,
            "DECREMENT" => state - 1,
            "ADD" => {
                let n: i32 = action
                    .payload
                    .as_deref()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                state + n
            }
            _ => *state,
        }
    }

    #[test]
    fn basic_dispatch() {
        let mut store = TypedStore::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.state(), 1);
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.state(), 2);
        store.dispatch(Action::new("DECREMENT"));
        assert_eq!(*store.state(), 1);
    }

    #[test]
    fn dispatch_with_payload() {
        let mut store = TypedStore::new(0, counter_reducer);
        store.dispatch(Action::with_payload("ADD", "10"));
        assert_eq!(*store.state(), 10);
        store.dispatch(Action::with_payload("ADD", "-3"));
        assert_eq!(*store.state(), 7);
    }

    #[test]
    fn subscribe_and_notify() {
        let count = Rc::new(Cell::new(0));
        let c = count.clone();
        let mut store = TypedStore::new(0, counter_reducer);
        store.subscribe(move |_s| {
            c.set(c.get() + 1);
        });
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(count.get(), 2);
    }

    #[test]
    fn unsubscribe() {
        let count = Rc::new(Cell::new(0));
        let c = count.clone();
        let mut store = TypedStore::new(0, counter_reducer);
        let id = store.subscribe(move |_s| {
            c.set(c.get() + 1);
        });
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(count.get(), 1);
        store.unsubscribe(id);
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn middleware_passthrough() {
        let mut store = TypedStore::new(0, counter_reducer);
        store.add_middleware(|_state, action, next| {
            next(action.clone());
        });
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.state(), 1);
    }

    #[test]
    fn middleware_transforms_action() {
        let mut store = TypedStore::new(0, counter_reducer);
        // Middleware that doubles increments by dispatching ADD(2) instead
        store.add_middleware(|_state, action, next| {
            if action.action_type == "INCREMENT" {
                next(Action::with_payload("ADD", "2"));
            } else {
                next(action.clone());
            }
        });
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.state(), 2);
    }

    #[test]
    fn time_travel_history() {
        let mut store = TypedStore::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT")); // state=1
        store.dispatch(Action::new("INCREMENT")); // state=2
        store.dispatch(Action::new("INCREMENT")); // state=3

        assert_eq!(store.history_len(), 4); // initial + 3 dispatches
        assert_eq!(*store.state(), 3);

        assert!(store.jump_to(1));
        assert_eq!(*store.state(), 1);

        assert!(store.jump_to(0));
        assert_eq!(*store.state(), 0);

        assert!(!store.jump_to(100));
    }

    #[test]
    fn action_log() {
        let mut store = TypedStore::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::with_payload("ADD", "5"));
        store.dispatch(Action::new("DECREMENT"));

        let log = store.action_log();
        assert_eq!(log.len(), 3);
        assert_eq!(log[0].action_type, "INCREMENT");
        assert_eq!(log[1].action_type, "ADD");
        assert_eq!(log[1].payload.as_deref(), Some("5"));
        assert_eq!(log[2].action_type, "DECREMENT");
    }

    #[test]
    fn clear_action_log_and_history() {
        let mut store = TypedStore::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(store.action_log().len(), 2);
        assert_eq!(store.history_len(), 3);

        store.clear_action_log();
        assert_eq!(store.action_log().len(), 0);

        store.clear_history();
        assert_eq!(store.history_len(), 1);
        assert_eq!(*store.state(), 2);
    }

    #[test]
    fn recording_toggle() {
        let mut store = TypedStore::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT")); // recorded
        assert_eq!(store.history_len(), 2);

        store.set_recording(false);
        store.dispatch(Action::new("INCREMENT")); // not recorded
        assert_eq!(store.history_len(), 2);
        assert_eq!(*store.state(), 2);

        store.set_recording(true);
        store.dispatch(Action::new("INCREMENT")); // recorded
        assert_eq!(store.history_len(), 3);
    }

    #[test]
    fn replace_state() {
        let mut store = TypedStore::new(0, counter_reducer);
        store.replace_state(42);
        assert_eq!(*store.state(), 42);
        assert_eq!(store.history_len(), 2);
    }

    #[test]
    fn unknown_action_no_change() {
        let mut store = TypedStore::new(5, counter_reducer);
        store.dispatch(Action::new("UNKNOWN"));
        assert_eq!(*store.state(), 5);
    }
}
