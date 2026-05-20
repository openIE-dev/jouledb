//! Redux-style state store with actions, reducers, middleware, and selectors.
//!
//! Provides a complete Redux implementation: dispatch, getState, subscribe,
//! middleware pipeline (logger, thunk, undo/redo), combined reducers, and
//! memoized selectors.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

// ── Action ───────────────────────────────────────────────────

/// An action dispatched to the store.
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    pub action_type: String,
    pub payload: Option<serde_json::Value>,
}

impl Action {
    pub fn new(action_type: impl Into<String>) -> Self {
        Self {
            action_type: action_type.into(),
            payload: None,
        }
    }

    pub fn with_payload(action_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            action_type: action_type.into(),
            payload: Some(payload),
        }
    }
}

// ── Reducer ──────────────────────────────────────────────────

/// A reducer function: (state, action) -> new_state.
pub type ReducerFn<S> = Box<dyn Fn(&S, &Action) -> S>;

// ── Middleware ───────────────────────────────────────────────

/// Middleware receives the store's state, the action, and a `next` callback.
/// It can modify the action, dispatch additional actions, or skip forwarding.
pub type MiddlewareFn<S> = Box<dyn Fn(&S, &Action, &dyn Fn(Action))>;

// ── Store ────────────────────────────────────────────────────

/// Redux-style store with middleware, subscriptions, and undo/redo.
pub struct Store<S> {
    state: S,
    reducer: ReducerFn<S>,
    middleware: Vec<MiddlewareFn<S>>,
    listeners: Vec<Option<Box<dyn Fn(&S)>>>,
    next_listener_id: usize,
    /// Undo stack (previous states).
    undo_stack: Vec<S>,
    /// Redo stack (states undone).
    redo_stack: Vec<S>,
    /// Maximum undo history size.
    max_history: usize,
    /// Action log for debugging.
    action_log: Vec<Action>,
    /// Whether to record actions in the log.
    logging_enabled: bool,
}

impl<S: Clone> Store<S> {
    /// Create a new store with an initial state and reducer.
    pub fn new(initial: S, reducer: impl Fn(&S, &Action) -> S + 'static) -> Self {
        Self {
            state: initial,
            reducer: Box::new(reducer),
            middleware: Vec::new(),
            listeners: Vec::new(),
            next_listener_id: 0,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_history: 100,
            action_log: Vec::new(),
            logging_enabled: false,
        }
    }

    /// Get a reference to the current state.
    pub fn get_state(&self) -> &S {
        &self.state
    }

    /// Enable action logging.
    pub fn enable_logging(&mut self) {
        self.logging_enabled = true;
    }

    /// Set maximum undo history size.
    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
    }

    /// Add a middleware to the pipeline.
    pub fn add_middleware(&mut self, mw: impl Fn(&S, &Action, &dyn Fn(Action)) + 'static) {
        self.middleware.push(Box::new(mw));
    }

    /// Subscribe to state changes. Returns a listener ID.
    pub fn subscribe(&mut self, listener: impl Fn(&S) + 'static) -> usize {
        let id = self.next_listener_id;
        while self.listeners.len() <= id {
            self.listeners.push(None);
        }
        self.listeners[id] = Some(Box::new(listener));
        self.next_listener_id += 1;
        id
    }

    /// Remove a listener.
    pub fn unsubscribe(&mut self, id: usize) {
        if id < self.listeners.len() {
            self.listeners[id] = None;
        }
    }

    /// Dispatch an action through the middleware pipeline and reducer.
    pub fn dispatch(&mut self, action: Action) {
        // Run through middleware chain
        let final_action = self.run_middleware(action);

        if let Some(action) = final_action {
            if self.logging_enabled {
                self.action_log.push(action.clone());
            }

            // Save current state for undo
            self.undo_stack.push(self.state.clone());
            if self.undo_stack.len() > self.max_history {
                self.undo_stack.remove(0);
            }
            // Clear redo stack on new action
            self.redo_stack.clear();

            // Reduce
            self.state = (self.reducer)(&self.state, &action);

            // Notify listeners
            self.notify_listeners();
        }
    }

    fn run_middleware(&self, action: Action) -> Option<Action> {
        if self.middleware.is_empty() {
            return Some(action);
        }

        let result = Rc::new(RefCell::new(None::<Action>));
        let result_inner = result.clone();

        // Build the chain from inside out
        let final_next: Box<dyn Fn(Action)> = Box::new(move |a| {
            *result_inner.borrow_mut() = Some(a);
        });

        // We apply middleware in reverse order so the first middleware
        // added is the outermost wrapper
        let mut chain: Box<dyn Fn(Action)> = final_next;
        for mw in self.middleware.iter().rev() {
            let state_ref = &self.state;
            let prev_chain = chain;
            // We need to capture the middleware and state
            let applied: Box<dyn Fn(Action)> = Box::new(move |a: Action| {
                mw(state_ref, &a, &*prev_chain);
            });
            chain = applied;
        }

        chain(action);
        let out = result.borrow().clone();
        out
    }

    fn notify_listeners(&mut self) {
        // Collect state clone to avoid borrow issues
        let state = self.state.clone();
        for slot in &self.listeners {
            if let Some(f) = slot.as_ref() {
                f(&state);
            }
        }
    }

    /// Undo the last action. Returns true if successful.
    pub fn undo(&mut self) -> bool {
        if let Some(prev_state) = self.undo_stack.pop() {
            self.redo_stack.push(self.state.clone());
            self.state = prev_state;
            self.notify_listeners();
            true
        } else {
            false
        }
    }

    /// Redo the last undone action. Returns true if successful.
    pub fn redo(&mut self) -> bool {
        if let Some(next_state) = self.redo_stack.pop() {
            self.undo_stack.push(self.state.clone());
            self.state = next_state;
            self.notify_listeners();
            true
        } else {
            false
        }
    }

    /// Check if undo is available.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Get the action log.
    pub fn action_log(&self) -> &[Action] {
        &self.action_log
    }

    /// Replace the reducer (useful for hot-reloading).
    pub fn replace_reducer(&mut self, reducer: impl Fn(&S, &Action) -> S + 'static) {
        self.reducer = Box::new(reducer);
    }
}

// ── Combined Reducer ─────────────────────────────────────────

/// Combines multiple named reducers that each handle a field of a HashMap state.
pub fn combine_reducers(
    reducers: Vec<(
        String,
        Box<dyn Fn(&serde_json::Value, &Action) -> serde_json::Value>,
    )>,
) -> impl Fn(&HashMap<String, serde_json::Value>, &Action) -> HashMap<String, serde_json::Value> {
    move |state: &HashMap<String, serde_json::Value>, action: &Action| {
        let mut new_state = state.clone();
        for (key, reducer) in &reducers {
            let current = state
                .get(key)
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let next = reducer(&current, action);
            new_state.insert(key.clone(), next);
        }
        new_state
    }
}

// ── Selector ─────────────────────────────────────────────────

/// A memoized selector that caches its result until the input changes.
pub struct Selector<S, R> {
    select: Box<dyn Fn(&S) -> R>,
    cached_input_hash: Option<u64>,
    cached_result: Option<R>,
}

impl<S, R: Clone> Selector<S, R> {
    pub fn new(select: impl Fn(&S) -> R + 'static) -> Self {
        Self {
            select: Box::new(select),
            cached_input_hash: None,
            cached_result: None,
        }
    }

    /// Select from state, using cache if the state hash matches.
    pub fn select(&mut self, state: &S) -> R
    where
        S: std::hash::Hash,
    {
        use std::hash::{DefaultHasher, Hasher};
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        let hash = hasher.finish();

        if self.cached_input_hash == Some(hash) {
            if let Some(cached) = &self.cached_result {
                return cached.clone();
            }
        }

        let result = (self.select)(state);
        self.cached_input_hash = Some(hash);
        self.cached_result = Some(result.clone());
        result
    }

    /// Force recomputation.
    pub fn invalidate(&mut self) {
        self.cached_input_hash = None;
        self.cached_result = None;
    }
}

// ── Thunk Dispatcher ─────────────────────────────────────────

/// A thunk is a function that receives dispatch and getState, allowing
/// asynchronous or conditional dispatch.
pub struct ThunkDispatcher<S: Clone> {
    actions: Vec<ThunkAction<S>>,
}

enum ThunkAction<S: Clone> {
    Plain(Action),
    Thunk(Box<dyn FnOnce(&mut Store<S>)>),
}

impl<S: Clone> ThunkDispatcher<S> {
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
        }
    }

    /// Queue a plain action.
    pub fn dispatch_action(&mut self, action: Action) {
        self.actions.push(ThunkAction::Plain(action));
    }

    /// Queue a thunk (a function that can dispatch multiple actions).
    pub fn dispatch_thunk(&mut self, thunk: impl FnOnce(&mut Store<S>) + 'static) {
        self.actions.push(ThunkAction::Thunk(Box::new(thunk)));
    }

    /// Execute all queued actions against the store.
    pub fn execute(self, store: &mut Store<S>) {
        for action in self.actions {
            match action {
                ThunkAction::Plain(a) => store.dispatch(a),
                ThunkAction::Thunk(f) => f(store),
            }
        }
    }
}

impl<S: Clone> Default for ThunkDispatcher<S> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Logger Middleware ────────────────────────────────────────

/// Creates a logging middleware that records dispatched action types.
pub fn logger_middleware<S>(
    log: Rc<RefCell<Vec<String>>>,
) -> impl Fn(&S, &Action, &dyn Fn(Action)) {
    move |_state: &S, action: &Action, next: &dyn Fn(Action)| {
        log.borrow_mut().push(action.action_type.clone());
        next(action.clone());
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn counter_reducer(state: &i32, action: &Action) -> i32 {
        match action.action_type.as_str() {
            "INCREMENT" => state + 1,
            "DECREMENT" => state - 1,
            "ADD" => {
                if let Some(serde_json::Value::Number(n)) = &action.payload {
                    state + n.as_i64().unwrap_or(0) as i32
                } else {
                    *state
                }
            }
            _ => *state,
        }
    }

    #[test]
    fn basic_dispatch() {
        let mut store = Store::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.get_state(), 1);
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.get_state(), 2);
        store.dispatch(Action::new("DECREMENT"));
        assert_eq!(*store.get_state(), 1);
    }

    #[test]
    fn dispatch_with_payload() {
        let mut store = Store::new(0, counter_reducer);
        store.dispatch(Action::with_payload(
            "ADD",
            serde_json::Value::Number(serde_json::Number::from(5)),
        ));
        assert_eq!(*store.get_state(), 5);
    }

    #[test]
    fn subscribe_receives_updates() {
        let values = Rc::new(RefCell::new(Vec::new()));
        let v = values.clone();
        let mut store = Store::new(0, counter_reducer);
        store.subscribe(move |s| {
            v.borrow_mut().push(*s);
        });
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*values.borrow(), vec![1, 2]);
    }

    #[test]
    fn unsubscribe_stops_updates() {
        let count = Rc::new(RefCell::new(0));
        let c = count.clone();
        let mut store = Store::new(0, counter_reducer);
        let id = store.subscribe(move |_| {
            *c.borrow_mut() += 1;
        });
        store.dispatch(Action::new("INCREMENT"));
        store.unsubscribe(id);
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*count.borrow(), 1);
    }

    #[test]
    fn undo_redo() {
        let mut store = Store::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.get_state(), 2);

        assert!(store.undo());
        assert_eq!(*store.get_state(), 1);

        assert!(store.undo());
        assert_eq!(*store.get_state(), 0);

        assert!(!store.undo()); // nothing to undo

        assert!(store.redo());
        assert_eq!(*store.get_state(), 1);

        assert!(store.redo());
        assert_eq!(*store.get_state(), 2);

        assert!(!store.redo()); // nothing to redo
    }

    #[test]
    fn redo_cleared_on_new_dispatch() {
        let mut store = Store::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::new("INCREMENT"));
        store.undo();
        assert!(store.can_redo());
        store.dispatch(Action::new("DECREMENT"));
        assert!(!store.can_redo());
    }

    #[test]
    fn max_history_limit() {
        let mut store = Store::new(0, counter_reducer);
        store.set_max_history(3);
        for _ in 0..5 {
            store.dispatch(Action::new("INCREMENT"));
        }
        assert_eq!(*store.get_state(), 5);
        // Only 3 undo steps available
        assert!(store.undo());
        assert!(store.undo());
        assert!(store.undo());
        assert!(!store.undo());
        assert_eq!(*store.get_state(), 2);
    }

    #[test]
    fn action_logging() {
        let mut store = Store::new(0, counter_reducer);
        store.enable_logging();
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::new("DECREMENT"));
        assert_eq!(store.action_log().len(), 2);
        assert_eq!(store.action_log()[0].action_type, "INCREMENT");
        assert_eq!(store.action_log()[1].action_type, "DECREMENT");
    }

    #[test]
    fn middleware_passes_through() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut store = Store::new(0, counter_reducer);
        store.add_middleware(logger_middleware(log.clone()));
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.get_state(), 1);
        assert_eq!(*log.borrow(), vec!["INCREMENT"]);
    }

    #[test]
    fn middleware_can_block() {
        let mut store = Store::new(0, counter_reducer);
        store.add_middleware(|_state: &i32, action: &Action, next: &dyn Fn(Action)| {
            // Block DECREMENT actions
            if action.action_type != "DECREMENT" {
                next(action.clone());
            }
        });
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::new("DECREMENT"));
        assert_eq!(*store.get_state(), 1); // decrement was blocked
    }

    #[test]
    fn combined_reducers() {
        let reducer = combine_reducers(vec![
            (
                "counter".to_string(),
                Box::new(|state: &serde_json::Value, action: &Action| {
                    let n = state.as_i64().unwrap_or(0);
                    match action.action_type.as_str() {
                        "INCREMENT" => serde_json::Value::Number((n + 1).into()),
                        _ => state.clone(),
                    }
                }),
            ),
            (
                "name".to_string(),
                Box::new(|state: &serde_json::Value, action: &Action| {
                    match action.action_type.as_str() {
                        "SET_NAME" => action
                            .payload
                            .clone()
                            .unwrap_or(serde_json::Value::Null),
                        _ => state.clone(),
                    }
                }),
            ),
        ]);

        let initial: HashMap<String, serde_json::Value> = HashMap::new();
        let mut store = Store::new(initial, reducer);

        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(
            store.get_state().get("counter"),
            Some(&serde_json::Value::Number(1.into()))
        );

        store.dispatch(Action::with_payload(
            "SET_NAME",
            serde_json::Value::String("Alice".into()),
        ));
        assert_eq!(
            store.get_state().get("name"),
            Some(&serde_json::Value::String("Alice".into()))
        );
    }

    #[test]
    fn selector_memoization() {
        let mut selector = Selector::new(|state: &i32| state * 2);
        let result1 = selector.select(&5);
        assert_eq!(result1, 10);
        // Same input should use cache
        let result2 = selector.select(&5);
        assert_eq!(result2, 10);
        // Different input
        let result3 = selector.select(&3);
        assert_eq!(result3, 6);
    }

    #[test]
    fn selector_invalidate() {
        let mut selector = Selector::new(|state: &i32| state * 2);
        let _ = selector.select(&5);
        selector.invalidate();
        // After invalidation, should recompute
        let result = selector.select(&5);
        assert_eq!(result, 10);
    }

    #[test]
    fn thunk_dispatcher() {
        let mut store = Store::new(0, counter_reducer);
        let mut thunk = ThunkDispatcher::new();
        thunk.dispatch_action(Action::new("INCREMENT"));
        thunk.dispatch_thunk(|s| {
            s.dispatch(Action::new("INCREMENT"));
            s.dispatch(Action::new("INCREMENT"));
        });
        thunk.execute(&mut store);
        assert_eq!(*store.get_state(), 3);
    }

    #[test]
    fn replace_reducer() {
        let mut store = Store::new(0, counter_reducer);
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.get_state(), 1);

        // Replace with a reducer that doubles on INCREMENT
        store.replace_reducer(|state: &i32, action: &Action| match action.action_type.as_str() {
            "INCREMENT" => state + 2,
            _ => *state,
        });
        store.dispatch(Action::new("INCREMENT"));
        assert_eq!(*store.get_state(), 3);
    }

    #[test]
    fn unknown_action_preserves_state() {
        let mut store = Store::new(42, counter_reducer);
        store.dispatch(Action::new("UNKNOWN"));
        assert_eq!(*store.get_state(), 42);
    }

    #[test]
    fn undo_notifies_listeners() {
        let values = Rc::new(RefCell::new(Vec::new()));
        let v = values.clone();
        let mut store = Store::new(0, counter_reducer);
        store.subscribe(move |s| {
            v.borrow_mut().push(*s);
        });
        store.dispatch(Action::new("INCREMENT"));
        store.dispatch(Action::new("INCREMENT"));
        store.undo();
        assert_eq!(*values.borrow(), vec![1, 2, 1]);
    }
}
