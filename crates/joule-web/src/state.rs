//! Typed state management with actions, reducers, middleware, and undo support.
//!
//! Replaces Redux/Zustand with a pure-Rust store featuring middleware chains,
//! subscriptions, selectors, and built-in undo/redo capability.

use chrono::{DateTime, Utc};
use std::fmt::Debug;

// ── Store ──

/// Typed state store with reducer, middleware, and subscriptions.
pub struct Store<S, A> {
    state: S,
    reducer: Box<dyn Fn(&S, &A) -> S>,
    listeners: Vec<Option<Box<dyn Fn(&S)>>>,
    middleware: Vec<Box<dyn Fn(&S, &A, &dyn Fn(&A))>>,
    next_listener_id: usize,
}

impl<S, A> Store<S, A>
where
    S: Clone,
    A: Clone,
{
    /// Create a store with an initial state and reducer function.
    pub fn new(initial_state: S, reducer: impl Fn(&S, &A) -> S + 'static) -> Self {
        Self {
            state: initial_state,
            reducer: Box::new(reducer),
            listeners: Vec::new(),
            middleware: Vec::new(),
            next_listener_id: 0,
        }
    }

    /// Dispatch an action through the middleware chain, then the reducer.
    pub fn dispatch(&mut self, action: A) {
        let final_action = if self.middleware.is_empty() {
            action
        } else {
            use std::cell::RefCell;
            let collected: RefCell<Option<A>> = RefCell::new(None);

            // Build the base "next" that stores the action for the reducer.
            let base_next = |a: &A| {
                let ptr = &collected as *const RefCell<Option<A>>;
                // SAFETY: single-threaded, `collected` lives on this stack frame.
                unsafe {
                    (*ptr).replace(Some(a.clone()));
                }
            };

            for mw in &self.middleware {
                mw(&self.state, &action, &base_next);
            }

            collected.into_inner().unwrap_or(action)
        };

        self.state = (self.reducer)(&self.state, &final_action);

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

    /// Subscribe to state changes. Returns a subscription ID.
    pub fn subscribe(&mut self, listener: impl Fn(&S) + 'static) -> usize {
        let id = self.next_listener_id;
        self.next_listener_id += 1;
        while self.listeners.len() <= id {
            self.listeners.push(None);
        }
        self.listeners[id] = Some(Box::new(listener));
        id
    }

    /// Remove a subscription by ID.
    pub fn unsubscribe(&mut self, id: usize) {
        if id < self.listeners.len() {
            self.listeners[id] = None;
        }
    }

    /// Add a middleware function.
    pub fn add_middleware(&mut self, mw: impl Fn(&S, &A, &dyn Fn(&A)) + 'static) {
        self.middleware.push(Box::new(mw));
    }
}

// ── Selector ──

/// Create a selector function that derives a value from state.
pub fn create_selector<S, T>(f: impl Fn(&S) -> T + 'static) -> Box<dyn Fn(&S) -> T> {
    Box::new(f)
}

// ── Logging Middleware ──

/// A logging middleware that records dispatched actions with timestamps.
pub struct LoggingMiddleware {
    entries: Vec<(DateTime<Utc>, String)>,
}

impl LoggingMiddleware {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record an action.
    pub fn log(&mut self, action_debug: &str) {
        self.entries.push((Utc::now(), action_debug.to_string()));
    }

    /// Get all recorded log entries.
    pub fn entries(&self) -> &[(DateTime<Utc>, String)] {
        &self.entries
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a logging middleware closure that records actions to a shared log.
pub fn create_logging_middleware<S, A: Debug>() -> (
    impl Fn(&S, &A, &dyn Fn(&A)),
    std::rc::Rc<std::cell::RefCell<LoggingMiddleware>>,
) {
    let log = std::rc::Rc::new(std::cell::RefCell::new(LoggingMiddleware::new()));
    let log_clone = log.clone();
    let mw = move |_state: &S, action: &A, next: &dyn Fn(&A)| {
        log_clone.borrow_mut().log(&format!("{:?}", action));
        next(action);
    };
    (mw, log)
}

// ── Undo Store ──

/// A store wrapper that adds undo/redo capability.
pub struct UndoStore<S, A>
where
    S: Clone,
    A: Clone,
{
    store: Store<S, A>,
    past: Vec<S>,
    future: Vec<S>,
}

impl<S, A> UndoStore<S, A>
where
    S: Clone,
    A: Clone,
{
    /// Create an undo-capable store.
    pub fn new(initial_state: S, reducer: impl Fn(&S, &A) -> S + 'static) -> Self {
        Self {
            store: Store::new(initial_state, reducer),
            past: Vec::new(),
            future: Vec::new(),
        }
    }

    /// Dispatch an action, saving the current state for undo.
    pub fn dispatch(&mut self, action: A) {
        self.past.push(self.store.state.clone());
        self.future.clear();
        self.store.dispatch(action);
    }

    /// Undo the last action. Returns `true` if successful.
    pub fn undo(&mut self) -> bool {
        if let Some(prev_state) = self.past.pop() {
            self.future.push(self.store.state.clone());
            self.store.state = prev_state;
            true
        } else {
            false
        }
    }

    /// Redo the last undone action. Returns `true` if successful.
    pub fn redo(&mut self) -> bool {
        if let Some(next_state) = self.future.pop() {
            self.past.push(self.store.state.clone());
            self.store.state = next_state;
            true
        } else {
            false
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    pub fn history_len(&self) -> usize {
        self.past.len()
    }

    /// Get a reference to the current state.
    pub fn state(&self) -> &S {
        self.store.state()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Clone, Debug, PartialEq)]
    struct Counter {
        count: i32,
    }

    #[derive(Clone, Debug)]
    enum CounterAction {
        Increment,
        Decrement,
        Add(i32),
    }

    fn counter_reducer(state: &Counter, action: &CounterAction) -> Counter {
        match action {
            CounterAction::Increment => Counter {
                count: state.count + 1,
            },
            CounterAction::Decrement => Counter {
                count: state.count - 1,
            },
            CounterAction::Add(n) => Counter {
                count: state.count + n,
            },
        }
    }

    #[test]
    fn test_initial_state() {
        let store = Store::new(Counter { count: 0 }, counter_reducer);
        assert_eq!(store.state().count, 0);
    }

    #[test]
    fn test_dispatch_updates_state() {
        let mut store = Store::new(Counter { count: 0 }, counter_reducer);
        store.dispatch(CounterAction::Increment);
        assert_eq!(store.state().count, 1);
        store.dispatch(CounterAction::Add(10));
        assert_eq!(store.state().count, 11);
    }

    #[test]
    fn test_subscriber_notified() {
        let mut store = Store::new(Counter { count: 0 }, counter_reducer);
        let seen = Rc::new(RefCell::new(Vec::new()));
        let seen_clone = seen.clone();
        store.subscribe(move |s: &Counter| {
            seen_clone.borrow_mut().push(s.count);
        });
        store.dispatch(CounterAction::Increment);
        store.dispatch(CounterAction::Increment);
        assert_eq!(*seen.borrow(), vec![1, 2]);
    }

    #[test]
    fn test_unsubscribe_stops_notifications() {
        let mut store = Store::new(Counter { count: 0 }, counter_reducer);
        let seen = Rc::new(RefCell::new(0i32));
        let seen_clone = seen.clone();
        let id = store.subscribe(move |_s: &Counter| {
            *seen_clone.borrow_mut() += 1;
        });
        store.dispatch(CounterAction::Increment);
        assert_eq!(*seen.borrow(), 1);
        store.unsubscribe(id);
        store.dispatch(CounterAction::Increment);
        assert_eq!(*seen.borrow(), 1);
    }

    #[test]
    fn test_selector_derives_value() {
        let store = Store::new(Counter { count: 42 }, counter_reducer);
        let is_positive = create_selector(|s: &Counter| s.count > 0);
        assert!(is_positive(store.state()));
    }

    #[test]
    fn test_multiple_subscribers() {
        let mut store = Store::new(Counter { count: 0 }, counter_reducer);
        let a = Rc::new(RefCell::new(0i32));
        let b = Rc::new(RefCell::new(0i32));
        let a2 = a.clone();
        let b2 = b.clone();
        store.subscribe(move |_| *a2.borrow_mut() += 1);
        store.subscribe(move |_| *b2.borrow_mut() += 1);
        store.dispatch(CounterAction::Increment);
        assert_eq!(*a.borrow(), 1);
        assert_eq!(*b.borrow(), 1);
    }

    #[test]
    fn test_middleware_intercepts_action() {
        let mut store = Store::new(Counter { count: 0 }, counter_reducer);
        store.add_middleware(
            |_state: &Counter, action: &CounterAction, next: &dyn Fn(&CounterAction)| match action
            {
                CounterAction::Add(n) => next(&CounterAction::Add(n * 2)),
                other => next(other),
            },
        );
        store.dispatch(CounterAction::Add(5));
        assert_eq!(store.state().count, 10);
    }

    #[test]
    fn test_undo_redo() {
        let mut store = UndoStore::new(Counter { count: 0 }, counter_reducer);
        store.dispatch(CounterAction::Increment);
        store.dispatch(CounterAction::Increment);
        assert_eq!(store.state().count, 2);

        assert!(store.undo());
        assert_eq!(store.state().count, 1);

        assert!(store.undo());
        assert_eq!(store.state().count, 0);

        assert!(!store.undo());

        assert!(store.redo());
        assert_eq!(store.state().count, 1);
    }

    #[test]
    fn test_undo_clears_redo_on_new_action() {
        let mut store = UndoStore::new(Counter { count: 0 }, counter_reducer);
        store.dispatch(CounterAction::Increment);
        store.dispatch(CounterAction::Increment);
        store.undo();
        assert!(store.can_redo());
        store.dispatch(CounterAction::Add(10));
        assert!(!store.can_redo());
        assert_eq!(store.state().count, 11);
    }

    #[test]
    fn test_undo_history_len() {
        let mut store = UndoStore::new(Counter { count: 0 }, counter_reducer);
        assert_eq!(store.history_len(), 0);
        store.dispatch(CounterAction::Increment);
        store.dispatch(CounterAction::Increment);
        assert_eq!(store.history_len(), 2);
        store.undo();
        assert_eq!(store.history_len(), 1);
    }

    #[test]
    fn test_complex_nested_state() {
        #[derive(Clone, Debug, PartialEq)]
        struct AppState {
            user: String,
            items: Vec<String>,
        }

        #[derive(Clone, Debug)]
        enum AppAction {
            AddItem(String),
            SetUser(String),
        }

        let reducer = |state: &AppState, action: &AppAction| -> AppState {
            match action {
                AppAction::AddItem(item) => {
                    let mut new = state.clone();
                    new.items.push(item.clone());
                    new
                }
                AppAction::SetUser(name) => AppState {
                    user: name.clone(),
                    ..state.clone()
                },
            }
        };

        let mut store = Store::new(
            AppState {
                user: "anon".into(),
                items: vec![],
            },
            reducer,
        );

        store.dispatch(AppAction::SetUser("alice".into()));
        store.dispatch(AppAction::AddItem("widget".into()));

        assert_eq!(store.state().user, "alice");
        assert_eq!(store.state().items, vec!["widget"]);
    }

    #[test]
    fn test_logging_middleware() {
        let (mw, log) = create_logging_middleware::<Counter, CounterAction>();
        let mut store = Store::new(Counter { count: 0 }, counter_reducer);
        store.add_middleware(mw);
        store.dispatch(CounterAction::Increment);
        assert_eq!(log.borrow().entries().len(), 1);
        assert!(log.borrow().entries()[0].1.contains("Increment"));
    }
}
