//! Observable/Observer pattern with operators and lifecycle management.
//!
//! Provides `Observable` values with subscriber notification, map/filter/combine
//! operators, debounce/throttle, error propagation, completion signaling,
//! and subscription cleanup.

use std::cell::RefCell;
use std::rc::Rc;

// ── Notification ─────────────────────────────────────────────

/// A notification emitted by an observable.
#[derive(Debug, Clone, PartialEq)]
pub enum Notification<T> {
    /// A value emission.
    Next(T),
    /// An error occurred; the observable terminates.
    Error(String),
    /// The observable completed normally.
    Complete,
}

// ── Subscription ─────────────────────────────────────────────

/// Handle for managing a subscription. Dropping it does not auto-unsubscribe;
/// call `unsubscribe()` explicitly or use the observable's `unsubscribe(id)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(pub usize);

// ── Observable ───────────────────────────────────────────────

/// A synchronous observable with lifecycle (next/error/complete).
pub struct Observable<T> {
    subscribers: Vec<Option<Box<dyn FnMut(&Notification<T>)>>>,
    next_id: usize,
    completed: bool,
    errored: bool,
}

impl<T> Default for Observable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Observable<T> {
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
            next_id: 0,
            completed: false,
            errored: false,
        }
    }

    /// Whether this observable has completed or errored.
    pub fn is_terminated(&self) -> bool {
        self.completed || self.errored
    }

    /// Subscribe to notifications. Returns an ID for unsubscription.
    pub fn subscribe(&mut self, f: impl FnMut(&Notification<T>) + 'static) -> SubscriptionId {
        let id = self.next_id;
        while self.subscribers.len() <= id {
            self.subscribers.push(None);
        }
        self.subscribers[id] = Some(Box::new(f));
        self.next_id += 1;
        SubscriptionId(id)
    }

    /// Subscribe only to `Next` values (convenience).
    pub fn subscribe_next(&mut self, mut f: impl FnMut(&T) + 'static) -> SubscriptionId {
        self.subscribe(move |n| {
            if let Notification::Next(v) = n {
                f(v);
            }
        })
    }

    /// Remove a subscription.
    pub fn unsubscribe(&mut self, id: SubscriptionId) {
        if id.0 < self.subscribers.len() {
            self.subscribers[id.0] = None;
        }
    }

    /// Remove all subscriptions.
    pub fn unsubscribe_all(&mut self) {
        for slot in &mut self.subscribers {
            *slot = None;
        }
    }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.iter().filter(|s| s.is_some()).count()
    }

    fn notify(&mut self, notification: Notification<T>)
    where
        T: Clone,
    {
        if self.is_terminated() {
            return;
        }
        match &notification {
            Notification::Complete => self.completed = true,
            Notification::Error(_) => self.errored = true,
            Notification::Next(_) => {}
        }
        for slot in &mut self.subscribers {
            if let Some(f) = slot.as_mut() {
                f(&notification);
            }
        }
        // Clean up subscribers on termination
        if self.is_terminated() {
            self.unsubscribe_all();
        }
    }

    /// Emit a value.
    pub fn emit(&mut self, value: T)
    where
        T: Clone,
    {
        self.notify(Notification::Next(value));
    }

    /// Signal completion.
    pub fn complete(&mut self)
    where
        T: Clone,
    {
        self.notify(Notification::Complete);
    }

    /// Signal an error.
    pub fn error(&mut self, msg: impl Into<String>)
    where
        T: Clone,
    {
        self.notify(Notification::Error(msg.into()));
    }
}

// ── Operators (free functions on slices) ─────────────────────

/// Map each element through a function.
pub fn map<T, U>(source: &[T], f: impl Fn(&T) -> U) -> Vec<U> {
    source.iter().map(f).collect()
}

/// Filter elements by predicate.
pub fn filter<T: Clone>(values: &[T], f: impl Fn(&T) -> bool) -> Vec<T> {
    values.iter().filter(|v| f(v)).cloned().collect()
}

/// Take the first n elements.
pub fn take<T: Clone>(values: &[T], n: usize) -> Vec<T> {
    values.iter().take(n).cloned().collect()
}

/// Skip the first n elements.
pub fn skip<T: Clone>(values: &[T], n: usize) -> Vec<T> {
    values.iter().skip(n).cloned().collect()
}

/// Remove duplicate elements, keeping first occurrence.
pub fn distinct<T: Clone + PartialEq>(values: &[T]) -> Vec<T> {
    let mut out = Vec::new();
    for v in values {
        if !out.contains(v) {
            out.push(v.clone());
        }
    }
    out
}

/// Running accumulation over elements.
pub fn scan<T, U: Clone>(values: &[T], initial: U, f: impl Fn(U, &T) -> U) -> Vec<U> {
    let mut acc = initial;
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        acc = f(acc, v);
        out.push(acc.clone());
    }
    out
}

/// Interleave two slices: a[0], b[0], a[1], b[1], ...
pub fn merge<T: Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut ai = a.iter();
    let mut bi = b.iter();
    loop {
        match (ai.next(), bi.next()) {
            (Some(av), Some(bv)) => {
                out.push(av.clone());
                out.push(bv.clone());
            }
            (Some(av), None) => {
                out.push(av.clone());
                out.extend(ai.cloned());
                break;
            }
            (None, Some(bv)) => {
                out.push(bv.clone());
                out.extend(bi.cloned());
                break;
            }
            (None, None) => break,
        }
    }
    out
}

/// Combine two slices element-wise with a combiner function.
pub fn combine_latest<A: Clone, B: Clone, R>(
    a: &[A],
    b: &[B],
    f: impl Fn(&A, &B) -> R,
) -> Vec<R> {
    let len = a.len().min(b.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(f(&a[i], &b[i]));
    }
    out
}

/// Group events by time windows. Each `(T, u64)` is (value, timestamp_ms).
pub fn debounce_collect<T: Clone>(events: &[(T, u64)], window_ms: u64) -> Vec<Vec<T>> {
    if events.is_empty() {
        return Vec::new();
    }
    let mut groups: Vec<Vec<T>> = Vec::new();
    let mut current: Vec<T> = vec![events[0].0.clone()];
    let mut window_start = events[0].1;

    for (val, ts) in &events[1..] {
        if ts - window_start <= window_ms {
            current.push(val.clone());
        } else {
            groups.push(std::mem::take(&mut current));
            current.push(val.clone());
            window_start = *ts;
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Throttle: keep only one event per `window_ms`.
pub fn throttle<T: Clone>(events: &[(T, u64)], window_ms: u64) -> Vec<T> {
    if events.is_empty() {
        return Vec::new();
    }
    let mut out = vec![events[0].0.clone()];
    let mut last_ts = events[0].1;
    for (val, ts) in &events[1..] {
        if ts - last_ts >= window_ms {
            out.push(val.clone());
            last_ts = *ts;
        }
    }
    out
}

/// Debounce: keep only the last event after a quiet period of `window_ms`.
pub fn debounce<T: Clone>(events: &[(T, u64)], window_ms: u64) -> Vec<T> {
    if events.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..events.len() {
        let is_last = i + 1 == events.len();
        let next_gap = if is_last {
            window_ms + 1
        } else {
            events[i + 1].1 - events[i].1
        };
        if next_gap > window_ms {
            out.push(events[i].0.clone());
        }
    }
    out
}

// ── Subject ──────────────────────────────────────────────────

/// A subject is both an observable and an observer.
pub struct Subject<T> {
    observable: Observable<T>,
    last_value: Option<T>,
}

impl<T> Default for Subject<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Subject<T> {
    pub fn new() -> Self {
        Self {
            observable: Observable::new(),
            last_value: None,
        }
    }

    pub fn subscribe(&mut self, f: impl FnMut(&Notification<T>) + 'static) -> SubscriptionId {
        self.observable.subscribe(f)
    }

    pub fn subscribe_next(&mut self, f: impl FnMut(&T) + 'static) -> SubscriptionId {
        self.observable.subscribe_next(f)
    }

    pub fn unsubscribe(&mut self, id: SubscriptionId) {
        self.observable.unsubscribe(id);
    }

    pub fn last_value(&self) -> Option<&T> {
        self.last_value.as_ref()
    }

    pub fn is_terminated(&self) -> bool {
        self.observable.is_terminated()
    }
}

impl<T: Clone> Subject<T> {
    pub fn next(&mut self, value: T) {
        self.last_value = Some(value.clone());
        self.observable.emit(value);
    }

    pub fn complete(&mut self) {
        self.observable.complete();
    }

    pub fn error(&mut self, msg: impl Into<String>) {
        self.observable.error(msg);
    }
}

// ── BehaviorSubject ──────────────────────────────────────────

/// Like `Subject` but always has a current value. New subscribers
/// receive the current value immediately.
pub struct BehaviorSubject<T> {
    value: T,
    observable: Observable<T>,
}

impl<T: Clone> BehaviorSubject<T> {
    pub fn new(initial: T) -> Self {
        Self {
            value: initial,
            observable: Observable::new(),
        }
    }

    pub fn subscribe(&mut self, mut f: impl FnMut(&Notification<T>) + 'static) -> SubscriptionId {
        f(&Notification::Next(self.value.clone()));
        self.observable.subscribe(f)
    }

    pub fn subscribe_next(&mut self, mut f: impl FnMut(&T) + 'static) -> SubscriptionId {
        f(&self.value);
        self.observable.subscribe_next(f)
    }

    pub fn unsubscribe(&mut self, id: SubscriptionId) {
        self.observable.unsubscribe(id);
    }

    pub fn next(&mut self, value: T) {
        self.value = value.clone();
        self.observable.emit(value);
    }

    pub fn complete(&mut self) {
        self.observable.complete();
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

// ── ReplaySubject ────────────────────────────────────────────

/// Replays the last `buffer_size` values to new subscribers.
pub struct ReplaySubject<T> {
    buffer: Vec<T>,
    buffer_size: usize,
    observable: Observable<T>,
}

impl<T: Clone> ReplaySubject<T> {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            buffer_size,
            observable: Observable::new(),
        }
    }

    pub fn subscribe(&mut self, mut f: impl FnMut(&Notification<T>) + 'static) -> SubscriptionId {
        for v in &self.buffer {
            f(&Notification::Next(v.clone()));
        }
        self.observable.subscribe(f)
    }

    pub fn next(&mut self, value: T) {
        self.buffer.push(value.clone());
        if self.buffer.len() > self.buffer_size {
            self.buffer.remove(0);
        }
        self.observable.emit(value);
    }

    pub fn unsubscribe(&mut self, id: SubscriptionId) {
        self.observable.unsubscribe(id);
    }

    pub fn buffer(&self) -> &[T] {
        &self.buffer
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_notifies_subscribers() {
        let received = Rc::new(RefCell::new(Vec::new()));
        let r = received.clone();
        let mut obs = Observable::<i32>::new();
        obs.subscribe_next(move |v| {
            r.borrow_mut().push(*v);
        });
        obs.emit(1);
        obs.emit(2);
        assert_eq!(*received.borrow(), vec![1, 2]);
    }

    #[test]
    fn unsubscribe_stops_delivery() {
        let received = Rc::new(RefCell::new(Vec::new()));
        let r = received.clone();
        let mut obs = Observable::<i32>::new();
        let id = obs.subscribe_next(move |v| {
            r.borrow_mut().push(*v);
        });
        obs.emit(1);
        obs.unsubscribe(id);
        obs.emit(2);
        assert_eq!(*received.borrow(), vec![1]);
    }

    #[test]
    fn complete_terminates_observable() {
        let received = Rc::new(RefCell::new(Vec::new()));
        let r = received.clone();
        let mut obs = Observable::<i32>::new();
        obs.subscribe_next(move |v| {
            r.borrow_mut().push(*v);
        });
        obs.emit(1);
        obs.complete();
        obs.emit(2); // should be ignored
        assert_eq!(*received.borrow(), vec![1]);
        assert!(obs.is_terminated());
    }

    #[test]
    fn error_terminates_observable() {
        let err_msg = Rc::new(RefCell::new(String::new()));
        let e = err_msg.clone();
        let mut obs = Observable::<i32>::new();
        obs.subscribe(move |n| {
            if let Notification::Error(msg) = n {
                *e.borrow_mut() = msg.clone();
            }
        });
        obs.error("something failed");
        assert_eq!(*err_msg.borrow(), "something failed");
        assert!(obs.is_terminated());
    }

    #[test]
    fn completion_notification_received() {
        let completed = Rc::new(RefCell::new(false));
        let c = completed.clone();
        let mut obs = Observable::<i32>::new();
        obs.subscribe(move |n| {
            if matches!(n, Notification::Complete) {
                *c.borrow_mut() = true;
            }
        });
        obs.complete();
        assert!(*completed.borrow());
    }

    #[test]
    fn map_transforms() {
        let data = vec![1, 2, 3];
        let result = map(&data, |v| v * 10);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn filter_selects() {
        let data = vec![1, 2, 3, 4, 5];
        let result = filter(&data, |v| v % 2 == 0);
        assert_eq!(result, vec![2, 4]);
    }

    #[test]
    fn take_and_skip() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(take(&data, 3), vec![1, 2, 3]);
        assert_eq!(skip(&data, 3), vec![4, 5]);
    }

    #[test]
    fn scan_accumulates() {
        let data = vec![1, 2, 3, 4];
        let result = scan(&data, 0, |acc, v| acc + v);
        assert_eq!(result, vec![1, 3, 6, 10]);
    }

    #[test]
    fn distinct_removes_dupes() {
        let data = vec![1, 2, 2, 3, 1, 4, 3];
        let result = distinct(&data);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn merge_interleaves() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4];
        let result = merge(&a, &b);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn combine_latest_pairs() {
        let a = vec![1, 2, 3];
        let b = vec![10, 20, 30];
        let result = combine_latest(&a, &b, |x, y| x + y);
        assert_eq!(result, vec![11, 22, 33]);
    }

    #[test]
    fn combine_latest_unequal_lengths() {
        let a = vec![1, 2];
        let b = vec![10, 20, 30];
        let result = combine_latest(&a, &b, |x, y| x + y);
        assert_eq!(result, vec![11, 22]);
    }

    #[test]
    fn debounce_collect_groups() {
        let events = vec![
            ("a", 0),
            ("b", 10),
            ("c", 20),
            ("d", 200),
            ("e", 210),
            ("f", 500),
        ];
        let groups = debounce_collect(&events, 50);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec!["a", "b", "c"]);
        assert_eq!(groups[1], vec!["d", "e"]);
        assert_eq!(groups[2], vec!["f"]);
    }

    #[test]
    fn throttle_limits_rate() {
        let events = vec![("a", 0), ("b", 10), ("c", 50), ("d", 60), ("e", 110)];
        let result = throttle(&events, 50);
        assert_eq!(result, vec!["a", "c", "e"]);
    }

    #[test]
    fn debounce_keeps_last_before_gap() {
        let events = vec![("a", 0), ("b", 10), ("c", 20), ("d", 200)];
        let result = debounce(&events, 50);
        assert_eq!(result, vec!["c", "d"]);
    }

    #[test]
    fn subject_last_value() {
        let mut sub = Subject::<i32>::new();
        assert!(sub.last_value().is_none());
        sub.next(42);
        assert_eq!(sub.last_value(), Some(&42));
        sub.next(99);
        assert_eq!(sub.last_value(), Some(&99));
    }

    #[test]
    fn subject_completion() {
        let completed = Rc::new(RefCell::new(false));
        let c = completed.clone();
        let mut sub = Subject::<i32>::new();
        sub.subscribe(move |n| {
            if matches!(n, Notification::Complete) {
                *c.borrow_mut() = true;
            }
        });
        sub.complete();
        assert!(*completed.borrow());
        assert!(sub.is_terminated());
    }

    #[test]
    fn behavior_subject_initial_value() {
        let received = Rc::new(RefCell::new(Vec::new()));
        let r = received.clone();
        let mut bs = BehaviorSubject::new(10);
        bs.subscribe_next(move |v| {
            r.borrow_mut().push(*v);
        });
        assert_eq!(received.borrow()[0], 10);
        bs.next(20);
        assert_eq!(*received.borrow(), vec![10, 20]);
        assert_eq!(*bs.value(), 20);
    }

    #[test]
    fn multiple_subscribers() {
        let r1 = Rc::new(RefCell::new(0));
        let r2 = Rc::new(RefCell::new(0));
        let r1c = r1.clone();
        let r2c = r2.clone();
        let mut obs = Observable::<i32>::new();
        obs.subscribe_next(move |v| {
            *r1c.borrow_mut() += v;
        });
        obs.subscribe_next(move |v| {
            *r2c.borrow_mut() += v * 2;
        });
        obs.emit(5);
        assert_eq!(*r1.borrow(), 5);
        assert_eq!(*r2.borrow(), 10);
    }

    #[test]
    fn unsubscribe_all_clears() {
        let count = Rc::new(RefCell::new(0));
        let c = count.clone();
        let mut obs = Observable::<i32>::new();
        obs.subscribe_next(move |_| {
            *c.borrow_mut() += 1;
        });
        obs.emit(1);
        assert_eq!(*count.borrow(), 1);
        obs.unsubscribe_all();
        obs.emit(2);
        assert_eq!(*count.borrow(), 1);
        assert_eq!(obs.subscriber_count(), 0);
    }

    #[test]
    fn replay_subject_replays_buffer() {
        let received = Rc::new(RefCell::new(Vec::new()));
        let mut rs = ReplaySubject::<i32>::new(3);
        rs.next(1);
        rs.next(2);
        rs.next(3);
        rs.next(4); // buffer is [2, 3, 4]

        let r = received.clone();
        rs.subscribe(move |n| {
            if let Notification::Next(v) = n {
                r.borrow_mut().push(*v);
            }
        });
        // Should have received 2, 3, 4 from replay
        assert_eq!(*received.borrow(), vec![2, 3, 4]);

        // New value should also arrive
        rs.next(5);
        assert_eq!(*received.borrow(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn subscriber_count_tracks() {
        let mut obs = Observable::<i32>::new();
        assert_eq!(obs.subscriber_count(), 0);
        let id1 = obs.subscribe_next(|_| {});
        assert_eq!(obs.subscriber_count(), 1);
        let _id2 = obs.subscribe_next(|_| {});
        assert_eq!(obs.subscriber_count(), 2);
        obs.unsubscribe(id1);
        assert_eq!(obs.subscriber_count(), 1);
    }

    #[test]
    fn throttle_empty() {
        let result: Vec<&str> = throttle(&[], 50);
        assert!(result.is_empty());
    }

    #[test]
    fn debounce_empty() {
        let result: Vec<&str> = debounce(&[], 50);
        assert!(result.is_empty());
    }
}
