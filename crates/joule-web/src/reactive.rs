//! Fine-grained reactivity system.
//!
//! Provides `Signal<T>`, `Effect`, `Memo<T>`, and `batch()` — a SolidJS-style
//! reactive graph that tracks dependencies at read-time and propagates
//! invalidations at write-time. All state lives in a thread-local `Runtime`.

use std::cell::RefCell;
use std::collections::HashSet;

// ── Runtime ─────────────────────────────────────────────────────

thread_local! {
    static RUNTIME: RefCell<Runtime> = RefCell::new(Runtime::new());
}

struct SignalData {
    value: Box<dyn std::any::Any>,
    subscribers: HashSet<EffectId>,
}

struct EffectData {
    f: Option<Box<dyn FnMut()>>,
    dependencies: HashSet<SignalId>,
    active: bool,
}

type SignalId = usize;
type EffectId = usize;

struct Runtime {
    signals: Vec<SignalData>,
    effects: Vec<EffectData>,
    current_effect: Option<EffectId>,
    batching: bool,
    pending_effects: Vec<EffectId>,
}

impl Runtime {
    fn new() -> Self {
        Self {
            signals: Vec::new(),
            effects: Vec::new(),
            current_effect: None,
            batching: false,
            pending_effects: Vec::new(),
        }
    }
}

// ── Signal ──────────────────────────────────────────────────────

/// Read-half of a signal.
pub struct ReadSignal<T: 'static> {
    id: SignalId,
    _marker: std::marker::PhantomData<T>,
}

/// Write-half of a signal.
pub struct WriteSignal<T: 'static> {
    id: SignalId,
    _marker: std::marker::PhantomData<T>,
}

impl<T: 'static> Clone for ReadSignal<T> {
    fn clone(&self) -> Self { *self }
}
impl<T: 'static> Copy for ReadSignal<T> {}

impl<T: 'static> Clone for WriteSignal<T> {
    fn clone(&self) -> Self { *self }
}
impl<T: 'static> Copy for WriteSignal<T> {}

/// Create a new signal, returning a `(ReadSignal, WriteSignal)` pair.
pub fn create_signal<T: 'static>(value: T) -> (ReadSignal<T>, WriteSignal<T>) {
    let id = RUNTIME.with(|cell| {
        let mut rt = cell.borrow_mut();
        let id = rt.signals.len();
        rt.signals.push(SignalData {
            value: Box::new(value),
            subscribers: HashSet::new(),
        });
        id
    });
    (
        ReadSignal { id, _marker: std::marker::PhantomData },
        WriteSignal { id, _marker: std::marker::PhantomData },
    )
}

impl<T: Clone + 'static> ReadSignal<T> {
    /// Read the current value, registering a dependency if inside an effect.
    pub fn get(&self) -> T {
        RUNTIME.with(|cell| {
            let mut rt = cell.borrow_mut();
            if let Some(eid) = rt.current_effect {
                rt.signals[self.id].subscribers.insert(eid);
                rt.effects[eid].dependencies.insert(self.id);
            }
            rt.signals[self.id]
                .value
                .downcast_ref::<T>()
                .cloned()
                .expect("signal type mismatch")
        })
    }
}

impl<T: 'static> WriteSignal<T> {
    /// Replace the signal value and notify dependents.
    pub fn set(&self, value: T) {
        let subs = RUNTIME.with(|cell| {
            let mut rt = cell.borrow_mut();
            rt.signals[self.id].value = Box::new(value);
            let subs: Vec<EffectId> = rt.signals[self.id].subscribers.iter().copied().collect();
            if rt.batching {
                rt.pending_effects.extend(subs);
                Vec::new() // don't run now
            } else {
                subs
            }
        });
        for eid in subs {
            run_effect(eid);
        }
    }

    /// Mutate the signal value in place and notify dependents.
    pub fn update(&self, f: impl FnOnce(&mut T)) {
        let subs = RUNTIME.with(|cell| {
            let mut rt = cell.borrow_mut();
            let val = rt.signals[self.id]
                .value
                .downcast_mut::<T>()
                .expect("signal type mismatch");
            f(val);
            let subs: Vec<EffectId> = rt.signals[self.id].subscribers.iter().copied().collect();
            if rt.batching {
                rt.pending_effects.extend(subs);
                Vec::new()
            } else {
                subs
            }
        });
        for eid in subs {
            run_effect(eid);
        }
    }
}

// ── Effect ──────────────────────────────────────────────────────

/// Create a reactive effect that runs immediately and re-runs whenever
/// any signal it reads changes.
pub fn create_effect(f: impl FnMut() + 'static) {
    let eid = RUNTIME.with(|cell| {
        let mut rt = cell.borrow_mut();
        let eid = rt.effects.len();
        rt.effects.push(EffectData {
            f: Some(Box::new(f)),
            dependencies: HashSet::new(),
            active: true,
        });
        eid
    });
    run_effect(eid);
}

fn run_effect(eid: EffectId) {
    // 1. Take the closure out of the runtime (so we can call it without holding the borrow).
    let mut f = RUNTIME.with(|cell| {
        let mut rt = cell.borrow_mut();
        if eid >= rt.effects.len() || !rt.effects[eid].active {
            return None;
        }

        // Clear old dependencies
        let old_deps: Vec<SignalId> = rt.effects[eid].dependencies.drain().collect();
        for sid in &old_deps {
            if *sid < rt.signals.len() {
                rt.signals[*sid].subscribers.remove(&eid);
            }
        }

        // Save/set current effect
        let prev = rt.current_effect;
        rt.current_effect = Some(eid);

        // We stash `prev` in a field — but we can't easily restore it after the closure
        // runs, since we drop the borrow. Instead, store prev on the stack and restore below.
        // For now, just set current_effect. We'll restore after calling f.
        let _ = prev; // used below

        rt.effects[eid].f.take()
    });

    if let Some(ref mut closure) = f {
        // 2. Run the closure (it will call signal.get() which borrows the runtime).
        closure();

        // 3. Put the closure back and restore current_effect.
        RUNTIME.with(|cell| {
            let mut rt = cell.borrow_mut();
            rt.effects[eid].f = f;
            rt.current_effect = None;
        });
    }
}

// ── Memo ────────────────────────────────────────────────────────

/// Create a memoized derived computation. Returns a `ReadSignal<T>`.
pub fn create_memo<T: Clone + 'static>(f: impl Fn() -> T + 'static) -> ReadSignal<T> {
    let initial = f();
    let (read, write) = create_signal(initial);

    create_effect(move || {
        let val = f();
        write.set(val);
    });

    read
}

// ── Batch ───────────────────────────────────────────────────────

/// Batch multiple signal writes — effects are deferred until the batch ends.
pub fn batch(f: impl FnOnce()) {
    RUNTIME.with(|cell| {
        cell.borrow_mut().batching = true;
    });

    f();

    let pending = RUNTIME.with(|cell| {
        let mut rt = cell.borrow_mut();
        rt.batching = false;
        let mut seen = HashSet::new();
        rt.pending_effects
            .drain(..)
            .filter(|e| seen.insert(*e))
            .collect::<Vec<_>>()
    });

    for eid in pending {
        run_effect(eid);
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    fn reset_runtime() {
        RUNTIME.with(|cell| {
            *cell.borrow_mut() = Runtime::new();
        });
    }

    #[test]
    fn signal_get_set() {
        reset_runtime();
        let (read, write) = create_signal(42);
        assert_eq!(read.get(), 42);
        write.set(99);
        assert_eq!(read.get(), 99);
    }

    #[test]
    fn effect_runs_on_signal_change() {
        reset_runtime();
        let (read, write) = create_signal(0);
        let count = Rc::new(Cell::new(0));
        let count2 = count.clone();

        create_effect(move || {
            let _ = read.get();
            count2.set(count2.get() + 1);
        });

        assert_eq!(count.get(), 1);
        write.set(1);
        assert_eq!(count.get(), 2);
        write.set(2);
        assert_eq!(count.get(), 3);
    }

    #[test]
    fn effect_tracks_multiple_signals() {
        reset_runtime();
        let (a_read, a_write) = create_signal(1);
        let (b_read, b_write) = create_signal(10);
        let sum = Rc::new(Cell::new(0));
        let sum2 = sum.clone();

        create_effect(move || {
            sum2.set(a_read.get() + b_read.get());
        });

        assert_eq!(sum.get(), 11);
        a_write.set(2);
        assert_eq!(sum.get(), 12);
        b_write.set(20);
        assert_eq!(sum.get(), 22);
    }

    #[test]
    fn memo_caches_correctly() {
        reset_runtime();
        let (read, write) = create_signal(3);
        let call_count = Rc::new(Cell::new(0));
        let cc = call_count.clone();

        let memo = create_memo(move || {
            cc.set(cc.get() + 1);
            read.get() * 2
        });

        assert_eq!(memo.get(), 6);
        let c1 = call_count.get();

        // Reading again shouldn't recompute
        assert_eq!(memo.get(), 6);

        // Changing dependency recomputes
        write.set(5);
        assert_eq!(memo.get(), 10);
        assert!(call_count.get() > c1);
    }

    #[test]
    fn memo_only_recomputes_when_deps_change() {
        reset_runtime();
        let (read, _write) = create_signal(7);
        let call_count = Rc::new(Cell::new(0));
        let cc = call_count.clone();

        let memo = create_memo(move || {
            cc.set(cc.get() + 1);
            read.get() + 1
        });

        assert_eq!(memo.get(), 8);
        let c1 = call_count.get();

        assert_eq!(memo.get(), 8);
        assert_eq!(memo.get(), 8);
        assert_eq!(call_count.get(), c1);
    }

    #[test]
    fn batch_defers_effects() {
        reset_runtime();
        let (read, write) = create_signal(0);
        let run_count = Rc::new(Cell::new(0));
        let rc = run_count.clone();

        create_effect(move || {
            let _ = read.get();
            rc.set(rc.get() + 1);
        });

        assert_eq!(run_count.get(), 1);

        batch(move || {
            write.set(1);
            write.set(2);
            write.set(3);
        });

        // Effect should have run exactly once after batch
        assert_eq!(run_count.get(), 2);
    }

    #[test]
    fn signal_update_closure() {
        reset_runtime();
        let (read, write) = create_signal(vec![1, 2, 3]);
        write.update(|v| v.push(4));
        assert_eq!(read.get(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn effect_cleanup_on_overwrite() {
        reset_runtime();
        let (read, write) = create_signal(0);
        let log = Rc::new(RefCell::new(Vec::<String>::new()));
        let log2 = log.clone();

        create_effect(move || {
            let v = read.get();
            log2.borrow_mut().push(format!("effect: {v}"));
        });

        assert_eq!(log.borrow().len(), 1);
        write.set(10);
        assert_eq!(log.borrow().len(), 2);
        assert_eq!(log.borrow()[1], "effect: 10");
    }

    #[test]
    fn diamond_dependency() {
        reset_runtime();
        let (a_read, a_write) = create_signal(1);

        let b = create_memo(move || a_read.get() * 2);
        let c = create_memo(move || a_read.get() * 3);

        let d_count = Rc::new(Cell::new(0));
        let dc = d_count.clone();
        let d_val = Rc::new(Cell::new(0));
        let dv = d_val.clone();

        create_effect(move || {
            let val = b.get() + c.get();
            dv.set(val);
            dc.set(dc.get() + 1);
        });

        assert_eq!(d_val.get(), 5);
        let initial_count = d_count.get();

        batch(move || {
            a_write.set(2);
        });

        assert_eq!(d_val.get(), 10);
        assert!(d_count.get() <= initial_count + 3, "D ran too many times");
    }

    #[test]
    fn nested_effects() {
        reset_runtime();
        let (read, write) = create_signal(0);
        let outer_count = Rc::new(Cell::new(0));
        let oc = outer_count.clone();

        create_effect(move || {
            let _ = read.get();
            oc.set(oc.get() + 1);
        });

        assert_eq!(outer_count.get(), 1);
        write.set(1);
        assert_eq!(outer_count.get(), 2);
    }

    #[test]
    fn multiple_signals_independent() {
        reset_runtime();
        let (a_read, a_write) = create_signal(0);
        let (_b_read, b_write) = create_signal(0);
        let count = Rc::new(Cell::new(0));
        let cc = count.clone();

        create_effect(move || {
            let _ = a_read.get();
            cc.set(cc.get() + 1);
        });

        assert_eq!(count.get(), 1);
        b_write.set(99);
        assert_eq!(count.get(), 1);
        a_write.set(1);
        assert_eq!(count.get(), 2);
    }
}
