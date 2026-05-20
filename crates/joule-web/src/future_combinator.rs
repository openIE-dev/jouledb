//! Future/Promise combinators — synchronous simulation of poll-based futures.
//!
//! Replaces JavaScript's `Promise.all`, `Promise.race`, `p-retry`, etc. with
//! a pure-Rust poll-based future model. Supports map, flat_map, and_then,
//! join (wait-all), race (first-to-complete), timeout, retry, and chaining.

use std::collections::HashMap;

// ── Poll Result ─────────────────────────────────────────────────

/// Result of polling a future.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Poll<T> {
    /// The future has completed with a value.
    Ready(T),
    /// The future is not yet complete.
    Pending,
}

impl<T> Poll<T> {
    pub fn is_ready(&self) -> bool {
        matches!(self, Poll::Ready(_))
    }

    pub fn is_pending(&self) -> bool {
        matches!(self, Poll::Pending)
    }

    /// Map over the ready value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Poll<U> {
        match self {
            Poll::Ready(v) => Poll::Ready(f(v)),
            Poll::Pending => Poll::Pending,
        }
    }
}

// ── Future Error ────────────────────────────────────────────────

/// Errors in the future combinator system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FutureError {
    /// Future not found.
    NotFound(u64),
    /// Future timed out.
    Timeout { future_id: u64, deadline_tick: u64 },
    /// All retries exhausted.
    RetriesExhausted { future_id: u64, attempts: u32 },
    /// Future already completed.
    AlreadyCompleted(u64),
    /// Future failed with a reason.
    Failed(String),
}

impl std::fmt::Display for FutureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "future not found: {id}"),
            Self::Timeout { future_id, deadline_tick } => {
                write!(f, "future {future_id} timed out at tick {deadline_tick}")
            }
            Self::RetriesExhausted { future_id, attempts } => {
                write!(f, "future {future_id}: all {attempts} retries exhausted")
            }
            Self::AlreadyCompleted(id) => write!(f, "future {id} already completed"),
            Self::Failed(reason) => write!(f, "future failed: {reason}"),
        }
    }
}

impl std::error::Error for FutureError {}

// ── Future State ────────────────────────────────────────────────

/// State of a managed future.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FutureState {
    Pending,
    Ready(String),
    Failed(String),
    TimedOut,
}

// ── Future Kind ─────────────────────────────────────────────────

/// Kind of future in the executor.
#[derive(Debug, Clone)]
enum FutureKind {
    /// A basic value future that resolves at a given tick.
    Value {
        value: String,
        ready_at_tick: u64,
    },
    /// Map combinator: transforms the result of another future.
    Map {
        source_id: u64,
        transform_suffix: String,
    },
    /// FlatMap / AndThen: chains to another future after source resolves.
    AndThen {
        source_id: u64,
        next_suffix: String,
        chained_id: Option<u64>,
    },
    /// Join: waits for all sub-futures to complete.
    Join {
        sub_ids: Vec<u64>,
    },
    /// Race: resolves when the first sub-future completes.
    Race {
        sub_ids: Vec<u64>,
    },
    /// Timeout: wraps a future with a deadline.
    Timeout {
        inner_id: u64,
        deadline_tick: u64,
    },
    /// Retry: retries a failing future up to N times.
    Retry {
        attempts: Vec<u64>,
        max_retries: u32,
    },
}

/// A managed future in the executor.
#[derive(Debug, Clone)]
struct ManagedFuture {
    id: u64,
    state: FutureState,
    kind: FutureKind,
}

// ── Future Executor ─────────────────────────────────────────────

/// Drives poll-based futures to completion.
pub struct FutureExecutor {
    futures: HashMap<u64, ManagedFuture>,
    next_id: u64,
    current_tick: u64,
}

impl FutureExecutor {
    pub fn new() -> Self {
        Self {
            futures: HashMap::new(),
            next_id: 1,
            current_tick: 0,
        }
    }

    /// Advance the simulated clock.
    pub fn tick(&mut self, ticks: u64) {
        self.current_tick += ticks;
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Create a value future that resolves at a given tick.
    pub fn value(&mut self, value: impl Into<String>, ready_at_tick: u64) -> u64 {
        self.insert(FutureKind::Value {
            value: value.into(),
            ready_at_tick,
        })
    }

    /// Create a future that resolves immediately.
    pub fn ready(&mut self, value: impl Into<String>) -> u64 {
        self.value(value, 0)
    }

    /// Map: transform the result of a future.
    pub fn map(&mut self, source_id: u64, suffix: impl Into<String>) -> u64 {
        self.insert(FutureKind::Map {
            source_id,
            transform_suffix: suffix.into(),
        })
    }

    /// AndThen / FlatMap: chain a dependent future after source resolves.
    pub fn and_then(&mut self, source_id: u64, suffix: impl Into<String>) -> u64 {
        self.insert(FutureKind::AndThen {
            source_id,
            next_suffix: suffix.into(),
            chained_id: None,
        })
    }

    /// Join: wait for all futures to complete. Result is all values joined by `,`.
    pub fn join(&mut self, ids: Vec<u64>) -> u64 {
        self.insert(FutureKind::Join { sub_ids: ids })
    }

    /// Race: resolve with the first future that completes.
    pub fn race(&mut self, ids: Vec<u64>) -> u64 {
        self.insert(FutureKind::Race { sub_ids: ids })
    }

    /// Timeout: wraps a future with a deadline tick.
    pub fn timeout(&mut self, inner_id: u64, deadline_tick: u64) -> u64 {
        self.insert(FutureKind::Timeout {
            inner_id,
            deadline_tick,
        })
    }

    /// Retry: retries a set of attempt futures in order.
    pub fn retry(&mut self, attempt_ids: Vec<u64>, max_retries: u32) -> u64 {
        self.insert(FutureKind::Retry {
            attempts: attempt_ids,
            max_retries,
        })
    }

    /// Fail a future explicitly.
    pub fn fail(&mut self, id: u64, reason: impl Into<String>) -> Result<(), FutureError> {
        let fut = self.futures.get_mut(&id).ok_or(FutureError::NotFound(id))?;
        fut.state = FutureState::Failed(reason.into());
        Ok(())
    }

    fn insert(&mut self, kind: FutureKind) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.futures.insert(
            id,
            ManagedFuture {
                id,
                state: FutureState::Pending,
                kind,
            },
        );
        id
    }

    /// Poll a specific future. Drives it one step towards completion.
    pub fn poll(&mut self, id: u64) -> Result<Poll<String>, FutureError> {
        if !self.futures.contains_key(&id) {
            return Err(FutureError::NotFound(id));
        }

        // Check if already resolved
        let state = self.futures[&id].state.clone();
        match &state {
            FutureState::Ready(v) => return Ok(Poll::Ready(v.clone())),
            FutureState::Failed(e) => return Err(FutureError::Failed(e.clone())),
            FutureState::TimedOut => {
                return Err(FutureError::Timeout {
                    future_id: id,
                    deadline_tick: 0,
                })
            }
            FutureState::Pending => {}
        }

        let kind = self.futures[&id].kind.clone();
        match kind {
            FutureKind::Value { value, ready_at_tick } => {
                if self.current_tick >= ready_at_tick {
                    let fut = self.futures.get_mut(&id).unwrap();
                    fut.state = FutureState::Ready(value.clone());
                    Ok(Poll::Ready(value))
                } else {
                    Ok(Poll::Pending)
                }
            }
            FutureKind::Map { source_id, transform_suffix } => {
                let source_poll = self.poll(source_id)?;
                match source_poll {
                    Poll::Ready(v) => {
                        let result = format!("{v}{transform_suffix}");
                        let fut = self.futures.get_mut(&id).unwrap();
                        fut.state = FutureState::Ready(result.clone());
                        Ok(Poll::Ready(result))
                    }
                    Poll::Pending => Ok(Poll::Pending),
                }
            }
            FutureKind::AndThen { source_id, next_suffix, chained_id } => {
                if let Some(cid) = chained_id {
                    // Poll the chained future
                    let chained_poll = self.poll(cid)?;
                    match chained_poll {
                        Poll::Ready(v) => {
                            let fut = self.futures.get_mut(&id).unwrap();
                            fut.state = FutureState::Ready(v.clone());
                            Ok(Poll::Ready(v))
                        }
                        Poll::Pending => Ok(Poll::Pending),
                    }
                } else {
                    // Poll the source
                    let source_poll = self.poll(source_id)?;
                    match source_poll {
                        Poll::Ready(v) => {
                            // Create the chained future
                            let chained_value = format!("{v}{next_suffix}");
                            let cid = self.ready(chained_value);
                            let fut = self.futures.get_mut(&id).unwrap();
                            if let FutureKind::AndThen { chained_id, .. } = &mut fut.kind {
                                *chained_id = Some(cid);
                            }
                            // Poll again to resolve the chain
                            self.poll(id)
                        }
                        Poll::Pending => Ok(Poll::Pending),
                    }
                }
            }
            FutureKind::Join { sub_ids } => {
                let mut all_ready = true;
                let mut results = Vec::new();
                for sid in &sub_ids {
                    match self.poll(*sid)? {
                        Poll::Ready(v) => results.push(v),
                        Poll::Pending => {
                            all_ready = false;
                            break;
                        }
                    }
                }
                if all_ready {
                    let joined = results.join(",");
                    let fut = self.futures.get_mut(&id).unwrap();
                    fut.state = FutureState::Ready(joined.clone());
                    Ok(Poll::Ready(joined))
                } else {
                    Ok(Poll::Pending)
                }
            }
            FutureKind::Race { sub_ids } => {
                for sid in &sub_ids {
                    if let Poll::Ready(v) = self.poll(*sid)? {
                        let fut = self.futures.get_mut(&id).unwrap();
                        fut.state = FutureState::Ready(v.clone());
                        return Ok(Poll::Ready(v));
                    }
                }
                Ok(Poll::Pending)
            }
            FutureKind::Timeout { inner_id, deadline_tick } => {
                if self.current_tick >= deadline_tick {
                    // Check if inner resolved first
                    let inner_state = self.futures.get(&inner_id).map(|f| f.state.clone());
                    if let Some(FutureState::Ready(v)) = inner_state {
                        let fut = self.futures.get_mut(&id).unwrap();
                        fut.state = FutureState::Ready(v.clone());
                        return Ok(Poll::Ready(v));
                    }
                    let fut = self.futures.get_mut(&id).unwrap();
                    fut.state = FutureState::TimedOut;
                    return Err(FutureError::Timeout {
                        future_id: id,
                        deadline_tick,
                    });
                }
                match self.poll(inner_id)? {
                    Poll::Ready(v) => {
                        let fut = self.futures.get_mut(&id).unwrap();
                        fut.state = FutureState::Ready(v.clone());
                        Ok(Poll::Ready(v))
                    }
                    Poll::Pending => Ok(Poll::Pending),
                }
            }
            FutureKind::Retry { attempts, max_retries } => {
                let max = max_retries as usize;
                for (i, aid) in attempts.iter().enumerate() {
                    if i >= max {
                        break;
                    }
                    let result = self.poll(*aid);
                    match result {
                        Ok(Poll::Ready(v)) => {
                            let fut = self.futures.get_mut(&id).unwrap();
                            fut.state = FutureState::Ready(v.clone());
                            return Ok(Poll::Ready(v));
                        }
                        Ok(Poll::Pending) => return Ok(Poll::Pending),
                        Err(FutureError::Failed(_)) => {
                            // Try next attempt
                            continue;
                        }
                        Err(e) => return Err(e),
                    }
                }
                Err(FutureError::RetriesExhausted {
                    future_id: id,
                    attempts: max_retries,
                })
            }
        }
    }

    /// Get the state of a future.
    pub fn state(&self, id: u64) -> Option<&FutureState> {
        self.futures.get(&id).map(|f| &f.state)
    }

    /// Count of managed futures.
    pub fn count(&self) -> usize {
        self.futures.len()
    }

    /// Count of pending futures.
    pub fn pending_count(&self) -> usize {
        self.futures
            .values()
            .filter(|f| f.state == FutureState::Pending)
            .count()
    }
}

impl Default for FutureExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ready_future() {
        let mut exec = FutureExecutor::new();
        let id = exec.ready("done");
        let result = exec.poll(id).unwrap();
        assert_eq!(result, Poll::Ready("done".to_string()));
    }

    #[test]
    fn test_pending_then_ready() {
        let mut exec = FutureExecutor::new();
        let id = exec.value("result", 10);
        assert_eq!(exec.poll(id).unwrap(), Poll::Pending);
        exec.tick(10);
        assert_eq!(exec.poll(id).unwrap(), Poll::Ready("result".to_string()));
    }

    #[test]
    fn test_map_combinator() {
        let mut exec = FutureExecutor::new();
        let base = exec.ready("hello");
        let mapped = exec.map(base, "_world");
        let result = exec.poll(mapped).unwrap();
        assert_eq!(result, Poll::Ready("hello_world".to_string()));
    }

    #[test]
    fn test_map_pending_source() {
        let mut exec = FutureExecutor::new();
        let base = exec.value("hello", 10);
        let mapped = exec.map(base, "_world");
        assert_eq!(exec.poll(mapped).unwrap(), Poll::Pending);
        exec.tick(10);
        assert_eq!(
            exec.poll(mapped).unwrap(),
            Poll::Ready("hello_world".to_string())
        );
    }

    #[test]
    fn test_and_then_combinator() {
        let mut exec = FutureExecutor::new();
        let base = exec.ready("step1");
        let chained = exec.and_then(base, "_step2");
        let result = exec.poll(chained).unwrap();
        assert_eq!(result, Poll::Ready("step1_step2".to_string()));
    }

    #[test]
    fn test_join_all_ready() {
        let mut exec = FutureExecutor::new();
        let a = exec.ready("a");
        let b = exec.ready("b");
        let c = exec.ready("c");
        let joined = exec.join(vec![a, b, c]);
        let result = exec.poll(joined).unwrap();
        assert_eq!(result, Poll::Ready("a,b,c".to_string()));
    }

    #[test]
    fn test_join_some_pending() {
        let mut exec = FutureExecutor::new();
        let a = exec.ready("a");
        let b = exec.value("b", 10);
        let joined = exec.join(vec![a, b]);
        assert_eq!(exec.poll(joined).unwrap(), Poll::Pending);
        exec.tick(10);
        assert_eq!(
            exec.poll(joined).unwrap(),
            Poll::Ready("a,b".to_string())
        );
    }

    #[test]
    fn test_race_first_wins() {
        let mut exec = FutureExecutor::new();
        let fast = exec.value("fast", 5);
        let slow = exec.value("slow", 20);
        let raced = exec.race(vec![fast, slow]);
        exec.tick(5);
        let result = exec.poll(raced).unwrap();
        assert_eq!(result, Poll::Ready("fast".to_string()));
    }

    #[test]
    fn test_race_all_pending() {
        let mut exec = FutureExecutor::new();
        let a = exec.value("a", 10);
        let b = exec.value("b", 20);
        let raced = exec.race(vec![a, b]);
        assert_eq!(exec.poll(raced).unwrap(), Poll::Pending);
    }

    #[test]
    fn test_timeout_succeeds() {
        let mut exec = FutureExecutor::new();
        let inner = exec.value("ok", 5);
        let tid = exec.timeout(inner, 10);
        exec.tick(5);
        let result = exec.poll(tid).unwrap();
        assert_eq!(result, Poll::Ready("ok".to_string()));
    }

    #[test]
    fn test_timeout_expires() {
        let mut exec = FutureExecutor::new();
        let inner = exec.value("ok", 20);
        let tid = exec.timeout(inner, 10);
        exec.tick(10);
        let result = exec.poll(tid);
        assert!(matches!(result, Err(FutureError::Timeout { .. })));
    }

    #[test]
    fn test_retry_succeeds_first() {
        let mut exec = FutureExecutor::new();
        let a1 = exec.ready("success");
        let retry_id = exec.retry(vec![a1], 3);
        let result = exec.poll(retry_id).unwrap();
        assert_eq!(result, Poll::Ready("success".to_string()));
    }

    #[test]
    fn test_retry_succeeds_second() {
        let mut exec = FutureExecutor::new();
        let a1 = exec.ready("fail");
        exec.fail(a1, "err").unwrap();
        let a2 = exec.ready("success");
        let retry_id = exec.retry(vec![a1, a2], 3);
        let result = exec.poll(retry_id).unwrap();
        assert_eq!(result, Poll::Ready("success".to_string()));
    }

    #[test]
    fn test_retry_all_fail() {
        let mut exec = FutureExecutor::new();
        let a1 = exec.ready("x");
        exec.fail(a1, "e1").unwrap();
        let a2 = exec.ready("y");
        exec.fail(a2, "e2").unwrap();
        let retry_id = exec.retry(vec![a1, a2], 2);
        let result = exec.poll(retry_id);
        assert!(matches!(result, Err(FutureError::RetriesExhausted { .. })));
    }

    #[test]
    fn test_poll_nonexistent() {
        let mut exec = FutureExecutor::new();
        let result = exec.poll(999);
        assert_eq!(result, Err(FutureError::NotFound(999)));
    }

    #[test]
    fn test_chained_map() {
        let mut exec = FutureExecutor::new();
        let base = exec.ready("a");
        let m1 = exec.map(base, "b");
        let m2 = exec.map(m1, "c");
        let result = exec.poll(m2).unwrap();
        assert_eq!(result, Poll::Ready("abc".to_string()));
    }

    #[test]
    fn test_poll_already_ready() {
        let mut exec = FutureExecutor::new();
        let id = exec.ready("done");
        exec.poll(id).unwrap();
        // Polling again should return the same result
        let result = exec.poll(id).unwrap();
        assert_eq!(result, Poll::Ready("done".to_string()));
    }

    #[test]
    fn test_future_state() {
        let mut exec = FutureExecutor::new();
        let id = exec.value("val", 10);
        assert_eq!(exec.state(id), Some(&FutureState::Pending));
        exec.tick(10);
        exec.poll(id).unwrap();
        assert_eq!(
            exec.state(id),
            Some(&FutureState::Ready("val".to_string()))
        );
    }

    #[test]
    fn test_pending_count() {
        let mut exec = FutureExecutor::new();
        exec.ready("a");
        exec.value("b", 10);
        exec.value("c", 20);
        // All are pending until polled (even ready futures start pending internally)
        assert_eq!(exec.pending_count(), 3);
    }

    #[test]
    fn test_fail_then_poll() {
        let mut exec = FutureExecutor::new();
        let id = exec.ready("x");
        exec.fail(id, "broken").unwrap();
        let result = exec.poll(id);
        assert!(matches!(result, Err(FutureError::Failed(_))));
    }
}
