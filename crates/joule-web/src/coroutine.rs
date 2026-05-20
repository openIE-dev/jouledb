//! Stackless coroutine simulation — yield/resume, state machine, generators.
//!
//! Replaces JavaScript generators and async iterators with a pure-Rust
//! stackless coroutine model. Supports yield/resume protocol, bidirectional
//! value passing, generator-to-iterator adapters, coroutine composition,
//! and async-like state machine simulation.

use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Coroutine domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoroutineError {
    /// Coroutine not found.
    NotFound(u64),
    /// Coroutine is already completed.
    Completed(u64),
    /// Coroutine is cancelled.
    Cancelled(u64),
    /// Coroutine panicked/failed.
    Failed { id: u64, reason: String },
    /// Invalid state transition.
    InvalidTransition { id: u64, from: &'static str, to: &'static str },
}

impl std::fmt::Display for CoroutineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "coroutine not found: {id}"),
            Self::Completed(id) => write!(f, "coroutine {id} already completed"),
            Self::Cancelled(id) => write!(f, "coroutine {id} cancelled"),
            Self::Failed { id, reason } => write!(f, "coroutine {id} failed: {reason}"),
            Self::InvalidTransition { id, from, to } => {
                write!(f, "coroutine {id}: invalid transition {from} -> {to}")
            }
        }
    }
}

impl std::error::Error for CoroutineError {}

// ── Coroutine State ─────────────────────────────────────────────

/// State of a coroutine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoroutineState {
    /// Created, not yet started.
    Created,
    /// Suspended at a yield point.
    Suspended,
    /// Currently executing.
    Running,
    /// Completed normally.
    Completed,
    /// Cancelled externally.
    Cancelled,
    /// Failed with an error.
    Failed,
}

impl CoroutineState {
    fn name(&self) -> &'static str {
        match self {
            Self::Created => "Created",
            Self::Suspended => "Suspended",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Cancelled => "Cancelled",
            Self::Failed => "Failed",
        }
    }
}

// ── Yield Result ────────────────────────────────────────────────

/// Result of resuming a coroutine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YieldResult {
    /// Coroutine yielded a value and is suspended.
    Yielded(String),
    /// Coroutine completed with a final value.
    Completed(String),
}

impl YieldResult {
    pub fn value(&self) -> &str {
        match self {
            Self::Yielded(v) | Self::Completed(v) => v,
        }
    }

    pub fn is_yielded(&self) -> bool {
        matches!(self, Self::Yielded(_))
    }

    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed(_))
    }
}

// ── Yield Point ─────────────────────────────────────────────────

/// A single yield point in a coroutine's program.
#[derive(Debug, Clone)]
struct YieldPoint {
    /// Value to yield.
    value: String,
    /// Whether this is the final yield (coroutine completes after).
    is_final: bool,
}

// ── Coroutine ───────────────────────────────────────────────────

/// A stackless coroutine with pre-defined yield points.
#[derive(Debug, Clone)]
pub struct Coroutine {
    pub id: u64,
    pub name: String,
    pub state: CoroutineState,
    yield_points: Vec<YieldPoint>,
    /// Index of the next yield point to execute.
    program_counter: usize,
    /// Value received from the last resume call.
    received_value: Option<String>,
    /// All values yielded so far.
    pub yielded_values: Vec<String>,
    /// All values received via resume so far.
    pub received_values: Vec<String>,
    /// Resume count.
    pub resume_count: u64,
}

impl Coroutine {
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            state: CoroutineState::Created,
            yield_points: Vec::new(),
            program_counter: 0,
            received_value: None,
            yielded_values: Vec::new(),
            received_values: Vec::new(),
            resume_count: 0,
        }
    }

    /// Add a yield point.
    pub fn yield_value(mut self, value: impl Into<String>) -> Self {
        self.yield_points.push(YieldPoint {
            value: value.into(),
            is_final: false,
        });
        self
    }

    /// Add a final return value.
    pub fn return_value(mut self, value: impl Into<String>) -> Self {
        self.yield_points.push(YieldPoint {
            value: value.into(),
            is_final: true,
        });
        self
    }

    /// Whether the coroutine can be resumed.
    pub fn is_resumable(&self) -> bool {
        matches!(
            self.state,
            CoroutineState::Created | CoroutineState::Suspended
        )
    }

    /// Last received value from resume.
    pub fn last_received(&self) -> Option<&str> {
        self.received_value.as_deref()
    }
}

// ── Coroutine Manager ───────────────────────────────────────────

/// Manages a set of coroutines.
pub struct CoroutineManager {
    coroutines: HashMap<u64, Coroutine>,
    next_id: u64,
}

impl CoroutineManager {
    pub fn new() -> Self {
        Self {
            coroutines: HashMap::new(),
            next_id: 1,
        }
    }

    /// Register a coroutine. Returns its ID.
    pub fn register(&mut self, mut coro: Coroutine) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        coro.id = id;
        self.coroutines.insert(id, coro);
        id
    }

    /// Resume a coroutine without sending a value.
    pub fn resume(&mut self, id: u64) -> Result<YieldResult, CoroutineError> {
        self.resume_with(id, None)
    }

    /// Resume a coroutine, optionally sending a value.
    pub fn resume_with(
        &mut self,
        id: u64,
        send_value: Option<String>,
    ) -> Result<YieldResult, CoroutineError> {
        let coro = self
            .coroutines
            .get_mut(&id)
            .ok_or(CoroutineError::NotFound(id))?;

        match coro.state {
            CoroutineState::Completed => return Err(CoroutineError::Completed(id)),
            CoroutineState::Cancelled => return Err(CoroutineError::Cancelled(id)),
            CoroutineState::Failed => {
                return Err(CoroutineError::Failed {
                    id,
                    reason: "previously failed".to_string(),
                })
            }
            CoroutineState::Running => {
                return Err(CoroutineError::InvalidTransition {
                    id,
                    from: "Running",
                    to: "Running",
                })
            }
            CoroutineState::Created | CoroutineState::Suspended => {}
        }

        // Store the received value
        if let Some(val) = &send_value {
            coro.received_values.push(val.clone());
        }
        coro.received_value = send_value;
        coro.resume_count += 1;
        coro.state = CoroutineState::Running;

        if coro.program_counter >= coro.yield_points.len() {
            // No more yield points — complete with empty value
            coro.state = CoroutineState::Completed;
            return Ok(YieldResult::Completed(String::new()));
        }

        let yp = coro.yield_points[coro.program_counter].clone();
        coro.program_counter += 1;

        if yp.is_final {
            coro.state = CoroutineState::Completed;
            coro.yielded_values.push(yp.value.clone());
            Ok(YieldResult::Completed(yp.value))
        } else {
            coro.state = CoroutineState::Suspended;
            coro.yielded_values.push(yp.value.clone());
            Ok(YieldResult::Yielded(yp.value))
        }
    }

    /// Cancel a coroutine.
    pub fn cancel(&mut self, id: u64) -> Result<(), CoroutineError> {
        let coro = self
            .coroutines
            .get_mut(&id)
            .ok_or(CoroutineError::NotFound(id))?;
        if coro.state == CoroutineState::Completed {
            return Err(CoroutineError::InvalidTransition {
                id,
                from: coro.state.name(),
                to: "Cancelled",
            });
        }
        coro.state = CoroutineState::Cancelled;
        Ok(())
    }

    /// Fail a coroutine with a reason.
    pub fn fail(&mut self, id: u64, reason: impl Into<String>) -> Result<(), CoroutineError> {
        let coro = self
            .coroutines
            .get_mut(&id)
            .ok_or(CoroutineError::NotFound(id))?;
        if coro.state == CoroutineState::Completed {
            return Err(CoroutineError::InvalidTransition {
                id,
                from: coro.state.name(),
                to: "Failed",
            });
        }
        coro.state = CoroutineState::Failed;
        Ok(())
    }

    /// Collect all values from a coroutine by exhausting it. Returns all yielded values.
    pub fn collect_all(&mut self, id: u64) -> Result<Vec<String>, CoroutineError> {
        let mut values = Vec::new();
        loop {
            match self.resume(id)? {
                YieldResult::Yielded(v) => values.push(v),
                YieldResult::Completed(v) => {
                    if !v.is_empty() {
                        values.push(v);
                    }
                    break;
                }
            }
        }
        Ok(values)
    }

    /// Create a composed coroutine that runs two coroutines in sequence.
    /// The first coroutine runs to completion, then the second.
    pub fn compose(
        &mut self,
        first_id: u64,
        second_id: u64,
    ) -> Result<Vec<String>, CoroutineError> {
        let mut results = Vec::new();
        // Exhaust first
        loop {
            match self.resume(first_id)? {
                YieldResult::Yielded(v) => results.push(v),
                YieldResult::Completed(v) => {
                    if !v.is_empty() {
                        results.push(v);
                    }
                    break;
                }
            }
        }
        // Exhaust second
        loop {
            match self.resume(second_id)? {
                YieldResult::Yielded(v) => results.push(v),
                YieldResult::Completed(v) => {
                    if !v.is_empty() {
                        results.push(v);
                    }
                    break;
                }
            }
        }
        Ok(results)
    }

    /// Get a reference to a coroutine.
    pub fn get(&self, id: u64) -> Option<&Coroutine> {
        self.coroutines.get(&id)
    }

    /// Number of managed coroutines.
    pub fn count(&self) -> usize {
        self.coroutines.len()
    }

    /// Number of resumable coroutines.
    pub fn resumable_count(&self) -> usize {
        self.coroutines
            .values()
            .filter(|c| c.is_resumable())
            .count()
    }
}

impl Default for CoroutineManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_counter(n: u32) -> Coroutine {
        let mut coro = Coroutine::new(0, "counter");
        for i in 0..n {
            coro = coro.yield_value(format!("{i}"));
        }
        coro.return_value("done".to_string())
    }

    #[test]
    fn test_basic_yield_resume() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(
            Coroutine::new(0, "simple")
                .yield_value("a")
                .yield_value("b")
                .return_value("c"),
        );

        let r1 = mgr.resume(id).unwrap();
        assert_eq!(r1, YieldResult::Yielded("a".to_string()));

        let r2 = mgr.resume(id).unwrap();
        assert_eq!(r2, YieldResult::Yielded("b".to_string()));

        let r3 = mgr.resume(id).unwrap();
        assert_eq!(r3, YieldResult::Completed("c".to_string()));
    }

    #[test]
    fn test_resume_completed_error() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(Coroutine::new(0, "once").return_value("done"));
        mgr.resume(id).unwrap();
        let err = mgr.resume(id).unwrap_err();
        assert_eq!(err, CoroutineError::Completed(id));
    }

    #[test]
    fn test_bidirectional_values() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(
            Coroutine::new(0, "echo")
                .yield_value("request1")
                .yield_value("request2")
                .return_value("final"),
        );

        mgr.resume(id).unwrap();
        mgr.resume_with(id, Some("response1".to_string())).unwrap();
        mgr.resume_with(id, Some("response2".to_string())).unwrap();

        let coro = mgr.get(id).unwrap();
        assert_eq!(coro.received_values, vec!["response1", "response2"]);
    }

    #[test]
    fn test_cancel_coroutine() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(make_counter(5));
        mgr.resume(id).unwrap();
        mgr.cancel(id).unwrap();
        let err = mgr.resume(id).unwrap_err();
        assert_eq!(err, CoroutineError::Cancelled(id));
    }

    #[test]
    fn test_fail_coroutine() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(make_counter(3));
        mgr.resume(id).unwrap();
        mgr.fail(id, "oops").unwrap();
        let err = mgr.resume(id).unwrap_err();
        assert!(matches!(err, CoroutineError::Failed { .. }));
    }

    #[test]
    fn test_collect_all() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(make_counter(3));
        let values = mgr.collect_all(id).unwrap();
        assert_eq!(values, vec!["0", "1", "2", "done"]);
    }

    #[test]
    fn test_coroutine_state_transitions() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(
            Coroutine::new(0, "states")
                .yield_value("a")
                .return_value("b"),
        );

        assert_eq!(mgr.get(id).unwrap().state, CoroutineState::Created);
        mgr.resume(id).unwrap();
        assert_eq!(mgr.get(id).unwrap().state, CoroutineState::Suspended);
        mgr.resume(id).unwrap();
        assert_eq!(mgr.get(id).unwrap().state, CoroutineState::Completed);
    }

    #[test]
    fn test_resume_count() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(make_counter(3));
        mgr.resume(id).unwrap();
        mgr.resume(id).unwrap();
        assert_eq!(mgr.get(id).unwrap().resume_count, 2);
    }

    #[test]
    fn test_yielded_values_tracked() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(
            Coroutine::new(0, "track")
                .yield_value("x")
                .yield_value("y")
                .return_value("z"),
        );
        mgr.collect_all(id).unwrap();
        assert_eq!(mgr.get(id).unwrap().yielded_values, vec!["x", "y", "z"]);
    }

    #[test]
    fn test_compose_two_coroutines() {
        let mut mgr = CoroutineManager::new();
        let c1 = mgr.register(
            Coroutine::new(0, "first")
                .yield_value("a")
                .return_value("b"),
        );
        let c2 = mgr.register(
            Coroutine::new(0, "second")
                .yield_value("c")
                .return_value("d"),
        );
        let results = mgr.compose(c1, c2).unwrap();
        assert_eq!(results, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_empty_coroutine() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(Coroutine::new(0, "empty"));
        let result = mgr.resume(id).unwrap();
        assert_eq!(result, YieldResult::Completed(String::new()));
    }

    #[test]
    fn test_single_return_no_yields() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(Coroutine::new(0, "ret").return_value("42"));
        let result = mgr.resume(id).unwrap();
        assert_eq!(result, YieldResult::Completed("42".to_string()));
    }

    #[test]
    fn test_is_resumable() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(Coroutine::new(0, "r").yield_value("x").return_value("y"));
        assert!(mgr.get(id).unwrap().is_resumable());
        mgr.resume(id).unwrap();
        assert!(mgr.get(id).unwrap().is_resumable());
        mgr.resume(id).unwrap();
        assert!(!mgr.get(id).unwrap().is_resumable());
    }

    #[test]
    fn test_not_found() {
        let mut mgr = CoroutineManager::new();
        let err = mgr.resume(999).unwrap_err();
        assert_eq!(err, CoroutineError::NotFound(999));
    }

    #[test]
    fn test_count_and_resumable_count() {
        let mut mgr = CoroutineManager::new();
        let c1 = mgr.register(Coroutine::new(0, "a").return_value("done"));
        mgr.register(Coroutine::new(0, "b").return_value("done"));
        assert_eq!(mgr.count(), 2);
        assert_eq!(mgr.resumable_count(), 2);
        mgr.resume(c1).unwrap();
        assert_eq!(mgr.resumable_count(), 1);
    }

    #[test]
    fn test_cancel_completed_fails() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(Coroutine::new(0, "done").return_value("x"));
        mgr.resume(id).unwrap();
        let err = mgr.cancel(id).unwrap_err();
        assert!(matches!(err, CoroutineError::InvalidTransition { .. }));
    }

    #[test]
    fn test_last_received() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(
            Coroutine::new(0, "recv")
                .yield_value("q1")
                .return_value("q2"),
        );
        mgr.resume(id).unwrap();
        assert!(mgr.get(id).unwrap().last_received().is_none());
        mgr.resume_with(id, Some("answer".to_string())).unwrap();
        assert_eq!(mgr.get(id).unwrap().last_received(), Some("answer"));
    }

    #[test]
    fn test_many_yield_points() {
        let mut mgr = CoroutineManager::new();
        let id = mgr.register(make_counter(10));
        let values = mgr.collect_all(id).unwrap();
        assert_eq!(values.len(), 11); // 10 yields + 1 return
        assert_eq!(values[0], "0");
        assert_eq!(values[9], "9");
        assert_eq!(values[10], "done");
    }
}
