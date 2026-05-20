//! AbortController / AbortSignal — cooperative cancellation.
//!
//! Headless implementation of the AbortController Web API pattern.

use thiserror::Error;

// ── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug, Error, Clone)]
#[error("operation aborted: {reason}")]
pub struct AbortError {
    pub reason: String,
}

// ── AbortSignal ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AbortSignal {
    aborted: bool,
    reason: Option<String>,
    listeners: Vec<u64>,
}

impl AbortSignal {
    fn new() -> Self {
        Self {
            aborted: false,
            reason: None,
            listeners: Vec::new(),
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.aborted
    }

    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// Register a handler ID to be notified on abort.
    pub fn on_abort(&mut self, handler_id: u64) {
        self.listeners.push(handler_id);
    }

    /// Return the list of notified handler IDs (useful after aborting).
    pub fn notified_listeners(&self) -> &[u64] {
        &self.listeners
    }

    /// Throw an error if already aborted.
    pub fn throw_if_aborted(&self) -> Result<(), AbortError> {
        if self.aborted {
            Err(AbortError {
                reason: self.reason.clone().unwrap_or_else(|| "AbortError".into()),
            })
        } else {
            Ok(())
        }
    }

    /// Create a pre-aborted signal (simulates `AbortSignal.timeout()`).
    pub fn timeout(ms: u64) -> AbortSignal {
        AbortSignal {
            aborted: true,
            reason: Some(format!("TimeoutError: {}ms", ms)),
            listeners: Vec::new(),
        }
    }

    /// Create a signal that is aborted if any of the given signals is aborted.
    pub fn any(signals: &[&AbortSignal]) -> AbortSignal {
        for s in signals {
            if s.aborted {
                return AbortSignal {
                    aborted: true,
                    reason: s.reason.clone(),
                    listeners: Vec::new(),
                };
            }
        }
        AbortSignal::new()
    }
}

// ── AbortController ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AbortController {
    signal: AbortSignal,
}

impl AbortController {
    pub fn new() -> Self {
        Self {
            signal: AbortSignal::new(),
        }
    }

    pub fn abort(&mut self, reason: Option<&str>) {
        self.signal.aborted = true;
        self.signal.reason = reason.map(|r| r.to_string());
    }

    pub fn signal(&self) -> &AbortSignal {
        &self.signal
    }

    pub fn signal_mut(&mut self) -> &mut AbortSignal {
        &mut self.signal
    }
}

impl Default for AbortController {
    fn default() -> Self {
        Self::new()
    }
}

// ── LinkedAbortController ───────────────────────────────────────────────────

/// A controller that chains to a parent signal — if the parent is aborted,
/// this controller is also considered aborted.
#[derive(Debug, Clone)]
pub struct LinkedAbortController {
    controller: AbortController,
    parent_aborted: bool,
    parent_reason: Option<String>,
}

impl LinkedAbortController {
    pub fn new(parent: &AbortSignal) -> Self {
        Self {
            controller: AbortController::new(),
            parent_aborted: parent.aborted,
            parent_reason: parent.reason.clone(),
        }
    }

    pub fn abort(&mut self, reason: Option<&str>) {
        self.controller.abort(reason);
    }

    pub fn is_aborted(&self) -> bool {
        self.parent_aborted || self.controller.signal().is_aborted()
    }

    pub fn reason(&self) -> Option<&str> {
        if self.controller.signal().is_aborted() {
            self.controller.signal().reason()
        } else if self.parent_aborted {
            self.parent_reason.as_deref()
        } else {
            None
        }
    }

    pub fn signal(&self) -> &AbortSignal {
        self.controller.signal()
    }

    /// Sync with a possibly-updated parent signal.
    pub fn sync_parent(&mut self, parent: &AbortSignal) {
        self.parent_aborted = parent.aborted;
        self.parent_reason = parent.reason.clone();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abort_sets_flag() {
        let mut ctrl = AbortController::new();
        assert!(!ctrl.signal().is_aborted());
        ctrl.abort(None);
        assert!(ctrl.signal().is_aborted());
    }

    #[test]
    fn reason_propagated() {
        let mut ctrl = AbortController::new();
        ctrl.abort(Some("user cancelled"));
        assert_eq!(ctrl.signal().reason(), Some("user cancelled"));
    }

    #[test]
    fn throw_if_aborted_ok() {
        let ctrl = AbortController::new();
        assert!(ctrl.signal().throw_if_aborted().is_ok());
    }

    #[test]
    fn throw_if_aborted_err() {
        let mut ctrl = AbortController::new();
        ctrl.abort(Some("cancelled"));
        let err = ctrl.signal().throw_if_aborted().unwrap_err();
        assert!(err.reason.contains("cancelled"));
    }

    #[test]
    fn any_combines() {
        let s1 = AbortSignal::new();
        let mut ctrl2 = AbortController::new();
        ctrl2.abort(Some("boom"));
        let s2 = ctrl2.signal();

        let combined = AbortSignal::any(&[&s1, s2]);
        assert!(combined.is_aborted());
        assert_eq!(combined.reason(), Some("boom"));
    }

    #[test]
    fn any_none_aborted() {
        let s1 = AbortSignal::new();
        let s2 = AbortSignal::new();
        let combined = AbortSignal::any(&[&s1, &s2]);
        assert!(!combined.is_aborted());
    }

    #[test]
    fn timeout_creates_aborted() {
        let sig = AbortSignal::timeout(5000);
        assert!(sig.is_aborted());
        assert!(sig.reason().unwrap().contains("5000"));
    }

    #[test]
    fn listener_registered() {
        let mut ctrl = AbortController::new();
        ctrl.signal_mut().on_abort(42);
        ctrl.signal_mut().on_abort(99);
        assert_eq!(ctrl.signal().notified_listeners(), &[42, 99]);
    }

    #[test]
    fn linked_controller_parent_abort() {
        let mut parent = AbortController::new();
        parent.abort(Some("parent done"));
        let linked = LinkedAbortController::new(parent.signal());
        assert!(linked.is_aborted());
        assert_eq!(linked.reason(), Some("parent done"));
    }

    #[test]
    fn linked_controller_own_abort() {
        let parent = AbortController::new();
        let mut linked = LinkedAbortController::new(parent.signal());
        assert!(!linked.is_aborted());
        linked.abort(Some("child done"));
        assert!(linked.is_aborted());
        assert_eq!(linked.reason(), Some("child done"));
    }

    #[test]
    fn linked_controller_sync_parent() {
        let mut parent = AbortController::new();
        let mut linked = LinkedAbortController::new(parent.signal());
        assert!(!linked.is_aborted());

        parent.abort(Some("late abort"));
        linked.sync_parent(parent.signal());
        assert!(linked.is_aborted());
    }
}
