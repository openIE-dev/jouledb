//! Web Worker Protocol — message-passing layer that replaces Comlink / workerize.
//!
//! Defines a serialization protocol for communicating with Web Workers.
//! No actual threads — just the message framing, dispatch, and timeout logic.

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Message types ──────────────────────────────────────────────

/// The kind of message flowing between proxy and host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageKind {
    Request,
    Response,
    Error,
    Event,
    Init,
    Terminate,
}

/// A message in the worker protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMessage {
    pub id: u64,
    pub kind: MessageKind,
    pub payload: Value,
    pub transfer_list: Vec<String>,
}

/// A typed request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub id: u64,
    pub method: String,
    pub args: Vec<Value>,
}

/// A typed response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResponse {
    pub id: u64,
    pub result: Result<Value, WorkerError>,
}

/// An error from the worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerError {
    pub code: String,
    pub message: String,
    pub data: Option<Value>,
}

// ── Pending request tracking ───────────────────────────────────

/// Metadata for a pending request.
#[derive(Debug, Clone)]
pub struct PendingRequest {
    pub method: String,
    pub sent_at: DateTime<Utc>,
    pub timeout_ms: Option<u64>,
}

// ── WorkerProxy ────────────────────────────────────────────────

/// Client-side proxy that queues outgoing requests and matches responses.
pub struct WorkerProxy {
    next_id: u64,
    pending: HashMap<u64, PendingRequest>,
    outbox: VecDeque<WorkerMessage>,
}

impl WorkerProxy {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            pending: HashMap::new(),
            outbox: VecDeque::new(),
        }
    }

    /// Queue a request and return its ID.
    pub fn call(&mut self, method: &str, args: Vec<Value>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let request = WorkerRequest {
            id,
            method: method.to_string(),
            args: args.clone(),
        };

        self.outbox.push_back(WorkerMessage {
            id,
            kind: MessageKind::Request,
            payload: serde_json::to_value(request).unwrap_or(Value::Null),
            transfer_list: Vec::new(),
        });

        self.pending.insert(
            id,
            PendingRequest {
                method: method.to_string(),
                sent_at: Utc::now(),
                timeout_ms: None,
            },
        );

        id
    }

    /// Queue a request with a timeout.
    pub fn call_with_timeout(&mut self, method: &str, args: Vec<Value>, timeout_ms: u64) -> u64 {
        let id = self.call(method, args);
        if let Some(pending) = self.pending.get_mut(&id) {
            pending.timeout_ms = Some(timeout_ms);
        }
        id
    }

    /// Take all pending outgoing messages.
    pub fn drain_outbox(&mut self) -> Vec<WorkerMessage> {
        self.outbox.drain(..).collect()
    }

    /// Match a response to a pending request.
    pub fn handle_response(&mut self, msg: WorkerMessage) -> Option<WorkerResponse> {
        if msg.kind != MessageKind::Response && msg.kind != MessageKind::Error {
            return None;
        }

        if self.pending.remove(&msg.id).is_some() {
            if msg.kind == MessageKind::Error {
                let err = serde_json::from_value::<WorkerError>(msg.payload.clone())
                    .unwrap_or(WorkerError {
                        code: "UNKNOWN".to_string(),
                        message: "Unknown error".to_string(),
                        data: Some(msg.payload),
                    });
                Some(WorkerResponse {
                    id: msg.id,
                    result: Err(err),
                })
            } else {
                Some(WorkerResponse {
                    id: msg.id,
                    result: Ok(msg.payload),
                })
            }
        } else {
            None
        }
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Cancel a pending request. Returns true if it was found.
    pub fn cancel(&mut self, id: u64) -> bool {
        self.pending.remove(&id).is_some()
    }

    /// Return IDs of requests that have timed out as of `now`.
    pub fn check_timeouts(&mut self, now: &DateTime<Utc>) -> Vec<u64> {
        let mut timed_out = Vec::new();
        for (id, pending) in &self.pending {
            if let Some(timeout_ms) = pending.timeout_ms {
                let elapsed = (*now - pending.sent_at).num_milliseconds();
                if elapsed >= timeout_ms as i64 {
                    timed_out.push(*id);
                }
            }
        }
        for id in &timed_out {
            self.pending.remove(id);
        }
        timed_out
    }
}

impl Default for WorkerProxy {
    fn default() -> Self {
        Self::new()
    }
}

// ── WorkerHost ─────────────────────────────────────────────────

/// Worker-side host that dispatches incoming requests to registered handlers.
pub struct WorkerHost {
    handlers: HashMap<String, Box<dyn Fn(Vec<Value>) -> Result<Value, WorkerError>>>,
    outbox: VecDeque<WorkerMessage>,
    next_event_id: u64,
}

impl WorkerHost {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            outbox: VecDeque::new(),
            next_event_id: u64::MAX / 2, // Use high IDs for events to avoid collisions.
        }
    }

    /// Register a handler for a method name.
    pub fn register(
        &mut self,
        method: &str,
        handler: impl Fn(Vec<Value>) -> Result<Value, WorkerError> + 'static,
    ) {
        self.handlers.insert(method.to_string(), Box::new(handler));
    }

    /// Handle an incoming message, returning any response messages.
    pub fn handle_message(&mut self, msg: WorkerMessage) -> Vec<WorkerMessage> {
        let mut responses = Vec::new();

        if msg.kind != MessageKind::Request {
            return responses;
        }

        let request: Option<WorkerRequest> = serde_json::from_value(msg.payload.clone()).ok();
        let Some(request) = request else {
            responses.push(WorkerMessage {
                id: msg.id,
                kind: MessageKind::Error,
                payload: serde_json::to_value(WorkerError {
                    code: "INVALID_REQUEST".to_string(),
                    message: "Could not parse request".to_string(),
                    data: None,
                })
                .unwrap_or(Value::Null),
                transfer_list: Vec::new(),
            });
            return responses;
        };

        if let Some(handler) = self.handlers.get(&request.method) {
            match handler(request.args) {
                Ok(result) => {
                    responses.push(WorkerMessage {
                        id: msg.id,
                        kind: MessageKind::Response,
                        payload: result,
                        transfer_list: Vec::new(),
                    });
                }
                Err(err) => {
                    responses.push(WorkerMessage {
                        id: msg.id,
                        kind: MessageKind::Error,
                        payload: serde_json::to_value(err).unwrap_or(Value::Null),
                        transfer_list: Vec::new(),
                    });
                }
            }
        } else {
            responses.push(WorkerMessage {
                id: msg.id,
                kind: MessageKind::Error,
                payload: serde_json::to_value(WorkerError {
                    code: "METHOD_NOT_FOUND".to_string(),
                    message: format!("No handler for method: {}", request.method),
                    data: None,
                })
                .unwrap_or(Value::Null),
                transfer_list: Vec::new(),
            });
        }

        // Also drain to outbox for consistency.
        for r in &responses {
            self.outbox.push_back(r.clone());
        }

        responses
    }

    /// Emit an event message.
    pub fn emit_event(&mut self, event: &str, data: Value) {
        let id = self.next_event_id;
        self.next_event_id += 1;
        self.outbox.push_back(WorkerMessage {
            id,
            kind: MessageKind::Event,
            payload: serde_json::json!({ "event": event, "data": data }),
            transfer_list: Vec::new(),
        });
    }

    /// Take all pending outgoing messages.
    pub fn drain_outbox(&mut self) -> Vec<WorkerMessage> {
        self.outbox.drain(..).collect()
    }
}

impl Default for WorkerHost {
    fn default() -> Self {
        Self::new()
    }
}

// ── WorkerPool ─────────────────────────────────────────────────

/// Round-robin pool of worker proxies.
pub struct WorkerPool {
    workers: Vec<WorkerProxy>,
    next_worker: usize,
}

impl WorkerPool {
    pub fn new(count: usize) -> Self {
        let workers = (0..count).map(|_| WorkerProxy::new()).collect();
        Self {
            workers,
            next_worker: 0,
        }
    }

    /// Dispatch a request to the next worker. Returns (worker_index, request_id).
    pub fn dispatch(&mut self, method: &str, args: Vec<Value>) -> (usize, u64) {
        let idx = self.next_worker;
        self.next_worker = (self.next_worker + 1) % self.workers.len();
        let id = self.workers[idx].call(method, args);
        (idx, id)
    }

    /// Drain outboxes from all workers.
    pub fn drain_all(&mut self) -> Vec<(usize, Vec<WorkerMessage>)> {
        self.workers
            .iter_mut()
            .enumerate()
            .map(|(i, w)| (i, w.drain_outbox()))
            .collect()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_call_queues_message() {
        let mut proxy = WorkerProxy::new();
        let id = proxy.call("add", vec![serde_json::json!(1), serde_json::json!(2)]);
        assert_eq!(id, 1);
        assert_eq!(proxy.pending_count(), 1);
        let msgs = proxy.drain_outbox();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].kind, MessageKind::Request);
    }

    #[test]
    fn drain_outbox_empties() {
        let mut proxy = WorkerProxy::new();
        proxy.call("test", vec![]);
        let msgs = proxy.drain_outbox();
        assert_eq!(msgs.len(), 1);
        let msgs2 = proxy.drain_outbox();
        assert!(msgs2.is_empty());
    }

    #[test]
    fn handle_response_matches_id() {
        let mut proxy = WorkerProxy::new();
        let id = proxy.call("test", vec![]);
        let _ = proxy.drain_outbox();

        let response = WorkerMessage {
            id,
            kind: MessageKind::Response,
            payload: serde_json::json!(42),
            transfer_list: Vec::new(),
        };

        let result = proxy.handle_response(response);
        assert!(result.is_some());
        let resp = result.unwrap();
        assert_eq!(resp.id, id);
        assert!(resp.result.is_ok());
        assert_eq!(resp.result.unwrap(), serde_json::json!(42));
        assert_eq!(proxy.pending_count(), 0);
    }

    #[test]
    fn cancel_removes_pending() {
        let mut proxy = WorkerProxy::new();
        let id = proxy.call("test", vec![]);
        assert_eq!(proxy.pending_count(), 1);
        assert!(proxy.cancel(id));
        assert_eq!(proxy.pending_count(), 0);
        assert!(!proxy.cancel(id));
    }

    #[test]
    fn timeout_detection() {
        let mut proxy = WorkerProxy::new();
        let _id = proxy.call_with_timeout("slow", vec![], 100);
        let _ = proxy.drain_outbox();

        // Simulate time passing: check with a future timestamp.
        let future = Utc::now() + chrono::Duration::milliseconds(200);
        let timed_out = proxy.check_timeouts(&future);
        assert_eq!(timed_out.len(), 1);
        assert_eq!(proxy.pending_count(), 0);
    }

    #[test]
    fn host_register_and_handle() {
        let mut host = WorkerHost::new();
        host.register("add", |args| {
            let a = args[0].as_i64().unwrap_or(0);
            let b = args[1].as_i64().unwrap_or(0);
            Ok(serde_json::json!(a + b))
        });

        let request = WorkerRequest {
            id: 1,
            method: "add".to_string(),
            args: vec![serde_json::json!(3), serde_json::json!(4)],
        };

        let msg = WorkerMessage {
            id: 1,
            kind: MessageKind::Request,
            payload: serde_json::to_value(request).unwrap(),
            transfer_list: Vec::new(),
        };

        let responses = host.handle_message(msg);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].kind, MessageKind::Response);
        assert_eq!(responses[0].payload, serde_json::json!(7));
    }

    #[test]
    fn host_unknown_method_returns_error() {
        let mut host = WorkerHost::new();
        let request = WorkerRequest {
            id: 1,
            method: "nonexistent".to_string(),
            args: vec![],
        };

        let msg = WorkerMessage {
            id: 1,
            kind: MessageKind::Request,
            payload: serde_json::to_value(request).unwrap(),
            transfer_list: Vec::new(),
        };

        let responses = host.handle_message(msg);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].kind, MessageKind::Error);
    }

    #[test]
    fn pool_round_robin() {
        let mut pool = WorkerPool::new(3);
        let (idx0, _) = pool.dispatch("a", vec![]);
        let (idx1, _) = pool.dispatch("b", vec![]);
        let (idx2, _) = pool.dispatch("c", vec![]);
        let (idx3, _) = pool.dispatch("d", vec![]);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(idx3, 0);
    }

    #[test]
    fn pool_dispatch_returns_ids() {
        let mut pool = WorkerPool::new(2);
        let (w0, id0) = pool.dispatch("test", vec![]);
        let (w1, id1) = pool.dispatch("test", vec![]);
        assert_eq!(w0, 0);
        assert_eq!(w1, 1);
        assert_eq!(id0, 1);
        assert_eq!(id1, 1);
    }

    #[test]
    fn event_emission() {
        let mut host = WorkerHost::new();
        host.emit_event("progress", serde_json::json!({"percent": 50}));
        let msgs = host.drain_outbox();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].kind, MessageKind::Event);
    }

    #[test]
    fn proxy_pending_count() {
        let mut proxy = WorkerProxy::new();
        assert_eq!(proxy.pending_count(), 0);
        proxy.call("a", vec![]);
        proxy.call("b", vec![]);
        assert_eq!(proxy.pending_count(), 2);
        proxy.cancel(1);
        assert_eq!(proxy.pending_count(), 1);
    }
}
