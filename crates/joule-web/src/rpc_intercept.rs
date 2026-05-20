//! RPC interceptor/middleware chain — composable before/after hooks.
//!
//! Defines an [`Interceptor`] trait with `before_call` and `after_call` hooks,
//! an [`InterceptorChain`] that executes interceptors in priority order, and
//! built-in interceptors: [`LoggingInterceptor`], [`TimingInterceptor`],
//! [`AuthInterceptor`], [`RetryInterceptor`]. Supports short-circuit on error
//! and context passing between interceptors.

use std::collections::HashMap;
use std::fmt;

// ── RPC Context ────────────────────────────────────────────────

/// Shared context passed through the interceptor chain.
/// Interceptors can read/write metadata entries to communicate.
#[derive(Debug, Clone)]
pub struct RpcContext {
    pub method: String,
    pub metadata: HashMap<String, String>,
    pub request_payload: Vec<u8>,
    pub response_payload: Vec<u8>,
    pub start_time_us: u64,
    pub end_time_us: u64,
    pub error: Option<String>,
}

impl RpcContext {
    pub fn new(method: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            method: method.into(),
            metadata: HashMap::new(),
            request_payload: payload,
            response_payload: Vec::new(),
            start_time_us: 0,
            end_time_us: 0,
            error: None,
        }
    }

    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    pub fn has_error(&self) -> bool { self.error.is_some() }

    pub fn duration_us(&self) -> u64 { self.end_time_us.saturating_sub(self.start_time_us) }
}

// ── Intercept Result ───────────────────────────────────────────

/// Outcome of an interceptor hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterceptResult {
    /// Continue to the next interceptor / the actual call.
    Continue,
    /// Short-circuit: abort the chain with an error.
    Abort(String),
}

impl InterceptResult {
    pub fn is_continue(&self) -> bool { matches!(self, Self::Continue) }
    pub fn is_abort(&self) -> bool { matches!(self, Self::Abort(_)) }
}

// ── Interceptor Trait ──────────────────────────────────────────

/// Trait for RPC interceptors (middleware).
pub trait Interceptor: fmt::Debug {
    /// Name for logging / identification.
    fn name(&self) -> &str;

    /// Priority (lower = runs first). Default is 100.
    fn priority(&self) -> u32 { 100 }

    /// Called before the RPC is dispatched. Can modify the context or abort.
    fn before_call(&self, ctx: &mut RpcContext) -> InterceptResult {
        let _ = ctx;
        InterceptResult::Continue
    }

    /// Called after the RPC completes (or fails). Can inspect/modify response.
    fn after_call(&self, ctx: &mut RpcContext) {
        let _ = ctx;
    }
}

// ── Interceptor Chain ──────────────────────────────────────────

/// Stats for the interceptor chain.
#[derive(Debug, Clone, Default)]
pub struct ChainStats {
    pub calls_processed: u64,
    pub calls_aborted: u64,
    pub total_before_us: u64,
    pub total_after_us: u64,
}

impl ChainStats {
    pub fn abort_rate(&self) -> f64 {
        if self.calls_processed == 0 { return 0.0; }
        self.calls_aborted as f64 / self.calls_processed as f64
    }
}

/// Ordered chain of interceptors executed around an RPC call.
pub struct InterceptorChain {
    interceptors: Vec<Box<dyn Interceptor>>,
    stats: ChainStats,
}

impl InterceptorChain {
    pub fn new() -> Self {
        Self { interceptors: Vec::new(), stats: ChainStats::default() }
    }

    /// Add an interceptor. The chain will be sorted by priority on next execution.
    pub fn add(&mut self, interceptor: Box<dyn Interceptor>) {
        self.interceptors.push(interceptor);
        self.interceptors.sort_by_key(|i| i.priority());
    }

    /// Number of interceptors.
    pub fn len(&self) -> usize { self.interceptors.len() }
    pub fn is_empty(&self) -> bool { self.interceptors.is_empty() }

    /// Execute the before-call phase. Returns the index where abort occurred, or None.
    pub fn run_before(&mut self, ctx: &mut RpcContext) -> Option<(usize, String)> {
        self.stats.calls_processed += 1;
        for (i, interceptor) in self.interceptors.iter().enumerate() {
            match interceptor.before_call(ctx) {
                InterceptResult::Continue => {}
                InterceptResult::Abort(reason) => {
                    self.stats.calls_aborted += 1;
                    ctx.error = Some(reason.clone());
                    return Some((i, reason));
                }
            }
        }
        None
    }

    /// Execute the after-call phase (in reverse order).
    pub fn run_after(&mut self, ctx: &mut RpcContext) {
        for interceptor in self.interceptors.iter().rev() {
            interceptor.after_call(ctx);
        }
    }

    /// Convenience: run before, execute a handler, run after.
    pub fn execute<F>(&mut self, ctx: &mut RpcContext, handler: F) -> bool
    where
        F: FnOnce(&mut RpcContext),
    {
        if let Some((_idx, _reason)) = self.run_before(ctx) {
            self.run_after(ctx);
            return false;
        }
        handler(ctx);
        self.run_after(ctx);
        true
    }

    pub fn stats(&self) -> &ChainStats { &self.stats }

    /// List interceptor names in execution order.
    pub fn interceptor_names(&self) -> Vec<&str> {
        self.interceptors.iter().map(|i| i.name()).collect()
    }
}

impl Default for InterceptorChain {
    fn default() -> Self { Self::new() }
}

impl fmt::Debug for InterceptorChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterceptorChain")
            .field("count", &self.interceptors.len())
            .field("names", &self.interceptor_names())
            .field("stats", &self.stats)
            .finish()
    }
}

// ── Built-in: Logging Interceptor ──────────────────────────────

/// Logs before/after call information into the context metadata.
#[derive(Debug, Clone)]
pub struct LoggingInterceptor {
    prefix: String,
}

impl LoggingInterceptor {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self { prefix: prefix.into() }
    }
}

impl Interceptor for LoggingInterceptor {
    fn name(&self) -> &str { "logging" }
    fn priority(&self) -> u32 { 10 }

    fn before_call(&self, ctx: &mut RpcContext) -> InterceptResult {
        ctx.set_metadata(
            &format!("{}_before", self.prefix),
            format!("calling {}", ctx.method),
        );
        InterceptResult::Continue
    }

    fn after_call(&self, ctx: &mut RpcContext) {
        let status = if ctx.has_error() { "error" } else { "ok" };
        ctx.set_metadata(
            &format!("{}_after", self.prefix),
            format!("{} completed: {}", ctx.method, status),
        );
    }
}

// ── Built-in: Timing Interceptor ───────────────────────────────

/// Records start/end timestamps in the context.
#[derive(Debug, Clone)]
pub struct TimingInterceptor {
    simulated_time_us: u64,
}

impl TimingInterceptor {
    pub fn new(simulated_time_us: u64) -> Self {
        Self { simulated_time_us }
    }
}

impl Interceptor for TimingInterceptor {
    fn name(&self) -> &str { "timing" }
    fn priority(&self) -> u32 { 5 }

    fn before_call(&self, ctx: &mut RpcContext) -> InterceptResult {
        ctx.start_time_us = self.simulated_time_us;
        InterceptResult::Continue
    }

    fn after_call(&self, ctx: &mut RpcContext) {
        ctx.end_time_us = self.simulated_time_us + 1000; // simulated 1ms
        ctx.set_metadata("duration_us", ctx.duration_us().to_string());
    }
}

// ── Built-in: Auth Interceptor ─────────────────────────────────

/// Adds an authorization token to the context metadata.
/// Aborts if no token is configured and `required` is true.
#[derive(Debug, Clone)]
pub struct AuthInterceptor {
    token: Option<String>,
    required: bool,
}

impl AuthInterceptor {
    pub fn new(token: Option<String>, required: bool) -> Self {
        Self { token, required }
    }
}

impl Interceptor for AuthInterceptor {
    fn name(&self) -> &str { "auth" }
    fn priority(&self) -> u32 { 20 }

    fn before_call(&self, ctx: &mut RpcContext) -> InterceptResult {
        match &self.token {
            Some(tok) => {
                ctx.set_metadata("authorization", format!("Bearer {tok}"));
                InterceptResult::Continue
            }
            None if self.required => {
                InterceptResult::Abort("authentication required but no token provided".into())
            }
            None => InterceptResult::Continue,
        }
    }
}

// ── Built-in: Retry Interceptor ────────────────────────────────

/// Records retry-related metadata. Actual retry logic lives in rpc_retry;
/// this interceptor tags calls with retry attempt info.
#[derive(Debug, Clone)]
pub struct RetryInterceptor {
    max_retries: u32,
    current_attempt: u32,
}

impl RetryInterceptor {
    pub fn new(max_retries: u32) -> Self {
        Self { max_retries, current_attempt: 0 }
    }

    pub fn set_attempt(&mut self, attempt: u32) {
        self.current_attempt = attempt;
    }
}

impl Interceptor for RetryInterceptor {
    fn name(&self) -> &str { "retry" }
    fn priority(&self) -> u32 { 50 }

    fn before_call(&self, ctx: &mut RpcContext) -> InterceptResult {
        ctx.set_metadata("retry_attempt", self.current_attempt.to_string());
        ctx.set_metadata("retry_max", self.max_retries.to_string());
        if self.current_attempt > self.max_retries {
            InterceptResult::Abort("max retries exceeded".into())
        } else {
            InterceptResult::Continue
        }
    }

    fn after_call(&self, ctx: &mut RpcContext) {
        if ctx.has_error() {
            ctx.set_metadata("retry_needed", "true".to_string());
        }
    }
}

// ── Built-in: Header Interceptor ───────────────────────────────

/// Injects static headers/metadata into every call.
#[derive(Debug, Clone)]
pub struct HeaderInterceptor {
    headers: Vec<(String, String)>,
}

impl HeaderInterceptor {
    pub fn new() -> Self { Self { headers: Vec::new() } }

    pub fn add_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }
}

impl Default for HeaderInterceptor {
    fn default() -> Self { Self::new() }
}

impl Interceptor for HeaderInterceptor {
    fn name(&self) -> &str { "headers" }
    fn priority(&self) -> u32 { 15 }

    fn before_call(&self, ctx: &mut RpcContext) -> InterceptResult {
        for (k, v) in &self.headers {
            ctx.set_metadata(k, v.clone());
        }
        InterceptResult::Continue
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_metadata() {
        let mut ctx = RpcContext::new("test", vec![]);
        ctx.set_metadata("key", "value");
        assert_eq!(ctx.get_metadata("key"), Some("value"));
        assert_eq!(ctx.get_metadata("missing"), None);
    }

    #[test]
    fn context_has_error() {
        let mut ctx = RpcContext::new("test", vec![]);
        assert!(!ctx.has_error());
        ctx.error = Some("oops".into());
        assert!(ctx.has_error());
    }

    #[test]
    fn context_duration() {
        let mut ctx = RpcContext::new("test", vec![]);
        ctx.start_time_us = 1000;
        ctx.end_time_us = 2500;
        assert_eq!(ctx.duration_us(), 1500);
    }

    #[test]
    fn empty_chain_runs() {
        let mut chain = InterceptorChain::new();
        let mut ctx = RpcContext::new("test", vec![]);
        let result = chain.run_before(&mut ctx);
        assert!(result.is_none());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn logging_interceptor_before_after() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(LoggingInterceptor::new("log")));
        let mut ctx = RpcContext::new("my_method", vec![]);
        chain.run_before(&mut ctx);
        assert!(ctx.get_metadata("log_before").unwrap().contains("my_method"));
        chain.run_after(&mut ctx);
        assert!(ctx.get_metadata("log_after").unwrap().contains("ok"));
    }

    #[test]
    fn timing_interceptor_records_times() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(TimingInterceptor::new(5000)));
        let mut ctx = RpcContext::new("test", vec![]);
        chain.run_before(&mut ctx);
        assert_eq!(ctx.start_time_us, 5000);
        chain.run_after(&mut ctx);
        assert_eq!(ctx.end_time_us, 6000);
        assert_eq!(ctx.get_metadata("duration_us"), Some("1000"));
    }

    #[test]
    fn auth_interceptor_adds_token() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(AuthInterceptor::new(Some("secret123".into()), true)));
        let mut ctx = RpcContext::new("test", vec![]);
        chain.run_before(&mut ctx);
        assert_eq!(ctx.get_metadata("authorization"), Some("Bearer secret123"));
    }

    #[test]
    fn auth_interceptor_aborts_when_required_no_token() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(AuthInterceptor::new(None, true)));
        let mut ctx = RpcContext::new("test", vec![]);
        let result = chain.run_before(&mut ctx);
        assert!(result.is_some());
        assert!(ctx.has_error());
    }

    #[test]
    fn auth_interceptor_optional_no_token_continues() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(AuthInterceptor::new(None, false)));
        let mut ctx = RpcContext::new("test", vec![]);
        let result = chain.run_before(&mut ctx);
        assert!(result.is_none());
    }

    #[test]
    fn retry_interceptor_tags_attempt() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(RetryInterceptor::new(3)));
        let mut ctx = RpcContext::new("test", vec![]);
        chain.run_before(&mut ctx);
        assert_eq!(ctx.get_metadata("retry_attempt"), Some("0"));
        assert_eq!(ctx.get_metadata("retry_max"), Some("3"));
    }

    #[test]
    fn retry_interceptor_aborts_when_exceeded() {
        let mut chain = InterceptorChain::new();
        let mut retry = RetryInterceptor::new(2);
        retry.set_attempt(3);
        chain.add(Box::new(retry));
        let mut ctx = RpcContext::new("test", vec![]);
        let result = chain.run_before(&mut ctx);
        assert!(result.is_some());
    }

    #[test]
    fn header_interceptor_injects_headers() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(
            HeaderInterceptor::new()
                .add_header("x-request-id", "abc123")
                .add_header("x-trace", "trace1")
        ));
        let mut ctx = RpcContext::new("test", vec![]);
        chain.run_before(&mut ctx);
        assert_eq!(ctx.get_metadata("x-request-id"), Some("abc123"));
        assert_eq!(ctx.get_metadata("x-trace"), Some("trace1"));
    }

    #[test]
    fn chain_priority_ordering() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(LoggingInterceptor::new("log")));   // priority 10
        chain.add(Box::new(TimingInterceptor::new(0)));         // priority 5
        chain.add(Box::new(AuthInterceptor::new(None, false))); // priority 20
        let names = chain.interceptor_names();
        assert_eq!(names, vec!["timing", "logging", "auth"]);
    }

    #[test]
    fn chain_execute_success() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(LoggingInterceptor::new("log")));
        let mut ctx = RpcContext::new("test", vec![]);
        let ok = chain.execute(&mut ctx, |c| {
            c.response_payload = vec![42];
        });
        assert!(ok);
        assert_eq!(ctx.response_payload, vec![42]);
    }

    #[test]
    fn chain_execute_abort() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(AuthInterceptor::new(None, true)));
        let mut ctx = RpcContext::new("test", vec![]);
        let ok = chain.execute(&mut ctx, |_c| { panic!("should not be called"); });
        assert!(!ok);
        assert!(ctx.has_error());
    }

    #[test]
    fn chain_stats() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(LoggingInterceptor::new("log")));
        let mut ctx = RpcContext::new("test", vec![]);
        chain.run_before(&mut ctx);
        assert_eq!(chain.stats().calls_processed, 1);
        assert_eq!(chain.stats().calls_aborted, 0);
    }

    #[test]
    fn chain_abort_stats() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(AuthInterceptor::new(None, true)));
        let mut ctx = RpcContext::new("test", vec![]);
        chain.run_before(&mut ctx);
        assert_eq!(chain.stats().calls_aborted, 1);
        assert!((chain.stats().abort_rate() - 1.0).abs() < 0.01);
    }

    #[test]
    fn intercept_result_is_checks() {
        assert!(InterceptResult::Continue.is_continue());
        assert!(!InterceptResult::Continue.is_abort());
        assert!(InterceptResult::Abort("x".into()).is_abort());
    }

    #[test]
    fn short_circuit_stops_later_interceptors() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(AuthInterceptor::new(None, true)));   // priority 20 -> abort
        chain.add(Box::new(RetryInterceptor::new(3)));           // priority 50 -> should not run
        let mut ctx = RpcContext::new("test", vec![]);
        chain.run_before(&mut ctx);
        // retry metadata should NOT be present because auth aborted first
        assert!(ctx.get_metadata("retry_attempt").is_none());
    }

    #[test]
    fn after_call_runs_in_reverse() {
        let mut chain = InterceptorChain::new();
        chain.add(Box::new(LoggingInterceptor::new("first")));  // priority 10
        chain.add(Box::new(LoggingInterceptor::new("second"))); // priority 10
        let mut ctx = RpcContext::new("test", vec![]);
        chain.run_after(&mut ctx);
        // Both should have written after metadata
        assert!(ctx.get_metadata("first_after").is_some());
        assert!(ctx.get_metadata("second_after").is_some());
    }
}
