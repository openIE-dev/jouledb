//! API rate limiting with quotas — per-endpoint limits, tiered plans
//! (free/pro/enterprise), quota tracking, sliding window, rate limit headers,
//! quota reset scheduling, and overage handling.
//!
//! Replaces `express-rate-limit`, `bottleneck`, `rate-limiter-flexible`, and
//! similar JS libraries with a pure-Rust, zero-allocation-path rate limiter.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Rate limiting error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitError {
    /// Rate limited — too many requests.
    RateLimited {
        retry_after_ms: u64,
        limit: u64,
        remaining: u64,
    },
    /// Quota exceeded for the billing period.
    QuotaExceeded {
        plan: String,
        limit: u64,
        used: u64,
        reset_at_ms: u64,
    },
    /// Unknown client ID.
    UnknownClient(String),
    /// Unknown endpoint.
    UnknownEndpoint(String),
    /// Plan not found.
    PlanNotFound(String),
}

impl fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RateLimited { retry_after_ms, limit, remaining } => {
                write!(
                    f,
                    "rate limited: {remaining}/{limit} remaining, retry after {retry_after_ms}ms"
                )
            }
            Self::QuotaExceeded { plan, limit, used, .. } => {
                write!(f, "quota exceeded on plan '{plan}': {used}/{limit}")
            }
            Self::UnknownClient(id) => write!(f, "unknown client: {id}"),
            Self::UnknownEndpoint(ep) => write!(f, "unknown endpoint: {ep}"),
            Self::PlanNotFound(name) => write!(f, "plan not found: {name}"),
        }
    }
}

impl std::error::Error for RateLimitError {}

// ── Types ────────────────────────────────────────────────────────

/// A tiered plan with rate limits and quotas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub name: String,
    /// Requests per window (rate limit).
    pub requests_per_window: u64,
    /// Window duration in milliseconds.
    pub window_ms: u64,
    /// Total requests allowed per billing period (quota). 0 = unlimited.
    pub quota_limit: u64,
    /// Billing period in milliseconds. 0 = no period.
    pub quota_period_ms: u64,
    /// Burst allowance above the rate limit.
    pub burst_allowance: u64,
    /// Whether to allow overage (with tracking) or hard-reject.
    pub allow_overage: bool,
}

impl Plan {
    /// Create a new plan.
    pub fn new(name: &str, requests_per_window: u64, window_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            requests_per_window,
            window_ms,
            quota_limit: 0,
            quota_period_ms: 0,
            burst_allowance: 0,
            allow_overage: false,
        }
    }

    /// Set the quota.
    pub fn with_quota(mut self, limit: u64, period_ms: u64) -> Self {
        self.quota_limit = limit;
        self.quota_period_ms = period_ms;
        self
    }

    /// Set the burst allowance.
    pub fn with_burst(mut self, burst: u64) -> Self {
        self.burst_allowance = burst;
        self
    }

    /// Allow overage instead of hard rejection.
    pub fn with_overage(mut self) -> Self {
        self.allow_overage = true;
        self
    }
}

/// Pre-defined plans.
pub fn free_plan() -> Plan {
    Plan::new("free", 60, 60_000)
        .with_quota(1_000, 86_400_000) // 1K/day
}

pub fn pro_plan() -> Plan {
    Plan::new("pro", 600, 60_000)
        .with_quota(100_000, 86_400_000) // 100K/day
        .with_burst(100)
}

pub fn enterprise_plan() -> Plan {
    Plan::new("enterprise", 6_000, 60_000)
        .with_quota(10_000_000, 86_400_000) // 10M/day
        .with_burst(1_000)
        .with_overage()
}

/// Per-endpoint rate limit override.
#[derive(Debug, Clone)]
pub struct EndpointLimit {
    pub endpoint: String,
    pub requests_per_window: u64,
    pub window_ms: u64,
}

/// Rate limit response headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitHeaders {
    pub limit: u64,
    pub remaining: u64,
    pub reset_ms: u64,
    pub retry_after_ms: Option<u64>,
    pub quota_limit: Option<u64>,
    pub quota_remaining: Option<u64>,
    pub quota_reset_ms: Option<u64>,
}

impl RateLimitHeaders {
    /// Convert to HTTP header name-value pairs.
    pub fn to_headers(&self) -> Vec<(String, String)> {
        let mut headers = vec![
            ("X-RateLimit-Limit".to_string(), self.limit.to_string()),
            (
                "X-RateLimit-Remaining".to_string(),
                self.remaining.to_string(),
            ),
            ("X-RateLimit-Reset".to_string(), self.reset_ms.to_string()),
        ];
        if let Some(retry) = self.retry_after_ms {
            let secs = (retry + 999) / 1000;
            headers.push(("Retry-After".to_string(), secs.to_string()));
        }
        if let Some(ql) = self.quota_limit {
            headers.push(("X-Quota-Limit".to_string(), ql.to_string()));
        }
        if let Some(qr) = self.quota_remaining {
            headers.push(("X-Quota-Remaining".to_string(), qr.to_string()));
        }
        if let Some(qreset) = self.quota_reset_ms {
            headers.push(("X-Quota-Reset".to_string(), qreset.to_string()));
        }
        headers
    }
}

// ── Sliding Window ───────────────────────────────────────────────

/// Sliding window counter for a single client+endpoint.
struct SlidingWindow {
    /// Fixed window start times and counts.
    windows: Vec<(u64, u64)>,
    window_ms: u64,
    max_requests: u64,
}

impl SlidingWindow {
    fn new(max_requests: u64, window_ms: u64) -> Self {
        Self {
            windows: Vec::new(),
            window_ms,
            max_requests,
        }
    }

    fn current_count(&self, now_ms: u64) -> u64 {
        let window_start = (now_ms / self.window_ms) * self.window_ms;

        let current_count = self
            .windows
            .iter()
            .find(|(start, _)| *start == window_start)
            .map(|(_, count)| *count)
            .unwrap_or(0);

        // Only add weighted previous window if it is a distinct window.
        let prev_window_start = window_start.saturating_sub(self.window_ms);
        if prev_window_start == window_start {
            return current_count;
        }

        let prev_count = self
            .windows
            .iter()
            .find(|(start, _)| *start == prev_window_start)
            .map(|(_, count)| *count)
            .unwrap_or(0);

        // Weighted: how far into the current window are we?
        let elapsed_in_window = now_ms - window_start;
        let weight = 1.0 - (elapsed_in_window as f64 / self.window_ms as f64);
        let weighted_prev = (prev_count as f64 * weight) as u64;

        weighted_prev + current_count
    }

    fn record(&mut self, now_ms: u64) {
        let window_start = (now_ms / self.window_ms) * self.window_ms;
        if let Some(entry) = self.windows.iter_mut().find(|(start, _)| *start == window_start) {
            entry.1 += 1;
        } else {
            self.windows.push((window_start, 1));
        }
        // Prune old windows (keep last 3)
        let cutoff = window_start.saturating_sub(self.window_ms * 2);
        self.windows.retain(|(start, _)| *start >= cutoff);
    }

    fn try_acquire(&mut self, now_ms: u64, burst: u64) -> bool {
        let count = self.current_count(now_ms);
        if count < self.max_requests + burst {
            self.record(now_ms);
            true
        } else {
            false
        }
    }

    fn remaining(&self, now_ms: u64, burst: u64) -> u64 {
        let count = self.current_count(now_ms);
        (self.max_requests + burst).saturating_sub(count)
    }

    fn reset_ms(&self, now_ms: u64) -> u64 {
        let window_start = (now_ms / self.window_ms) * self.window_ms;
        window_start + self.window_ms
    }
}

// ── Quota Tracker ────────────────────────────────────────────────

struct QuotaTracker {
    limit: u64,
    used: u64,
    period_ms: u64,
    period_start_ms: u64,
    allow_overage: bool,
    overage_count: u64,
}

impl QuotaTracker {
    fn new(limit: u64, period_ms: u64, start_ms: u64, allow_overage: bool) -> Self {
        Self {
            limit,
            used: 0,
            period_ms,
            period_start_ms: start_ms,
            allow_overage,
            overage_count: 0,
        }
    }

    fn reset_if_needed(&mut self, now_ms: u64) {
        if self.period_ms > 0 && now_ms >= self.period_start_ms + self.period_ms {
            // How many full periods have elapsed?
            let elapsed = now_ms - self.period_start_ms;
            let periods = elapsed / self.period_ms;
            self.period_start_ms += periods * self.period_ms;
            self.used = 0;
            self.overage_count = 0;
        }
    }

    fn try_consume(&mut self, now_ms: u64) -> Result<(), RateLimitError> {
        self.reset_if_needed(now_ms);
        if self.limit == 0 {
            // unlimited
            self.used += 1;
            return Ok(());
        }
        if self.used < self.limit {
            self.used += 1;
            Ok(())
        } else if self.allow_overage {
            self.used += 1;
            self.overage_count += 1;
            Ok(())
        } else {
            Err(RateLimitError::QuotaExceeded {
                plan: String::new(), // filled by caller
                limit: self.limit,
                used: self.used,
                reset_at_ms: self.period_start_ms + self.period_ms,
            })
        }
    }

    fn remaining(&self, now_ms: u64) -> u64 {
        // Don't mutate; just compute
        let effective_start = if self.period_ms > 0 && now_ms >= self.period_start_ms + self.period_ms {
            let elapsed = now_ms - self.period_start_ms;
            let periods = elapsed / self.period_ms;
            self.period_start_ms + periods * self.period_ms
        } else {
            self.period_start_ms
        };
        if now_ms >= effective_start + self.period_ms && self.period_ms > 0 {
            // period rolled over
            self.limit
        } else {
            self.limit.saturating_sub(self.used)
        }
    }

    fn reset_at_ms(&self) -> u64 {
        self.period_start_ms + self.period_ms
    }
}

// ── Client State ─────────────────────────────────────────────────

struct ClientState {
    plan_name: String,
    /// Global rate limit window.
    global_window: SlidingWindow,
    /// Per-endpoint windows.
    endpoint_windows: HashMap<String, SlidingWindow>,
    /// Quota tracker.
    quota: Option<QuotaTracker>,
}

// ── Rate Limiter ─────────────────────────────────────────────────

/// API rate limiter with quotas and per-endpoint limits.
pub struct ApiRateLimiter {
    plans: HashMap<String, Plan>,
    endpoint_limits: HashMap<String, EndpointLimit>,
    clients: HashMap<String, ClientState>,
}

impl ApiRateLimiter {
    /// Create a new rate limiter.
    pub fn new() -> Self {
        Self {
            plans: HashMap::new(),
            endpoint_limits: HashMap::new(),
            clients: HashMap::new(),
        }
    }

    /// Register a plan.
    pub fn add_plan(&mut self, plan: Plan) {
        self.plans.insert(plan.name.clone(), plan);
    }

    /// Register a per-endpoint limit override.
    pub fn add_endpoint_limit(&mut self, limit: EndpointLimit) {
        self.endpoint_limits.insert(limit.endpoint.clone(), limit);
    }

    /// Register a client with a specific plan.
    pub fn register_client(
        &mut self,
        client_id: &str,
        plan_name: &str,
        now_ms: u64,
    ) -> Result<(), RateLimitError> {
        let plan = self
            .plans
            .get(plan_name)
            .ok_or_else(|| RateLimitError::PlanNotFound(plan_name.to_string()))?
            .clone();

        let quota = if plan.quota_limit > 0 {
            Some(QuotaTracker::new(
                plan.quota_limit,
                plan.quota_period_ms,
                now_ms,
                plan.allow_overage,
            ))
        } else {
            None
        };

        self.clients.insert(
            client_id.to_string(),
            ClientState {
                plan_name: plan.name.clone(),
                global_window: SlidingWindow::new(plan.requests_per_window, plan.window_ms),
                endpoint_windows: HashMap::new(),
                quota,
            },
        );
        Ok(())
    }

    /// Attempt a request. Returns rate limit headers on success.
    pub fn check(
        &mut self,
        client_id: &str,
        endpoint: &str,
        now_ms: u64,
    ) -> Result<RateLimitHeaders, RateLimitError> {
        let plan_name = {
            let state = self
                .clients
                .get(client_id)
                .ok_or_else(|| RateLimitError::UnknownClient(client_id.to_string()))?;
            state.plan_name.clone()
        };

        let plan = self
            .plans
            .get(&plan_name)
            .cloned()
            .ok_or_else(|| RateLimitError::PlanNotFound(plan_name.clone()))?;

        let burst = plan.burst_allowance;

        // Check per-endpoint limit first
        let ep_limit = self.endpoint_limits.get(endpoint).cloned();
        if let Some(ref el) = ep_limit {
            let state = self.clients.get_mut(client_id).unwrap();
            let window = state
                .endpoint_windows
                .entry(endpoint.to_string())
                .or_insert_with(|| SlidingWindow::new(el.requests_per_window, el.window_ms));
            if !window.try_acquire(now_ms, 0) {
                let remaining = window.remaining(now_ms, 0);
                let reset = window.reset_ms(now_ms);
                return Err(RateLimitError::RateLimited {
                    retry_after_ms: reset.saturating_sub(now_ms),
                    limit: el.requests_per_window,
                    remaining,
                });
            }
        }

        let state = self.clients.get_mut(client_id).unwrap();

        // Check global rate limit
        if !state.global_window.try_acquire(now_ms, burst) {
            let remaining = state.global_window.remaining(now_ms, burst);
            let reset = state.global_window.reset_ms(now_ms);
            return Err(RateLimitError::RateLimited {
                retry_after_ms: reset.saturating_sub(now_ms),
                limit: plan.requests_per_window,
                remaining,
            });
        }

        // Check quota
        let mut quota_headers = (None, None, None);
        if let Some(ref mut quota) = state.quota {
            if let Err(mut e) = quota.try_consume(now_ms) {
                if let RateLimitError::QuotaExceeded { ref mut plan, .. } = e {
                    *plan = plan_name.clone();
                }
                return Err(e);
            }
            quota_headers = (
                Some(quota.limit),
                Some(quota.remaining(now_ms)),
                Some(quota.reset_at_ms()),
            );
        }

        let remaining = state.global_window.remaining(now_ms, burst);
        let reset = state.global_window.reset_ms(now_ms);

        Ok(RateLimitHeaders {
            limit: plan.requests_per_window + burst,
            remaining,
            reset_ms: reset,
            retry_after_ms: None,
            quota_limit: quota_headers.0,
            quota_remaining: quota_headers.1,
            quota_reset_ms: quota_headers.2,
        })
    }

    /// Get the current usage for a client.
    pub fn usage(&self, client_id: &str, now_ms: u64) -> Option<ClientUsage> {
        let state = self.clients.get(client_id)?;
        let plan = self.plans.get(&state.plan_name)?;
        let burst = plan.burst_allowance;
        Some(ClientUsage {
            plan: state.plan_name.clone(),
            rate_remaining: state.global_window.remaining(now_ms, burst),
            rate_limit: plan.requests_per_window + burst,
            quota_used: state.quota.as_ref().map(|q| q.used).unwrap_or(0),
            quota_limit: state.quota.as_ref().map(|q| q.limit).unwrap_or(0),
            overage_count: state.quota.as_ref().map(|q| q.overage_count).unwrap_or(0),
        })
    }

    /// Reset a client's rate limit state.
    pub fn reset_client(&mut self, client_id: &str, now_ms: u64) -> Result<(), RateLimitError> {
        let state = self
            .clients
            .get_mut(client_id)
            .ok_or_else(|| RateLimitError::UnknownClient(client_id.to_string()))?;
        let plan = self
            .plans
            .get(&state.plan_name)
            .ok_or_else(|| RateLimitError::PlanNotFound(state.plan_name.clone()))?
            .clone();
        state.global_window = SlidingWindow::new(plan.requests_per_window, plan.window_ms);
        state.endpoint_windows.clear();
        if let Some(ref mut quota) = state.quota {
            quota.used = 0;
            quota.overage_count = 0;
            quota.period_start_ms = now_ms;
        }
        Ok(())
    }
}

impl Default for ApiRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Usage summary for a client.
#[derive(Debug, Clone)]
pub struct ClientUsage {
    pub plan: String,
    pub rate_remaining: u64,
    pub rate_limit: u64,
    pub quota_used: u64,
    pub quota_limit: u64,
    pub overage_count: u64,
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_limiter() -> ApiRateLimiter {
        let mut limiter = ApiRateLimiter::new();
        limiter.add_plan(Plan::new("test", 5, 1000));
        limiter.register_client("client1", "test", 0).unwrap();
        limiter
    }

    #[test]
    fn basic_rate_limiting() {
        let mut limiter = setup_limiter();
        for i in 0..5 {
            let result = limiter.check("client1", "/api/data", i * 10);
            assert!(result.is_ok(), "request {i} should succeed");
        }
        let result = limiter.check("client1", "/api/data", 50);
        assert!(result.is_err());
    }

    #[test]
    fn rate_limit_resets_after_window() {
        let mut limiter = setup_limiter();
        for i in 0..5 {
            limiter.check("client1", "/api/data", i * 10).unwrap();
        }
        // After window passes, should be allowed again
        let result = limiter.check("client1", "/api/data", 2000);
        assert!(result.is_ok());
    }

    #[test]
    fn rate_limit_headers() {
        let mut limiter = setup_limiter();
        let headers = limiter.check("client1", "/api/data", 0).unwrap();
        assert_eq!(headers.limit, 5);
        // remaining should be <= limit
        assert!(headers.remaining <= 5);
    }

    #[test]
    fn headers_to_http() {
        let headers = RateLimitHeaders {
            limit: 100,
            remaining: 50,
            reset_ms: 1000,
            retry_after_ms: None,
            quota_limit: Some(10000),
            quota_remaining: Some(9950),
            quota_reset_ms: Some(86400000),
        };
        let http = headers.to_headers();
        assert!(http.iter().any(|(k, v)| k == "X-RateLimit-Limit" && v == "100"));
        assert!(http.iter().any(|(k, _)| k == "X-Quota-Limit"));
    }

    #[test]
    fn retry_after_header() {
        let headers = RateLimitHeaders {
            limit: 10,
            remaining: 0,
            reset_ms: 5000,
            retry_after_ms: Some(3000),
            quota_limit: None,
            quota_remaining: None,
            quota_reset_ms: None,
        };
        let http = headers.to_headers();
        assert!(http.iter().any(|(k, v)| k == "Retry-After" && v == "3"));
    }

    #[test]
    fn quota_enforcement() {
        let mut limiter = ApiRateLimiter::new();
        limiter.add_plan(Plan::new("quotaplan", 100, 1000).with_quota(3, 10_000));
        limiter.register_client("c1", "quotaplan", 0).unwrap();

        for i in 0..3 {
            assert!(limiter.check("c1", "/api", i * 100).is_ok());
        }
        let result = limiter.check("c1", "/api", 400);
        assert!(matches!(result, Err(RateLimitError::QuotaExceeded { .. })));
    }

    #[test]
    fn quota_resets_after_period() {
        let mut limiter = ApiRateLimiter::new();
        limiter.add_plan(Plan::new("qp", 100, 1000).with_quota(2, 5000));
        limiter.register_client("c1", "qp", 0).unwrap();

        limiter.check("c1", "/api", 100).unwrap();
        limiter.check("c1", "/api", 200).unwrap();
        assert!(limiter.check("c1", "/api", 300).is_err());

        // After period reset
        assert!(limiter.check("c1", "/api", 6000).is_ok());
    }

    #[test]
    fn overage_allowed() {
        let mut limiter = ApiRateLimiter::new();
        limiter.add_plan(
            Plan::new("overage", 100, 1000)
                .with_quota(2, 10_000)
                .with_overage(),
        );
        limiter.register_client("c1", "overage", 0).unwrap();

        limiter.check("c1", "/api", 0).unwrap();
        limiter.check("c1", "/api", 10).unwrap();
        // Third request would exceed quota, but overage is allowed
        let result = limiter.check("c1", "/api", 20);
        assert!(result.is_ok());

        let usage = limiter.usage("c1", 30).unwrap();
        assert_eq!(usage.overage_count, 1);
    }

    #[test]
    fn burst_allowance() {
        let mut limiter = ApiRateLimiter::new();
        limiter.add_plan(Plan::new("burst", 5, 1000).with_burst(3));
        limiter.register_client("c1", "burst", 0).unwrap();

        // Should allow 5 + 3 = 8 requests
        for i in 0..8 {
            assert!(
                limiter.check("c1", "/api", i * 10).is_ok(),
                "request {i} should succeed"
            );
        }
        assert!(limiter.check("c1", "/api", 80).is_err());
    }

    #[test]
    fn per_endpoint_limits() {
        let mut limiter = ApiRateLimiter::new();
        limiter.add_plan(Plan::new("test", 100, 1000));
        limiter.add_endpoint_limit(EndpointLimit {
            endpoint: "/api/expensive".to_string(),
            requests_per_window: 2,
            window_ms: 1000,
        });
        limiter.register_client("c1", "test", 0).unwrap();

        limiter.check("c1", "/api/expensive", 0).unwrap();
        limiter.check("c1", "/api/expensive", 10).unwrap();
        let result = limiter.check("c1", "/api/expensive", 20);
        assert!(result.is_err());

        // Regular endpoint still works
        assert!(limiter.check("c1", "/api/normal", 30).is_ok());
    }

    #[test]
    fn unknown_client_error() {
        let mut limiter = setup_limiter();
        let result = limiter.check("unknown", "/api", 0);
        assert!(matches!(result, Err(RateLimitError::UnknownClient(_))));
    }

    #[test]
    fn unknown_plan_error() {
        let mut limiter = ApiRateLimiter::new();
        let result = limiter.register_client("c1", "nonexistent", 0);
        assert!(matches!(result, Err(RateLimitError::PlanNotFound(_))));
    }

    #[test]
    fn predefined_plans() {
        let free = free_plan();
        assert_eq!(free.name, "free");
        assert_eq!(free.requests_per_window, 60);

        let pro = pro_plan();
        assert_eq!(pro.name, "pro");
        assert_eq!(pro.burst_allowance, 100);

        let ent = enterprise_plan();
        assert!(ent.allow_overage);
    }

    #[test]
    fn client_usage() {
        let mut limiter = setup_limiter();
        limiter.check("client1", "/api", 0).unwrap();
        let usage = limiter.usage("client1", 10).unwrap();
        assert_eq!(usage.plan, "test");
        assert!(usage.rate_remaining < 5);
    }

    #[test]
    fn reset_client_state() {
        let mut limiter = setup_limiter();
        for i in 0..5 {
            limiter.check("client1", "/api", i * 10).unwrap();
        }
        assert!(limiter.check("client1", "/api", 50).is_err());

        limiter.reset_client("client1", 100).unwrap();
        assert!(limiter.check("client1", "/api", 100).is_ok());
    }

    #[test]
    fn error_display() {
        let err = RateLimitError::RateLimited {
            retry_after_ms: 5000,
            limit: 100,
            remaining: 0,
        };
        let s = err.to_string();
        assert!(s.contains("rate limited"));

        let err = RateLimitError::QuotaExceeded {
            plan: "free".to_string(),
            limit: 1000,
            used: 1001,
            reset_at_ms: 86400000,
        };
        assert!(err.to_string().contains("quota exceeded"));
    }

    #[test]
    fn usage_nonexistent_client() {
        let limiter = setup_limiter();
        assert!(limiter.usage("nobody", 0).is_none());
    }

    #[test]
    fn plan_builder() {
        let plan = Plan::new("custom", 200, 5000)
            .with_quota(50000, 86400000)
            .with_burst(50)
            .with_overage();
        assert_eq!(plan.quota_limit, 50000);
        assert_eq!(plan.burst_allowance, 50);
        assert!(plan.allow_overage);
    }

    #[test]
    fn sliding_window_weighted_count() {
        // The sliding window uses weighted previous window count
        let mut window = SlidingWindow::new(10, 1000);
        // Fill up previous window
        for _ in 0..8 {
            window.record(500);
        }
        // At time 1500 (halfway into next window), ~50% of prev should count
        let count = window.current_count(1500);
        // Should be approximately 4 (50% of 8)
        assert!(count >= 3 && count <= 5, "weighted count was {count}");
    }
}
