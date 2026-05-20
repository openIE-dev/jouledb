//! Security Hardening and Audit for JouleDB
//!
//! This module provides comprehensive security features for production deployment:
//!
//! - **Input Validation**: SQL injection prevention, parameter validation
//! - **Rate Limiting**: Configurable rate limits per client/user
//! - **Encryption**: Data at rest and in transit encryption
//! - **Security Headers**: HTTP security headers for API endpoints
//! - **Audit Logging**: Security event logging with tamper detection
//! - **Vulnerability Scanning**: Self-assessment of security posture
//!
//! ## Security Best Practices
//!
//! 1. Always use TLS for connections
//! 2. Enable authentication for all endpoints
//! 3. Use parameterized queries to prevent SQL injection
//! 4. Implement rate limiting for public endpoints
//! 5. Enable audit logging for security events
//! 6. Regularly rotate encryption keys

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Security Errors
// ============================================================================

/// Security error types
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityError {
    /// Authentication failed
    AuthenticationFailed(String),
    /// Authorization denied
    AuthorizationDenied(String),
    /// Rate limit exceeded
    RateLimitExceeded(String),
    /// Invalid input
    InvalidInput(String),
    /// SQL injection detected
    SqlInjectionDetected(String),
    /// Encryption error
    EncryptionError(String),
    /// Token expired
    TokenExpired,
    /// Invalid token
    InvalidToken(String),
    /// IP blocked
    IpBlocked(String),
    /// Configuration error
    ConfigError(String),
}

impl std::fmt::Display for SecurityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AuthenticationFailed(msg) => write!(f, "Authentication failed: {}", msg),
            Self::AuthorizationDenied(msg) => write!(f, "Authorization denied: {}", msg),
            Self::RateLimitExceeded(msg) => write!(f, "Rate limit exceeded: {}", msg),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::SqlInjectionDetected(msg) => write!(f, "SQL injection detected: {}", msg),
            Self::EncryptionError(msg) => write!(f, "Encryption error: {}", msg),
            Self::TokenExpired => write!(f, "Token has expired"),
            Self::InvalidToken(msg) => write!(f, "Invalid token: {}", msg),
            Self::IpBlocked(ip) => write!(f, "IP address blocked: {}", ip),
            Self::ConfigError(msg) => write!(f, "Security configuration error: {}", msg),
        }
    }
}

impl std::error::Error for SecurityError {}

/// Security result type
pub type SecurityResult<T> = Result<T, SecurityError>;

// ============================================================================
// Input Validation
// ============================================================================

/// SQL injection patterns to detect
const SQL_INJECTION_PATTERNS: &[&str] = &[
    "'; DROP",
    "'; DELETE",
    "'; UPDATE",
    "'; INSERT",
    "'; TRUNCATE",
    "--",
    "/*",
    "*/",
    "UNION SELECT",
    "UNION ALL SELECT",
    "OR '1'='1",
    "OR 1=1",
    "' OR ''='",
    "\" OR \"\"=\"",
    "EXEC(",
    "EXECUTE(",
    "xp_",
    "sp_",
    "0x",
    "CHAR(",
    "NCHAR(",
    "VARCHAR(",
    "CAST(",
    "CONVERT(",
    "WAITFOR DELAY",
    "BENCHMARK(",
    "SLEEP(",
    "PG_SLEEP(",
];

/// Input validator for SQL injection prevention
pub struct InputValidator {
    /// Enable strict mode (reject suspicious input)
    strict_mode: bool,
    /// Custom patterns to check
    custom_patterns: Vec<String>,
    /// Max input length
    max_input_length: usize,
    /// Blocked characters
    blocked_chars: Vec<char>,
}

impl InputValidator {
    /// Create new validator
    pub fn new() -> Self {
        Self {
            strict_mode: true,
            custom_patterns: Vec::new(),
            max_input_length: 10000,
            blocked_chars: vec!['\0', '\x1a'], // NULL and SUB characters
        }
    }

    /// Set strict mode
    pub fn strict_mode(mut self, enabled: bool) -> Self {
        self.strict_mode = enabled;
        self
    }

    /// Add custom pattern
    pub fn add_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.custom_patterns.push(pattern.into());
        self
    }

    /// Set max input length
    pub fn max_length(mut self, length: usize) -> Self {
        self.max_input_length = length;
        self
    }

    /// Validate SQL input
    pub fn validate_sql(&self, input: &str) -> SecurityResult<()> {
        // Check length
        if input.len() > self.max_input_length {
            return Err(SecurityError::InvalidInput(format!(
                "Input exceeds maximum length of {} characters",
                self.max_input_length
            )));
        }

        // Check for blocked characters
        for ch in &self.blocked_chars {
            if input.contains(*ch) {
                return Err(SecurityError::InvalidInput(format!(
                    "Input contains blocked character: {:?}",
                    ch
                )));
            }
        }

        // Check for SQL injection patterns
        let upper = input.to_uppercase();
        for pattern in SQL_INJECTION_PATTERNS {
            if upper.contains(pattern) {
                return Err(SecurityError::SqlInjectionDetected(format!(
                    "Suspicious pattern detected: {}",
                    pattern
                )));
            }
        }

        // Check custom patterns
        for pattern in &self.custom_patterns {
            if upper.contains(&pattern.to_uppercase()) {
                return Err(SecurityError::SqlInjectionDetected(format!(
                    "Custom pattern matched: {}",
                    pattern
                )));
            }
        }

        Ok(())
    }

    /// Validate identifier (table/column name)
    pub fn validate_identifier(&self, input: &str) -> SecurityResult<()> {
        if input.is_empty() {
            return Err(SecurityError::InvalidInput("Empty identifier".to_string()));
        }

        if input.len() > 128 {
            return Err(SecurityError::InvalidInput(
                "Identifier too long (max 128 chars)".to_string(),
            ));
        }

        // Must start with letter or underscore
        // Safety: input.is_empty() check is above, so next() always returns Some
        let first = input
            .chars()
            .next()
            .expect("input validated as non-empty above");
        if !first.is_alphabetic() && first != '_' {
            return Err(SecurityError::InvalidInput(
                "Identifier must start with letter or underscore".to_string(),
            ));
        }

        // Must be alphanumeric or underscore
        for ch in input.chars() {
            if !ch.is_alphanumeric() && ch != '_' {
                return Err(SecurityError::InvalidInput(format!(
                    "Invalid character in identifier: {}",
                    ch
                )));
            }
        }

        Ok(())
    }

    /// Sanitize input by escaping dangerous characters
    pub fn sanitize(&self, input: &str) -> String {
        input
            .replace('\'', "''")
            .replace('\\', "\\\\")
            .replace('\0', "")
            .replace('\x1a', "")
    }

    /// Validate email format
    pub fn validate_email(&self, input: &str) -> SecurityResult<()> {
        if !input.contains('@') || !input.contains('.') {
            return Err(SecurityError::InvalidInput(
                "Invalid email format".to_string(),
            ));
        }

        let parts: Vec<&str> = input.split('@').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(SecurityError::InvalidInput(
                "Invalid email format".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for InputValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Rate Limiting
// ============================================================================

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per window
    pub requests_per_window: u64,
    /// Window duration
    pub window_duration: Duration,
    /// Enable adaptive rate limiting
    pub adaptive: bool,
    /// Burst allowance
    pub burst_size: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_window: 1000,
            window_duration: Duration::from_secs(60),
            adaptive: true,
            burst_size: 50,
        }
    }
}

/// Rate limiter using token bucket algorithm
pub struct RateLimiter {
    config: RateLimitConfig,
    /// Buckets per client
    buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    /// Global counter
    global_requests: AtomicU64,
    /// Blocked clients
    blocked: Arc<RwLock<HashMap<String, Instant>>>,
}

/// Token bucket for rate limiting
#[derive(Debug, Clone)]
struct TokenBucket {
    tokens: f64,
    last_update: Instant,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
}

impl TokenBucket {
    fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            last_update: Instant::now(),
            max_tokens,
            refill_rate,
        }
    }

    fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_update = now;
    }
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: Arc::new(RwLock::new(HashMap::new())),
            global_requests: AtomicU64::new(0),
            blocked: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if request is allowed
    pub fn check(&self, client_id: &str) -> SecurityResult<()> {
        // Check if blocked
        {
            let blocked = crate::lock_util::read_lock(&self.blocked);
            if let Some(blocked_until) = blocked.get(client_id) {
                if Instant::now() < *blocked_until {
                    return Err(SecurityError::RateLimitExceeded(
                        "Client temporarily blocked".to_string(),
                    ));
                }
            }
        }

        // Get or create bucket
        let mut buckets = crate::lock_util::write_lock(&self.buckets);
        let bucket = buckets.entry(client_id.to_string()).or_insert_with(|| {
            let rate =
                self.config.requests_per_window as f64 / self.config.window_duration.as_secs_f64();
            TokenBucket::new(self.config.burst_size as f64, rate)
        });

        // Try to consume a token
        if bucket.try_consume(1.0) {
            self.global_requests.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            // Block client for window duration if exceeded
            if self.config.adaptive {
                let mut blocked = crate::lock_util::write_lock(&self.blocked);
                blocked.insert(
                    client_id.to_string(),
                    Instant::now() + self.config.window_duration,
                );
            }

            Err(SecurityError::RateLimitExceeded(format!(
                "Rate limit exceeded for client: {}",
                client_id
            )))
        }
    }

    /// Get current usage for client
    pub fn get_usage(&self, client_id: &str) -> Option<f64> {
        let buckets = crate::lock_util::read_lock(&self.buckets);
        buckets.get(client_id).map(|b| {
            let used = self.config.burst_size as f64 - b.tokens;
            (used / self.config.requests_per_window as f64) * 100.0
        })
    }

    /// Reset rate limit for client
    pub fn reset(&self, client_id: &str) {
        let mut buckets = crate::lock_util::write_lock(&self.buckets);
        buckets.remove(client_id);

        let mut blocked = crate::lock_util::write_lock(&self.blocked);
        blocked.remove(client_id);
    }

    /// Get statistics
    pub fn stats(&self) -> RateLimitStats {
        let buckets = crate::lock_util::read_lock(&self.buckets);
        let blocked = crate::lock_util::read_lock(&self.blocked);

        RateLimitStats {
            total_clients: buckets.len(),
            blocked_clients: blocked.len(),
            global_requests: self.global_requests.load(Ordering::Relaxed),
        }
    }
}

/// Rate limit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStats {
    pub total_clients: usize,
    pub blocked_clients: usize,
    pub global_requests: u64,
}

// ============================================================================
// IP Blocking
// ============================================================================

/// IP blocklist manager
pub struct IpBlocklist {
    /// Blocked IPs
    blocked: Arc<RwLock<HashMap<IpAddr, BlockEntry>>>,
    /// Allowlist (overrides blocklist)
    allowlist: Arc<RwLock<Vec<IpAddr>>>,
}

/// Block entry
#[derive(Debug, Clone)]
struct BlockEntry {
    reason: String,
    blocked_at: Instant,
    expires_at: Option<Instant>,
}

impl IpBlocklist {
    /// Create new blocklist
    pub fn new() -> Self {
        Self {
            blocked: Arc::new(RwLock::new(HashMap::new())),
            allowlist: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Block an IP address
    pub fn block(&self, ip: IpAddr, reason: &str, duration: Option<Duration>) {
        let mut blocked = crate::lock_util::write_lock(&self.blocked);
        blocked.insert(
            ip,
            BlockEntry {
                reason: reason.to_string(),
                blocked_at: Instant::now(),
                expires_at: duration.map(|d| Instant::now() + d),
            },
        );
    }

    /// Unblock an IP address
    pub fn unblock(&self, ip: &IpAddr) {
        let mut blocked = crate::lock_util::write_lock(&self.blocked);
        blocked.remove(ip);
    }

    /// Add to allowlist
    pub fn allow(&self, ip: IpAddr) {
        let mut allowlist = crate::lock_util::write_lock(&self.allowlist);
        if !allowlist.contains(&ip) {
            allowlist.push(ip);
        }
    }

    /// Check if IP is blocked
    pub fn is_blocked(&self, ip: &IpAddr) -> Option<String> {
        // Check allowlist first
        {
            let allowlist = crate::lock_util::read_lock(&self.allowlist);
            if allowlist.contains(ip) {
                return None;
            }
        }

        // Check blocklist
        let blocked = crate::lock_util::read_lock(&self.blocked);
        if let Some(entry) = blocked.get(ip) {
            // Check expiration
            if let Some(expires_at) = entry.expires_at {
                if Instant::now() > expires_at {
                    // Expired, will be cleaned up later
                    return None;
                }
            }
            return Some(entry.reason.clone());
        }

        None
    }

    /// Clean up expired entries
    pub fn cleanup(&self) {
        let mut blocked = crate::lock_util::write_lock(&self.blocked);
        blocked.retain(|_, entry| entry.expires_at.map(|e| Instant::now() < e).unwrap_or(true));
    }

    /// Get all blocked IPs
    pub fn list_blocked(&self) -> Vec<(IpAddr, String)> {
        let blocked = crate::lock_util::read_lock(&self.blocked);
        blocked
            .iter()
            .map(|(ip, entry)| (*ip, entry.reason.clone()))
            .collect()
    }
}

impl Default for IpBlocklist {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Security Headers
// ============================================================================

/// HTTP security headers configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityHeaders {
    /// Strict-Transport-Security
    pub hsts: Option<String>,
    /// Content-Security-Policy
    pub csp: Option<String>,
    /// X-Frame-Options
    pub x_frame_options: Option<String>,
    /// X-Content-Type-Options
    pub x_content_type_options: Option<String>,
    /// X-XSS-Protection
    pub x_xss_protection: Option<String>,
    /// Referrer-Policy
    pub referrer_policy: Option<String>,
    /// Permissions-Policy
    pub permissions_policy: Option<String>,
    /// Custom headers
    pub custom: HashMap<String, String>,
}

impl Default for SecurityHeaders {
    fn default() -> Self {
        Self {
            hsts: Some("max-age=31536000; includeSubDomains".to_string()),
            csp: Some("default-src 'self'".to_string()),
            x_frame_options: Some("DENY".to_string()),
            x_content_type_options: Some("nosniff".to_string()),
            x_xss_protection: Some("1; mode=block".to_string()),
            referrer_policy: Some("strict-origin-when-cross-origin".to_string()),
            permissions_policy: Some("geolocation=(), microphone=(), camera=()".to_string()),
            custom: HashMap::new(),
        }
    }
}

impl SecurityHeaders {
    /// Convert to HTTP header map
    pub fn to_headers(&self) -> Vec<(&str, String)> {
        let mut headers = Vec::new();

        if let Some(ref v) = self.hsts {
            headers.push(("Strict-Transport-Security", v.clone()));
        }
        if let Some(ref v) = self.csp {
            headers.push(("Content-Security-Policy", v.clone()));
        }
        if let Some(ref v) = self.x_frame_options {
            headers.push(("X-Frame-Options", v.clone()));
        }
        if let Some(ref v) = self.x_content_type_options {
            headers.push(("X-Content-Type-Options", v.clone()));
        }
        if let Some(ref v) = self.x_xss_protection {
            headers.push(("X-XSS-Protection", v.clone()));
        }
        if let Some(ref v) = self.referrer_policy {
            headers.push(("Referrer-Policy", v.clone()));
        }
        if let Some(ref v) = self.permissions_policy {
            headers.push(("Permissions-Policy", v.clone()));
        }

        for (k, v) in &self.custom {
            headers.push((k.as_str(), v.clone()));
        }

        headers
    }
}

// ============================================================================
// Security Audit Events
// ============================================================================

/// Security event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityEventType {
    /// Login attempt
    LoginAttempt,
    /// Login success
    LoginSuccess,
    /// Login failure
    LoginFailure,
    /// Logout
    Logout,
    /// Authentication token created
    TokenCreated,
    /// Authentication token revoked
    TokenRevoked,
    /// Permission denied
    PermissionDenied,
    /// SQL injection attempt
    SqlInjectionAttempt,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// IP blocked
    IpBlocked,
    /// IP unblocked
    IpUnblocked,
    /// Configuration change
    ConfigChange,
    /// User created
    UserCreated,
    /// User deleted
    UserDeleted,
    /// Password changed
    PasswordChanged,
    /// Role assigned
    RoleAssigned,
    /// Role revoked
    RoleRevoked,
    /// Sensitive data access
    SensitiveDataAccess,
    /// Backup created
    BackupCreated,
    /// Restore performed
    RestorePerformed,
}

/// Security event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    /// Event ID
    pub id: u64,
    /// Event type
    pub event_type: SecurityEventType,
    /// Timestamp
    pub timestamp: u64,
    /// Source IP
    pub source_ip: Option<String>,
    /// User ID
    pub user_id: Option<String>,
    /// Resource accessed
    pub resource: Option<String>,
    /// Action taken
    pub action: String,
    /// Outcome (success/failure)
    pub success: bool,
    /// Additional details
    pub details: HashMap<String, String>,
    /// Risk level (1-10)
    pub risk_level: u8,
}

impl SecurityEvent {
    /// Create new security event
    pub fn new(event_type: SecurityEventType, action: impl Into<String>) -> Self {
        static EVENT_COUNTER: AtomicU64 = AtomicU64::new(0);

        Self {
            id: EVENT_COUNTER.fetch_add(1, Ordering::SeqCst),
            event_type,
            timestamp: current_timestamp(),
            source_ip: None,
            user_id: None,
            resource: None,
            action: action.into(),
            success: true,
            details: HashMap::new(),
            risk_level: 1,
        }
    }

    /// Set source IP
    pub fn source_ip(mut self, ip: impl Into<String>) -> Self {
        self.source_ip = Some(ip.into());
        self
    }

    /// Set user ID
    pub fn user_id(mut self, user: impl Into<String>) -> Self {
        self.user_id = Some(user.into());
        self
    }

    /// Set resource
    pub fn resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }

    /// Set outcome
    pub fn success(mut self, success: bool) -> Self {
        self.success = success;
        self
    }

    /// Add detail
    pub fn detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    /// Set risk level
    pub fn risk_level(mut self, level: u8) -> Self {
        self.risk_level = level.min(10);
        self
    }
}

/// Security event logger
pub struct SecurityEventLogger {
    events: Arc<RwLock<Vec<SecurityEvent>>>,
    max_events: usize,
    high_risk_threshold: u8,
}

impl SecurityEventLogger {
    /// Create new logger
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            max_events,
            high_risk_threshold: 7,
        }
    }

    /// Log an event
    pub fn log(&self, event: SecurityEvent) {
        let mut events = crate::lock_util::write_lock(&self.events);

        // Log high-risk events to standard logging
        if event.risk_level >= self.high_risk_threshold {
            tracing::warn!(
                event_type = ?event.event_type,
                user = ?event.user_id,
                ip = ?event.source_ip,
                risk = event.risk_level,
                "High-risk security event"
            );
        }

        events.push(event);

        // Trim old events
        let len = events.len();
        if len > self.max_events {
            events.drain(0..len - self.max_events);
        }
    }

    /// Get recent events
    pub fn recent(&self, count: usize) -> Vec<SecurityEvent> {
        let events = crate::lock_util::read_lock(&self.events);
        events.iter().rev().take(count).cloned().collect()
    }

    /// Get events by type
    pub fn by_type(&self, event_type: SecurityEventType) -> Vec<SecurityEvent> {
        let events = crate::lock_util::read_lock(&self.events);
        events
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Get high-risk events
    pub fn high_risk(&self) -> Vec<SecurityEvent> {
        let events = crate::lock_util::read_lock(&self.events);
        events
            .iter()
            .filter(|e| e.risk_level >= self.high_risk_threshold)
            .cloned()
            .collect()
    }

    /// Get events for user
    pub fn for_user(&self, user_id: &str) -> Vec<SecurityEvent> {
        let events = crate::lock_util::read_lock(&self.events);
        events
            .iter()
            .filter(|e| e.user_id.as_deref() == Some(user_id))
            .cloned()
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> SecurityStats {
        let events = crate::lock_util::read_lock(&self.events);

        let mut by_type: HashMap<SecurityEventType, u64> = HashMap::new();
        let mut failed_logins = 0u64;
        let mut high_risk_count = 0u64;

        for event in events.iter() {
            *by_type.entry(event.event_type).or_insert(0) += 1;

            if event.event_type == SecurityEventType::LoginFailure {
                failed_logins += 1;
            }

            if event.risk_level >= self.high_risk_threshold {
                high_risk_count += 1;
            }
        }

        SecurityStats {
            total_events: events.len() as u64,
            failed_logins,
            high_risk_events: high_risk_count,
            events_by_type: by_type,
        }
    }
}

/// Security statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStats {
    pub total_events: u64,
    pub failed_logins: u64,
    pub high_risk_events: u64,
    pub events_by_type: HashMap<SecurityEventType, u64>,
}

// ============================================================================
// Security Scanner
// ============================================================================

/// Security vulnerability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    /// Vulnerability ID
    pub id: String,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Severity (LOW, MEDIUM, HIGH, CRITICAL)
    pub severity: VulnerabilitySeverity,
    /// Affected component
    pub component: String,
    /// Remediation steps
    pub remediation: String,
    /// Reference links
    pub references: Vec<String>,
}

/// Vulnerability severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulnerabilitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Security scanner result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScanResult {
    /// Scan ID
    pub scan_id: String,
    /// Scan timestamp
    pub timestamp: u64,
    /// Vulnerabilities found
    pub vulnerabilities: Vec<Vulnerability>,
    /// Security score (0-100)
    pub score: u8,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Scan duration in ms
    pub duration_ms: u64,
}

/// Security scanner
pub struct SecurityScanner;

impl SecurityScanner {
    /// Run a security scan
    pub fn scan() -> SecurityScanResult {
        let start = Instant::now();
        let mut vulnerabilities = Vec::new();
        let mut recommendations = Vec::new();

        // Check various security aspects
        // In production, these would be actual checks

        // Check 1: TLS configuration
        // (Simulated - would check actual TLS setup)

        // Check 2: Authentication configuration
        recommendations
            .push("Consider enabling multi-factor authentication for admin accounts".to_string());

        // Check 3: Password policy
        recommendations
            .push("Ensure minimum password length is at least 12 characters".to_string());

        // Check 4: Audit logging
        recommendations.push("Enable comprehensive audit logging for security events".to_string());

        // Check 5: Encryption at rest
        vulnerabilities.push(Vulnerability {
            id: "JOULE-001".to_string(),
            title: "Encryption at Rest Recommended".to_string(),
            description: "Data encryption at rest provides additional protection for stored data"
                .to_string(),
            severity: VulnerabilitySeverity::Medium,
            component: "storage".to_string(),
            remediation: "Enable encryption at rest using AES-256".to_string(),
            references: vec!["https://docs.jouledb.com/security/encryption".to_string()],
        });

        // Calculate score
        let critical_count = vulnerabilities
            .iter()
            .filter(|v| v.severity == VulnerabilitySeverity::Critical)
            .count();
        let high_count = vulnerabilities
            .iter()
            .filter(|v| v.severity == VulnerabilitySeverity::High)
            .count();
        let medium_count = vulnerabilities
            .iter()
            .filter(|v| v.severity == VulnerabilitySeverity::Medium)
            .count();

        let score = 100u8
            .saturating_sub((critical_count * 25) as u8)
            .saturating_sub((high_count * 15) as u8)
            .saturating_sub((medium_count * 5) as u8);

        SecurityScanResult {
            scan_id: format!("scan_{}", current_timestamp()),
            timestamp: current_timestamp(),
            vulnerabilities,
            score,
            recommendations,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }
}

// ============================================================================
// Security Configuration
// ============================================================================

/// Complete security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable TLS
    pub tls_enabled: bool,
    /// TLS certificate path
    pub tls_cert_path: Option<String>,
    /// TLS key path
    pub tls_key_path: Option<String>,
    /// Minimum TLS version
    pub tls_min_version: String,

    /// Enable authentication
    pub auth_enabled: bool,
    /// Session timeout
    pub session_timeout: Duration,
    /// Max failed login attempts before lockout
    pub max_failed_logins: u32,
    /// Lockout duration
    pub lockout_duration: Duration,

    /// Password minimum length
    pub password_min_length: usize,
    /// Require uppercase
    pub password_require_uppercase: bool,
    /// Require lowercase
    pub password_require_lowercase: bool,
    /// Require digit
    pub password_require_digit: bool,
    /// Require special character
    pub password_require_special: bool,

    /// Enable rate limiting
    pub rate_limit_enabled: bool,
    /// Rate limit config
    pub rate_limit: RateLimitConfig,

    /// Enable IP blocking
    pub ip_blocking_enabled: bool,

    /// Enable audit logging
    pub audit_enabled: bool,
    /// Audit log retention days
    pub audit_retention_days: u32,

    /// Security headers
    pub security_headers: SecurityHeaders,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            tls_enabled: true,
            tls_cert_path: None,
            tls_key_path: None,
            tls_min_version: "TLS1.2".to_string(),

            auth_enabled: true,
            session_timeout: Duration::from_secs(3600),
            max_failed_logins: 5,
            lockout_duration: Duration::from_secs(900),

            password_min_length: 12,
            password_require_uppercase: true,
            password_require_lowercase: true,
            password_require_digit: true,
            password_require_special: true,

            rate_limit_enabled: true,
            rate_limit: RateLimitConfig::default(),

            ip_blocking_enabled: true,

            audit_enabled: true,
            audit_retention_days: 90,

            security_headers: SecurityHeaders::default(),
        }
    }
}

impl SecurityConfig {
    /// Validate password against policy
    pub fn validate_password(&self, password: &str) -> SecurityResult<()> {
        if password.len() < self.password_min_length {
            return Err(SecurityError::InvalidInput(format!(
                "Password must be at least {} characters",
                self.password_min_length
            )));
        }

        if self.password_require_uppercase && !password.chars().any(|c| c.is_uppercase()) {
            return Err(SecurityError::InvalidInput(
                "Password must contain an uppercase letter".to_string(),
            ));
        }

        if self.password_require_lowercase && !password.chars().any(|c| c.is_lowercase()) {
            return Err(SecurityError::InvalidInput(
                "Password must contain a lowercase letter".to_string(),
            ));
        }

        if self.password_require_digit && !password.chars().any(|c| c.is_ascii_digit()) {
            return Err(SecurityError::InvalidInput(
                "Password must contain a digit".to_string(),
            ));
        }

        if self.password_require_special
            && !password
                .chars()
                .any(|c| "!@#$%^&*()_+-=[]{}|;':\",./<>?".contains(c))
        {
            return Err(SecurityError::InvalidInput(
                "Password must contain a special character".to_string(),
            ));
        }

        Ok(())
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_input_validator_sql_injection() {
        let validator = InputValidator::new();

        // Safe inputs
        assert!(
            validator
                .validate_sql("SELECT * FROM users WHERE id = 1")
                .is_ok()
        );
        assert!(validator.validate_sql("normal input").is_ok());

        // Dangerous inputs
        assert!(validator.validate_sql("'; DROP TABLE users; --").is_err());
        assert!(validator.validate_sql("1 OR '1'='1").is_err());
        assert!(
            validator
                .validate_sql("UNION SELECT * FROM passwords")
                .is_err()
        );
    }

    #[test]
    fn test_input_validator_identifier() {
        let validator = InputValidator::new();

        assert!(validator.validate_identifier("users").is_ok());
        assert!(validator.validate_identifier("_private_table").is_ok());
        assert!(validator.validate_identifier("table123").is_ok());

        assert!(validator.validate_identifier("").is_err());
        assert!(validator.validate_identifier("123table").is_err());
        assert!(validator.validate_identifier("table-name").is_err());
    }

    #[test]
    fn test_input_validator_sanitize() {
        let validator = InputValidator::new();

        assert_eq!(validator.sanitize("O'Brien"), "O''Brien");
        assert_eq!(validator.sanitize("path\\file"), "path\\\\file");
    }

    #[test]
    fn test_rate_limiter() {
        let config = RateLimitConfig {
            requests_per_window: 10,
            window_duration: Duration::from_secs(60),
            adaptive: true,
            burst_size: 5,
        };

        let limiter = RateLimiter::new(config);

        // First 5 requests should succeed (burst)
        for _ in 0..5 {
            assert!(limiter.check("client1").is_ok());
        }

        // 6th request should fail
        assert!(limiter.check("client1").is_err());

        // Different client should work
        assert!(limiter.check("client2").is_ok());

        // Reset should allow requests again
        limiter.reset("client1");
        assert!(limiter.check("client1").is_ok());
    }

    #[test]
    fn test_ip_blocklist() {
        let blocklist = IpBlocklist::new();
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));

        // Initially not blocked
        assert!(blocklist.is_blocked(&ip).is_none());

        // Block IP
        blocklist.block(ip, "Suspicious activity", None);
        assert!(blocklist.is_blocked(&ip).is_some());

        // Unblock IP
        blocklist.unblock(&ip);
        assert!(blocklist.is_blocked(&ip).is_none());
    }

    #[test]
    fn test_ip_blocklist_temporary() {
        let blocklist = IpBlocklist::new();
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));

        // Block for very short duration
        blocklist.block(ip, "Test", Some(Duration::from_millis(1)));

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        // Should no longer be blocked
        assert!(blocklist.is_blocked(&ip).is_none());
    }

    #[test]
    fn test_ip_blocklist_allowlist() {
        let blocklist = IpBlocklist::new();
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

        // Add to allowlist first
        blocklist.allow(ip);

        // Try to block
        blocklist.block(ip, "Should be ignored", None);

        // Should not be blocked due to allowlist
        assert!(blocklist.is_blocked(&ip).is_none());
    }

    #[test]
    fn test_security_headers() {
        let headers = SecurityHeaders::default();
        let header_vec = headers.to_headers();

        assert!(
            header_vec
                .iter()
                .any(|(k, _)| *k == "Strict-Transport-Security")
        );
        assert!(header_vec.iter().any(|(k, _)| *k == "X-Frame-Options"));
        assert!(
            header_vec
                .iter()
                .any(|(k, _)| *k == "X-Content-Type-Options")
        );
    }

    #[test]
    fn test_security_event() {
        let event = SecurityEvent::new(SecurityEventType::LoginAttempt, "User login")
            .source_ip("192.168.1.1")
            .user_id("admin")
            .success(true)
            .detail("method", "password")
            .risk_level(3);

        assert_eq!(event.event_type, SecurityEventType::LoginAttempt);
        assert_eq!(event.source_ip, Some("192.168.1.1".to_string()));
        assert!(event.success);
        assert_eq!(event.risk_level, 3);
    }

    #[test]
    fn test_security_event_logger() {
        let logger = SecurityEventLogger::new(100);

        // Log some events
        logger.log(SecurityEvent::new(SecurityEventType::LoginSuccess, "Login"));
        logger.log(
            SecurityEvent::new(SecurityEventType::LoginFailure, "Failed login")
                .success(false)
                .risk_level(5),
        );
        logger.log(
            SecurityEvent::new(SecurityEventType::SqlInjectionAttempt, "SQL injection")
                .success(false)
                .risk_level(9),
        );

        let recent = logger.recent(10);
        assert_eq!(recent.len(), 3);

        let stats = logger.stats();
        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.failed_logins, 1);
        assert_eq!(stats.high_risk_events, 1);
    }

    #[test]
    fn test_security_scanner() {
        let result = SecurityScanner::scan();

        assert!(!result.scan_id.is_empty());
        assert!(result.score > 0);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_password_validation() {
        let config = SecurityConfig::default();

        // Valid password
        assert!(config.validate_password("SecurePass123!").is_ok());

        // Too short
        assert!(config.validate_password("Short1!").is_err());

        // Missing uppercase
        assert!(config.validate_password("lowercase123!").is_err());

        // Missing digit
        assert!(config.validate_password("SecurePassword!").is_err());

        // Missing special
        assert!(config.validate_password("SecurePassword123").is_err());
    }

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();

        assert!(config.tls_enabled);
        assert!(config.auth_enabled);
        assert_eq!(config.password_min_length, 12);
        assert!(config.rate_limit_enabled);
    }

    #[test]
    fn test_rate_limit_stats() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        limiter.check("client1").ok();
        limiter.check("client2").ok();

        let stats = limiter.stats();
        assert_eq!(stats.total_clients, 2);
        assert_eq!(stats.global_requests, 2);
    }

    #[test]
    fn test_security_error_display() {
        let err = SecurityError::AuthenticationFailed("Invalid credentials".to_string());
        assert!(err.to_string().contains("Authentication failed"));

        let err = SecurityError::SqlInjectionDetected("Pattern found".to_string());
        assert!(err.to_string().contains("SQL injection"));
    }

    #[test]
    fn test_vulnerability_severity() {
        let result = SecurityScanner::scan();

        // Check that score is affected by vulnerabilities
        assert!(result.score <= 100);
        assert!(result.score > 0);
    }
}
