//! Sandboxed execution environment — resource limits (memory, CPU cycles,
//! stack depth), capability-based permissions, syscall filtering, execution
//! timeout, resource usage tracking, sandbox policy.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Duration;

// ── Syscall ────────────────────────────────────────────────────────────────

/// System calls that sandboxed code might attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Syscall {
    FileRead,
    FileWrite,
    FileDelete,
    NetworkConnect,
    NetworkListen,
    NetworkDns,
    ProcessSpawn,
    ProcessSignal,
    MemoryMap,
    ClockRead,
    EnvRead,
    RandomRead,
}

impl fmt::Display for Syscall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::FileRead => "file_read",
            Self::FileWrite => "file_write",
            Self::FileDelete => "file_delete",
            Self::NetworkConnect => "net_connect",
            Self::NetworkListen => "net_listen",
            Self::NetworkDns => "net_dns",
            Self::ProcessSpawn => "proc_spawn",
            Self::ProcessSignal => "proc_signal",
            Self::MemoryMap => "mem_map",
            Self::ClockRead => "clock_read",
            Self::EnvRead => "env_read",
            Self::RandomRead => "random_read",
        };
        write!(f, "{name}")
    }
}

// ── Permission ─────────────────────────────────────────────────────────────

/// A capability-based permission.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permission {
    pub domain: String,
    pub action: String,
    pub resource: Option<String>,
}

impl Permission {
    pub fn new(
        domain: impl Into<String>,
        action: impl Into<String>,
        resource: Option<String>,
    ) -> Self {
        Self {
            domain: domain.into(),
            action: action.into(),
            resource,
        }
    }

    /// Check if this permission covers the requested permission.
    /// A permission with no resource is a wildcard for that domain+action.
    pub fn covers(&self, requested: &Permission) -> bool {
        if self.domain != requested.domain || self.action != requested.action {
            return false;
        }
        match (&self.resource, &requested.resource) {
            (None, _) => true, // wildcard
            (Some(have), Some(want)) => have == want,
            (Some(_), None) => false,
        }
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.domain, self.action)?;
        if let Some(r) = &self.resource {
            write!(f, "/{r}")?;
        }
        Ok(())
    }
}

// ── Resource Limits ────────────────────────────────────────────────────────

/// Resource limits for a sandbox.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceLimits {
    /// Maximum memory in bytes.
    pub max_memory_bytes: u64,
    /// Maximum CPU cycles.
    pub max_cpu_cycles: u64,
    /// Maximum stack depth.
    pub max_stack_depth: u32,
    /// Execution timeout.
    pub timeout: Duration,
    /// Maximum number of syscalls.
    pub max_syscalls: u64,
    /// Maximum open file descriptors.
    pub max_file_descriptors: u32,
    /// Maximum network connections.
    pub max_connections: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024, // 64 MiB
            max_cpu_cycles: 10_000_000,
            max_stack_depth: 256,
            timeout: Duration::from_secs(30),
            max_syscalls: 10_000,
            max_file_descriptors: 16,
            max_connections: 4,
        }
    }
}

// ── Resource Usage ─────────────────────────────────────────────────────────

/// Tracked resource usage.
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub memory_bytes: u64,
    pub cpu_cycles: u64,
    pub stack_depth: u32,
    pub peak_memory_bytes: u64,
    pub peak_stack_depth: u32,
    pub syscall_count: u64,
    pub file_descriptors: u32,
    pub connections: u32,
    pub elapsed: Duration,
}

impl ResourceUsage {
    /// Fraction of memory used (0.0 – 1.0).
    pub fn memory_utilization(&self, limits: &ResourceLimits) -> f64 {
        if limits.max_memory_bytes == 0 {
            return 0.0;
        }
        self.memory_bytes as f64 / limits.max_memory_bytes as f64
    }

    /// Fraction of CPU budget consumed.
    pub fn cpu_utilization(&self, limits: &ResourceLimits) -> f64 {
        if limits.max_cpu_cycles == 0 {
            return 0.0;
        }
        self.cpu_cycles as f64 / limits.max_cpu_cycles as f64
    }
}

// ── Sandbox Violation ──────────────────────────────────────────────────────

/// A policy violation.
#[derive(Debug, Clone, PartialEq)]
pub enum Violation {
    MemoryExceeded { used: u64, limit: u64 },
    CpuExceeded { used: u64, limit: u64 },
    StackDepthExceeded { depth: u32, limit: u32 },
    Timeout { elapsed: Duration, limit: Duration },
    SyscallDenied(Syscall),
    PermissionDenied(Permission),
    SyscallLimitExceeded { count: u64, limit: u64 },
    FileDescriptorLimit { count: u32, limit: u32 },
    ConnectionLimit { count: u32, limit: u32 },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryExceeded { used, limit } => {
                write!(f, "memory exceeded: {used} > {limit}")
            }
            Self::CpuExceeded { used, limit } => {
                write!(f, "CPU exceeded: {used} > {limit}")
            }
            Self::StackDepthExceeded { depth, limit } => {
                write!(f, "stack depth exceeded: {depth} > {limit}")
            }
            Self::Timeout { elapsed, limit } => {
                write!(f, "timeout: {:?} > {:?}", elapsed, limit)
            }
            Self::SyscallDenied(s) => write!(f, "syscall denied: {s}"),
            Self::PermissionDenied(p) => write!(f, "permission denied: {p}"),
            Self::SyscallLimitExceeded { count, limit } => {
                write!(f, "syscall limit exceeded: {count} > {limit}")
            }
            Self::FileDescriptorLimit { count, limit } => {
                write!(f, "FD limit: {count} > {limit}")
            }
            Self::ConnectionLimit { count, limit } => {
                write!(f, "connection limit: {count} > {limit}")
            }
        }
    }
}

// ── Sandbox Policy ─────────────────────────────────────────────────────────

/// Policy mode for the syscall filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// Only explicitly allowed syscalls are permitted.
    Allowlist,
    /// Only explicitly denied syscalls are blocked.
    Denylist,
}

/// A sandbox policy combining limits, permissions, and syscall filtering.
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    pub name: String,
    pub limits: ResourceLimits,
    pub permissions: HashSet<Permission>,
    pub syscall_filter_mode: FilterMode,
    pub syscall_filter: HashSet<Syscall>,
}

impl SandboxPolicy {
    /// Create a restrictive policy that allows nothing by default.
    pub fn restrictive(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            limits: ResourceLimits::default(),
            permissions: HashSet::new(),
            syscall_filter_mode: FilterMode::Allowlist,
            syscall_filter: HashSet::new(),
        }
    }

    /// Create a permissive policy that allows everything by default.
    pub fn permissive(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            limits: ResourceLimits {
                max_memory_bytes: 512 * 1024 * 1024,
                max_cpu_cycles: 100_000_000,
                max_stack_depth: 1024,
                timeout: Duration::from_secs(300),
                max_syscalls: 1_000_000,
                max_file_descriptors: 256,
                max_connections: 64,
            },
            permissions: HashSet::new(),
            syscall_filter_mode: FilterMode::Denylist,
            syscall_filter: HashSet::new(),
        }
    }

    /// Allow a syscall.
    pub fn allow_syscall(&mut self, syscall: Syscall) {
        match self.syscall_filter_mode {
            FilterMode::Allowlist => {
                self.syscall_filter.insert(syscall);
            }
            FilterMode::Denylist => {
                self.syscall_filter.remove(&syscall);
            }
        }
    }

    /// Deny a syscall.
    pub fn deny_syscall(&mut self, syscall: Syscall) {
        match self.syscall_filter_mode {
            FilterMode::Allowlist => {
                self.syscall_filter.remove(&syscall);
            }
            FilterMode::Denylist => {
                self.syscall_filter.insert(syscall);
            }
        }
    }

    /// Check if a syscall is permitted.
    pub fn is_syscall_allowed(&self, syscall: Syscall) -> bool {
        match self.syscall_filter_mode {
            FilterMode::Allowlist => self.syscall_filter.contains(&syscall),
            FilterMode::Denylist => !self.syscall_filter.contains(&syscall),
        }
    }

    /// Grant a permission.
    pub fn grant(&mut self, perm: Permission) {
        self.permissions.insert(perm);
    }

    /// Revoke a permission.
    pub fn revoke(&mut self, perm: &Permission) {
        self.permissions.remove(perm);
    }

    /// Check if a permission is granted.
    pub fn has_permission(&self, requested: &Permission) -> bool {
        self.permissions.iter().any(|p| p.covers(requested))
    }
}

// ── Audit Event ────────────────────────────────────────────────────────────

/// An event logged by the sandbox.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    pub timestamp_ms: u64,
    pub kind: AuditKind,
    pub detail: String,
}

/// Kind of audit event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditKind {
    SyscallAllowed,
    SyscallDenied,
    PermissionGranted,
    PermissionDenied,
    ResourceAllocated,
    ResourceReleased,
    LimitExceeded,
    PolicyViolation,
}

// ── Sandbox ────────────────────────────────────────────────────────────────

/// A sandboxed execution environment.
pub struct Sandbox {
    policy: SandboxPolicy,
    usage: ResourceUsage,
    violations: Vec<Violation>,
    audit_log: Vec<AuditEvent>,
    timestamp_counter: u64,
    active: bool,
    metadata: HashMap<String, String>,
}

impl Sandbox {
    /// Create a new sandbox from a policy.
    pub fn new(policy: SandboxPolicy) -> Self {
        Self {
            policy,
            usage: ResourceUsage::default(),
            violations: Vec::new(),
            audit_log: Vec::new(),
            timestamp_counter: 0,
            active: true,
            metadata: HashMap::new(),
        }
    }

    /// Get the policy.
    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }

    /// Get current resource usage.
    pub fn usage(&self) -> &ResourceUsage {
        &self.usage
    }

    /// Check if the sandbox is still active (not terminated by violation).
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// All violations so far.
    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }

    /// Audit log.
    pub fn audit_log(&self) -> &[AuditEvent] {
        &self.audit_log
    }

    /// Set metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get metadata.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Record an audit event.
    fn audit(&mut self, kind: AuditKind, detail: impl Into<String>) {
        self.timestamp_counter += 1;
        self.audit_log.push(AuditEvent {
            timestamp_ms: self.timestamp_counter,
            kind,
            detail: detail.into(),
        });
    }

    // ── Resource tracking ──────────────────────────────────────────────

    /// Allocate memory (track usage). Returns error if limit exceeded.
    pub fn allocate_memory(&mut self, bytes: u64) -> Result<(), Violation> {
        let new_usage = self.usage.memory_bytes.saturating_add(bytes);
        if new_usage > self.policy.limits.max_memory_bytes {
            let v = Violation::MemoryExceeded {
                used: new_usage,
                limit: self.policy.limits.max_memory_bytes,
            };
            self.violations.push(v.clone());
            self.audit(AuditKind::LimitExceeded, format!("memory: {new_usage}"));
            self.active = false;
            return Err(v);
        }
        self.usage.memory_bytes = new_usage;
        if new_usage > self.usage.peak_memory_bytes {
            self.usage.peak_memory_bytes = new_usage;
        }
        self.audit(
            AuditKind::ResourceAllocated,
            format!("memory: +{bytes} = {new_usage}"),
        );
        Ok(())
    }

    /// Free memory.
    pub fn free_memory(&mut self, bytes: u64) {
        self.usage.memory_bytes = self.usage.memory_bytes.saturating_sub(bytes);
        self.audit(
            AuditKind::ResourceReleased,
            format!("memory: -{bytes} = {}", self.usage.memory_bytes),
        );
    }

    /// Consume CPU cycles.
    pub fn consume_cycles(&mut self, cycles: u64) -> Result<(), Violation> {
        let new_usage = self.usage.cpu_cycles.saturating_add(cycles);
        if new_usage > self.policy.limits.max_cpu_cycles {
            let v = Violation::CpuExceeded {
                used: new_usage,
                limit: self.policy.limits.max_cpu_cycles,
            };
            self.violations.push(v.clone());
            self.active = false;
            return Err(v);
        }
        self.usage.cpu_cycles = new_usage;
        Ok(())
    }

    /// Push stack frame.
    pub fn push_stack(&mut self) -> Result<(), Violation> {
        let new_depth = self.usage.stack_depth + 1;
        if new_depth > self.policy.limits.max_stack_depth {
            let v = Violation::StackDepthExceeded {
                depth: new_depth,
                limit: self.policy.limits.max_stack_depth,
            };
            self.violations.push(v.clone());
            self.active = false;
            return Err(v);
        }
        self.usage.stack_depth = new_depth;
        if new_depth > self.usage.peak_stack_depth {
            self.usage.peak_stack_depth = new_depth;
        }
        Ok(())
    }

    /// Pop stack frame.
    pub fn pop_stack(&mut self) {
        self.usage.stack_depth = self.usage.stack_depth.saturating_sub(1);
    }

    /// Update elapsed time. Returns error if timeout exceeded.
    pub fn update_elapsed(&mut self, elapsed: Duration) -> Result<(), Violation> {
        self.usage.elapsed = elapsed;
        if elapsed > self.policy.limits.timeout {
            let v = Violation::Timeout {
                elapsed,
                limit: self.policy.limits.timeout,
            };
            self.violations.push(v.clone());
            self.active = false;
            return Err(v);
        }
        Ok(())
    }

    // ── Syscall filtering ──────────────────────────────────────────────

    /// Attempt a syscall. Returns Ok if allowed.
    pub fn attempt_syscall(&mut self, syscall: Syscall) -> Result<(), Violation> {
        self.usage.syscall_count += 1;
        if self.usage.syscall_count > self.policy.limits.max_syscalls {
            let v = Violation::SyscallLimitExceeded {
                count: self.usage.syscall_count,
                limit: self.policy.limits.max_syscalls,
            };
            self.violations.push(v.clone());
            self.active = false;
            return Err(v);
        }

        if self.policy.is_syscall_allowed(syscall) {
            self.audit(AuditKind::SyscallAllowed, syscall.to_string());
            Ok(())
        } else {
            let v = Violation::SyscallDenied(syscall);
            self.violations.push(v.clone());
            self.audit(AuditKind::SyscallDenied, syscall.to_string());
            Err(v)
        }
    }

    // ── Permission checking ────────────────────────────────────────────

    /// Check a permission. Returns Ok if granted.
    pub fn check_permission(&mut self, perm: &Permission) -> Result<(), Violation> {
        if self.policy.has_permission(perm) {
            self.audit(AuditKind::PermissionGranted, perm.to_string());
            Ok(())
        } else {
            let v = Violation::PermissionDenied(perm.clone());
            self.violations.push(v.clone());
            self.audit(AuditKind::PermissionDenied, perm.to_string());
            Err(v)
        }
    }

    // ── Open file descriptors ──────────────────────────────────────────

    /// Open a file descriptor.
    pub fn open_fd(&mut self) -> Result<(), Violation> {
        let new_count = self.usage.file_descriptors + 1;
        if new_count > self.policy.limits.max_file_descriptors {
            let v = Violation::FileDescriptorLimit {
                count: new_count,
                limit: self.policy.limits.max_file_descriptors,
            };
            self.violations.push(v.clone());
            return Err(v);
        }
        self.usage.file_descriptors = new_count;
        Ok(())
    }

    /// Close a file descriptor.
    pub fn close_fd(&mut self) {
        self.usage.file_descriptors = self.usage.file_descriptors.saturating_sub(1);
    }

    /// Open a network connection.
    pub fn open_connection(&mut self) -> Result<(), Violation> {
        let new_count = self.usage.connections + 1;
        if new_count > self.policy.limits.max_connections {
            let v = Violation::ConnectionLimit {
                count: new_count,
                limit: self.policy.limits.max_connections,
            };
            self.violations.push(v.clone());
            return Err(v);
        }
        self.usage.connections = new_count;
        Ok(())
    }

    /// Close a network connection.
    pub fn close_connection(&mut self) {
        self.usage.connections = self.usage.connections.saturating_sub(1);
    }

    /// Terminate the sandbox.
    pub fn terminate(&mut self) {
        self.active = false;
        self.audit(AuditKind::PolicyViolation, "sandbox terminated".to_string());
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_policy() -> SandboxPolicy {
        let mut policy = SandboxPolicy::restrictive("test");
        policy.limits.max_memory_bytes = 1024;
        policy.limits.max_cpu_cycles = 100;
        policy.limits.max_stack_depth = 10;
        policy.limits.max_syscalls = 5;
        policy.limits.max_file_descriptors = 3;
        policy.limits.max_connections = 2;
        policy.limits.timeout = Duration::from_secs(1);
        policy
    }

    #[test]
    fn sandbox_starts_active() {
        let sb = Sandbox::new(test_policy());
        assert!(sb.is_active());
        assert!(sb.violations().is_empty());
    }

    #[test]
    fn memory_allocation_and_free() {
        let mut sb = Sandbox::new(test_policy());
        assert!(sb.allocate_memory(512).is_ok());
        assert_eq!(sb.usage().memory_bytes, 512);
        sb.free_memory(256);
        assert_eq!(sb.usage().memory_bytes, 256);
    }

    #[test]
    fn memory_limit_exceeded() {
        let mut sb = Sandbox::new(test_policy());
        let result = sb.allocate_memory(2000);
        assert!(result.is_err());
        assert!(!sb.is_active());
    }

    #[test]
    fn cpu_cycle_tracking() {
        let mut sb = Sandbox::new(test_policy());
        assert!(sb.consume_cycles(50).is_ok());
        assert!(sb.consume_cycles(50).is_ok());
        assert!(sb.consume_cycles(1).is_err());
    }

    #[test]
    fn stack_depth_limit() {
        let mut sb = Sandbox::new(test_policy());
        for _ in 0..10 {
            assert!(sb.push_stack().is_ok());
        }
        assert!(sb.push_stack().is_err());
        sb.pop_stack();
        assert!(sb.push_stack().is_ok());
    }

    #[test]
    fn timeout_detection() {
        let mut sb = Sandbox::new(test_policy());
        assert!(sb.update_elapsed(Duration::from_millis(500)).is_ok());
        assert!(sb.update_elapsed(Duration::from_secs(2)).is_err());
    }

    #[test]
    fn syscall_allowlist() {
        let mut policy = test_policy();
        policy.allow_syscall(Syscall::ClockRead);
        policy.allow_syscall(Syscall::RandomRead);
        let mut sb = Sandbox::new(policy);
        assert!(sb.attempt_syscall(Syscall::ClockRead).is_ok());
        assert!(sb.attempt_syscall(Syscall::FileRead).is_err());
    }

    #[test]
    fn syscall_denylist() {
        let mut policy = SandboxPolicy::permissive("test");
        policy.deny_syscall(Syscall::ProcessSpawn);
        let mut sb = Sandbox::new(policy);
        assert!(sb.attempt_syscall(Syscall::FileRead).is_ok());
        assert!(sb.attempt_syscall(Syscall::ProcessSpawn).is_err());
    }

    #[test]
    fn syscall_limit() {
        let mut policy = test_policy();
        policy.allow_syscall(Syscall::ClockRead);
        policy.limits.max_syscalls = 3;
        let mut sb = Sandbox::new(policy);
        for _ in 0..3 {
            assert!(sb.attempt_syscall(Syscall::ClockRead).is_ok());
        }
        assert!(sb.attempt_syscall(Syscall::ClockRead).is_err());
    }

    #[test]
    fn permission_grant_and_check() {
        let mut policy = test_policy();
        policy.grant(Permission::new("fs", "read", Some("/tmp".to_string())));
        let mut sb = Sandbox::new(policy);
        let perm = Permission::new("fs", "read", Some("/tmp".to_string()));
        assert!(sb.check_permission(&perm).is_ok());
        let denied = Permission::new("fs", "write", Some("/tmp".to_string()));
        assert!(sb.check_permission(&denied).is_err());
    }

    #[test]
    fn wildcard_permission() {
        let mut policy = test_policy();
        // Wildcard: fs:read with no specific resource.
        policy.grant(Permission::new("fs", "read", None));
        let mut sb = Sandbox::new(policy);
        let perm = Permission::new("fs", "read", Some("/any/path".to_string()));
        assert!(sb.check_permission(&perm).is_ok());
    }

    #[test]
    fn file_descriptor_limit() {
        let mut sb = Sandbox::new(test_policy());
        for _ in 0..3 {
            assert!(sb.open_fd().is_ok());
        }
        assert!(sb.open_fd().is_err());
        sb.close_fd();
        assert!(sb.open_fd().is_ok());
    }

    #[test]
    fn connection_limit() {
        let mut sb = Sandbox::new(test_policy());
        assert!(sb.open_connection().is_ok());
        assert!(sb.open_connection().is_ok());
        assert!(sb.open_connection().is_err());
        sb.close_connection();
        assert!(sb.open_connection().is_ok());
    }

    #[test]
    fn audit_log_tracks_events() {
        let mut policy = test_policy();
        policy.allow_syscall(Syscall::ClockRead);
        let mut sb = Sandbox::new(policy);
        sb.attempt_syscall(Syscall::ClockRead).unwrap();
        sb.allocate_memory(100).unwrap();
        assert!(sb.audit_log().len() >= 2);
    }

    #[test]
    fn terminate_deactivates() {
        let mut sb = Sandbox::new(test_policy());
        assert!(sb.is_active());
        sb.terminate();
        assert!(!sb.is_active());
    }

    #[test]
    fn peak_tracking() {
        let mut sb = Sandbox::new(test_policy());
        sb.allocate_memory(500).unwrap();
        sb.free_memory(300);
        sb.allocate_memory(200).unwrap();
        assert_eq!(sb.usage().peak_memory_bytes, 500);
    }

    #[test]
    fn resource_utilization() {
        let mut sb = Sandbox::new(test_policy());
        sb.allocate_memory(512).unwrap();
        sb.consume_cycles(50).unwrap();
        let util = sb.usage().memory_utilization(&sb.policy().limits);
        assert!((util - 0.5).abs() < 0.01);
        let cpu_util = sb.usage().cpu_utilization(&sb.policy().limits);
        assert!((cpu_util - 0.5).abs() < 0.01);
    }

    #[test]
    fn metadata() {
        let mut sb = Sandbox::new(test_policy());
        sb.set_metadata("author", "test");
        assert_eq!(sb.get_metadata("author"), Some("test"));
        assert_eq!(sb.get_metadata("missing"), None);
    }

    #[test]
    fn violation_display() {
        let v = Violation::SyscallDenied(Syscall::FileWrite);
        assert!(v.to_string().contains("file_write"));
    }

    #[test]
    fn permission_covers() {
        let wildcard = Permission::new("net", "connect", None);
        let specific = Permission::new("net", "connect", Some("example.com".to_string()));
        assert!(wildcard.covers(&specific));
        assert!(!specific.covers(&wildcard));
    }

    #[test]
    fn policy_revoke() {
        let mut policy = test_policy();
        let perm = Permission::new("fs", "read", None);
        policy.grant(perm.clone());
        assert!(policy.has_permission(&perm));
        policy.revoke(&perm);
        assert!(!policy.has_permission(&perm));
    }

    #[test]
    fn multiple_violations_tracked() {
        let mut sb = Sandbox::new(test_policy());
        let _ = sb.attempt_syscall(Syscall::FileRead); // denied
        let _ = sb.attempt_syscall(Syscall::FileWrite); // denied
        assert_eq!(sb.violations().len(), 2);
    }
}
