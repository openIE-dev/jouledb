//! Database migration runner — migration versioning, up/down scripts, migration
//! status tracking, dry-run mode, rollback, checksum verification.
//!
//! Replaces JS migration tools (knex migrate, sequelize-cli, db-migrate,
//! flyway-js) with a pure-Rust migration runner that tracks every migration
//! step with energy awareness.

use std::collections::BTreeMap;

// ── Errors ──────────────────────────────────────────────────────

/// Migration errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationError {
    /// Migration not found.
    NotFound(u64),
    /// Duplicate migration version.
    DuplicateVersion(u64),
    /// Checksum mismatch.
    ChecksumMismatch { version: u64, expected: u64, actual: u64 },
    /// Migration already applied.
    AlreadyApplied(u64),
    /// Cannot rollback — migration not applied.
    NotApplied(u64),
    /// Empty migration script.
    EmptyScript { version: u64, direction: &'static str },
    /// Out-of-order migration.
    OutOfOrder { version: u64, last_applied: u64 },
    /// Script execution failure.
    ExecutionFailed { version: u64, reason: String },
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(v) => write!(f, "migration not found: v{v}"),
            Self::DuplicateVersion(v) => write!(f, "duplicate migration: v{v}"),
            Self::ChecksumMismatch { version, expected, actual } => {
                write!(f, "checksum mismatch v{version}: expected {expected}, got {actual}")
            }
            Self::AlreadyApplied(v) => write!(f, "migration already applied: v{v}"),
            Self::NotApplied(v) => write!(f, "migration not applied: v{v}"),
            Self::EmptyScript { version, direction } => {
                write!(f, "empty {direction} script for v{version}")
            }
            Self::OutOfOrder { version, last_applied } => {
                write!(f, "out of order: v{version} < last applied v{last_applied}")
            }
            Self::ExecutionFailed { version, reason } => {
                write!(f, "execution failed v{version}: {reason}")
            }
        }
    }
}

// ── Types ───────────────────────────────────────────────────────

/// Migration status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationStatus {
    Pending,
    Applied,
    RolledBack,
    Failed,
}

/// A single migration definition.
#[derive(Debug, Clone)]
pub struct Migration {
    pub version: u64,
    pub name: String,
    pub up_script: String,
    pub down_script: String,
    pub checksum: u64,
}

impl Migration {
    pub fn new(
        version: u64,
        name: impl Into<String>,
        up_script: impl Into<String>,
        down_script: impl Into<String>,
    ) -> Self {
        let up = up_script.into();
        let down = down_script.into();
        let checksum = compute_checksum(&up, &down);
        Self {
            version,
            name: name.into(),
            up_script: up,
            down_script: down,
            checksum,
        }
    }
}

/// Record of an applied migration.
#[derive(Debug, Clone)]
pub struct AppliedMigration {
    pub version: u64,
    pub name: String,
    pub status: MigrationStatus,
    pub checksum: u64,
    pub applied_at: Option<u64>,
    pub rolled_back_at: Option<u64>,
    pub execution_time_ms: u64,
}

/// Result of a dry-run migration step.
#[derive(Debug, Clone)]
pub struct DryRunStep {
    pub version: u64,
    pub name: String,
    pub direction: MigrationDirection,
    pub script: String,
}

/// Migration direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationDirection {
    Up,
    Down,
}

/// Execution callback result.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

impl ExecutionResult {
    pub fn ok(time_ms: u64) -> Self {
        Self { success: true, execution_time_ms: time_ms, error: None }
    }

    pub fn fail(time_ms: u64, error: impl Into<String>) -> Self {
        Self { success: false, execution_time_ms: time_ms, error: Some(error.into()) }
    }
}

/// Checksum computation (simple FNV-1a style hash for determinism).
fn compute_checksum(up: &str, down: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in up.bytes().chain(down.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ── Migration Runner ────────────────────────────────────────────

/// Script executor function type.
pub type ScriptExecutor = Box<dyn Fn(&str) -> ExecutionResult>;

/// Migration runner with registration, execution, and rollback.
pub struct MigrationRunner {
    migrations: BTreeMap<u64, Migration>,
    applied: BTreeMap<u64, AppliedMigration>,
    allow_out_of_order: bool,
    executor: Option<ScriptExecutor>,
    /// Monotonic timestamp counter for testing.
    timestamp: u64,
}

impl MigrationRunner {
    pub fn new() -> Self {
        Self {
            migrations: BTreeMap::new(),
            applied: BTreeMap::new(),
            allow_out_of_order: false,
            executor: None,
            timestamp: 1000,
        }
    }

    pub fn allow_out_of_order(mut self, allow: bool) -> Self {
        self.allow_out_of_order = allow;
        self
    }

    /// Set a custom script executor.
    pub fn set_executor(&mut self, executor: ScriptExecutor) {
        self.executor = Some(executor);
    }

    fn next_timestamp(&mut self) -> u64 {
        self.timestamp += 1;
        self.timestamp
    }

    /// Register a migration.
    pub fn register(&mut self, migration: Migration) -> Result<(), MigrationError> {
        if self.migrations.contains_key(&migration.version) {
            return Err(MigrationError::DuplicateVersion(migration.version));
        }
        self.migrations.insert(migration.version, migration);
        Ok(())
    }

    /// Get all registered migration versions (sorted).
    pub fn registered_versions(&self) -> Vec<u64> {
        self.migrations.keys().copied().collect()
    }

    /// Get the status of a migration.
    pub fn status(&self, version: u64) -> Result<MigrationStatus, MigrationError> {
        if !self.migrations.contains_key(&version) {
            return Err(MigrationError::NotFound(version));
        }
        Ok(self
            .applied
            .get(&version)
            .map(|a| a.status)
            .unwrap_or(MigrationStatus::Pending))
    }

    /// Get all pending migrations (sorted by version).
    pub fn pending(&self) -> Vec<u64> {
        self.migrations
            .keys()
            .filter(|v| {
                !self
                    .applied
                    .get(v)
                    .is_some_and(|a| a.status == MigrationStatus::Applied)
            })
            .copied()
            .collect()
    }

    /// Get all applied migration versions (sorted).
    pub fn applied_versions(&self) -> Vec<u64> {
        self.applied
            .values()
            .filter(|a| a.status == MigrationStatus::Applied)
            .map(|a| a.version)
            .collect()
    }

    /// Last applied version.
    pub fn last_applied_version(&self) -> Option<u64> {
        self.applied
            .values()
            .filter(|a| a.status == MigrationStatus::Applied)
            .map(|a| a.version)
            .max()
    }

    /// Verify checksum of an applied migration.
    pub fn verify_checksum(&self, version: u64) -> Result<bool, MigrationError> {
        let migration = self
            .migrations
            .get(&version)
            .ok_or(MigrationError::NotFound(version))?;
        let applied = self
            .applied
            .get(&version)
            .ok_or(MigrationError::NotApplied(version))?;
        Ok(migration.checksum == applied.checksum)
    }

    /// Verify all applied migrations' checksums.
    pub fn verify_all_checksums(&self) -> Vec<MigrationError> {
        let mut errors = Vec::new();
        for (version, applied) in &self.applied {
            if applied.status != MigrationStatus::Applied {
                continue;
            }
            if let Some(migration) = self.migrations.get(version) {
                if migration.checksum != applied.checksum {
                    errors.push(MigrationError::ChecksumMismatch {
                        version: *version,
                        expected: migration.checksum,
                        actual: applied.checksum,
                    });
                }
            }
        }
        errors
    }

    /// Apply a single migration (up).
    pub fn apply(&mut self, version: u64) -> Result<AppliedMigration, MigrationError> {
        let migration = self
            .migrations
            .get(&version)
            .ok_or(MigrationError::NotFound(version))?
            .clone();

        if self
            .applied
            .get(&version)
            .is_some_and(|a| a.status == MigrationStatus::Applied)
        {
            return Err(MigrationError::AlreadyApplied(version));
        }

        if migration.up_script.trim().is_empty() {
            return Err(MigrationError::EmptyScript { version, direction: "up" });
        }

        // Check ordering.
        if !self.allow_out_of_order {
            if let Some(last) = self.last_applied_version() {
                if version < last {
                    return Err(MigrationError::OutOfOrder {
                        version,
                        last_applied: last,
                    });
                }
            }
        }

        // Execute.
        let result = if let Some(executor) = &self.executor {
            executor(&migration.up_script)
        } else {
            ExecutionResult::ok(1)
        };

        let ts = self.next_timestamp();

        if !result.success {
            let applied = AppliedMigration {
                version,
                name: migration.name.clone(),
                status: MigrationStatus::Failed,
                checksum: migration.checksum,
                applied_at: Some(ts),
                rolled_back_at: None,
                execution_time_ms: result.execution_time_ms,
            };
            self.applied.insert(version, applied.clone());
            return Err(MigrationError::ExecutionFailed {
                version,
                reason: result.error.unwrap_or_default(),
            });
        }

        let applied = AppliedMigration {
            version,
            name: migration.name.clone(),
            status: MigrationStatus::Applied,
            checksum: migration.checksum,
            applied_at: Some(ts),
            rolled_back_at: None,
            execution_time_ms: result.execution_time_ms,
        };
        self.applied.insert(version, applied.clone());
        Ok(applied)
    }

    /// Apply all pending migrations in order.
    pub fn apply_all(&mut self) -> Result<Vec<AppliedMigration>, MigrationError> {
        let pending = self.pending();
        let mut results = Vec::new();
        for version in pending {
            results.push(self.apply(version)?);
        }
        Ok(results)
    }

    /// Rollback a single migration (down).
    pub fn rollback(&mut self, version: u64) -> Result<AppliedMigration, MigrationError> {
        let migration = self
            .migrations
            .get(&version)
            .ok_or(MigrationError::NotFound(version))?
            .clone();

        if !self
            .applied
            .get(&version)
            .is_some_and(|a| a.status == MigrationStatus::Applied)
        {
            return Err(MigrationError::NotApplied(version));
        }

        if migration.down_script.trim().is_empty() {
            return Err(MigrationError::EmptyScript { version, direction: "down" });
        }

        let result = if let Some(executor) = &self.executor {
            executor(&migration.down_script)
        } else {
            ExecutionResult::ok(1)
        };

        let ts = self.next_timestamp();

        if !result.success {
            return Err(MigrationError::ExecutionFailed {
                version,
                reason: result.error.unwrap_or_default(),
            });
        }

        let record = AppliedMigration {
            version,
            name: migration.name.clone(),
            status: MigrationStatus::RolledBack,
            checksum: migration.checksum,
            applied_at: self.applied.get(&version).and_then(|a| a.applied_at),
            rolled_back_at: Some(ts),
            execution_time_ms: result.execution_time_ms,
        };
        self.applied.insert(version, record.clone());
        Ok(record)
    }

    /// Rollback the last N applied migrations.
    pub fn rollback_last(&mut self, count: usize) -> Result<Vec<AppliedMigration>, MigrationError> {
        let mut applied: Vec<u64> = self.applied_versions();
        applied.sort();
        applied.reverse();
        let to_rollback: Vec<u64> = applied.into_iter().take(count).collect();

        let mut results = Vec::new();
        for version in to_rollback {
            results.push(self.rollback(version)?);
        }
        Ok(results)
    }

    /// Dry-run: return what would happen without applying.
    pub fn dry_run_pending(&self) -> Vec<DryRunStep> {
        self.pending()
            .iter()
            .filter_map(|v| {
                self.migrations.get(v).map(|m| DryRunStep {
                    version: m.version,
                    name: m.name.clone(),
                    direction: MigrationDirection::Up,
                    script: m.up_script.clone(),
                })
            })
            .collect()
    }

    /// Dry-run rollback of last N migrations.
    pub fn dry_run_rollback(&self, count: usize) -> Vec<DryRunStep> {
        let mut applied: Vec<u64> = self.applied_versions();
        applied.sort();
        applied.reverse();
        applied
            .into_iter()
            .take(count)
            .filter_map(|v| {
                self.migrations.get(&v).map(|m| DryRunStep {
                    version: m.version,
                    name: m.name.clone(),
                    direction: MigrationDirection::Down,
                    script: m.down_script.clone(),
                })
            })
            .collect()
    }

    /// Get a migration record by version.
    pub fn get_applied(&self, version: u64) -> Option<&AppliedMigration> {
        self.applied.get(&version)
    }

    /// Generate a status report.
    pub fn report(&self) -> MigrationReport {
        let mut entries = Vec::new();
        for (version, migration) in &self.migrations {
            let status = self
                .applied
                .get(version)
                .map(|a| a.status)
                .unwrap_or(MigrationStatus::Pending);
            entries.push(ReportEntry {
                version: *version,
                name: migration.name.clone(),
                status,
            });
        }
        MigrationReport {
            total: self.migrations.len(),
            applied: self.applied_versions().len(),
            pending: self.pending().len(),
            entries,
        }
    }
}

impl Default for MigrationRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Migration report.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    pub total: usize,
    pub applied: usize,
    pub pending: usize,
    pub entries: Vec<ReportEntry>,
}

/// Single entry in a migration report.
#[derive(Debug, Clone)]
pub struct ReportEntry {
    pub version: u64,
    pub name: String,
    pub status: MigrationStatus,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn m(version: u64, name: &str) -> Migration {
        Migration::new(
            version,
            name,
            format!("CREATE TABLE {name}"),
            format!("DROP TABLE {name}"),
        )
    }

    #[test]
    fn test_register_migration() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        assert_eq!(runner.registered_versions(), vec![1]);
    }

    #[test]
    fn test_register_duplicate() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        let err = runner.register(m(1, "users_again")).unwrap_err();
        assert!(matches!(err, MigrationError::DuplicateVersion(1)));
    }

    #[test]
    fn test_apply_migration() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        let result = runner.apply(1).unwrap();
        assert_eq!(result.version, 1);
        assert_eq!(result.status, MigrationStatus::Applied);
        assert!(result.applied_at.is_some());
    }

    #[test]
    fn test_apply_not_found() {
        let mut runner = MigrationRunner::new();
        let err = runner.apply(999).unwrap_err();
        assert!(matches!(err, MigrationError::NotFound(999)));
    }

    #[test]
    fn test_apply_already_applied() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.apply(1).unwrap();
        let err = runner.apply(1).unwrap_err();
        assert!(matches!(err, MigrationError::AlreadyApplied(1)));
    }

    #[test]
    fn test_apply_empty_script() {
        let mut runner = MigrationRunner::new();
        runner.register(Migration::new(1, "empty", "  ", "DROP")).unwrap();
        let err = runner.apply(1).unwrap_err();
        assert!(matches!(err, MigrationError::EmptyScript { version: 1, direction: "up" }));
    }

    #[test]
    fn test_apply_all() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        runner.register(m(3, "comments")).unwrap();
        let results = runner.apply_all().unwrap();
        assert_eq!(results.len(), 3);
        assert!(runner.pending().is_empty());
    }

    #[test]
    fn test_pending_and_applied() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        assert_eq!(runner.pending(), vec![1, 2]);
        runner.apply(1).unwrap();
        assert_eq!(runner.pending(), vec![2]);
        assert_eq!(runner.applied_versions(), vec![1]);
    }

    #[test]
    fn test_rollback() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.apply(1).unwrap();
        let result = runner.rollback(1).unwrap();
        assert_eq!(result.status, MigrationStatus::RolledBack);
        assert!(result.rolled_back_at.is_some());
    }

    #[test]
    fn test_rollback_not_applied() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        let err = runner.rollback(1).unwrap_err();
        assert!(matches!(err, MigrationError::NotApplied(1)));
    }

    #[test]
    fn test_rollback_last() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        runner.register(m(3, "comments")).unwrap();
        runner.apply_all().unwrap();
        let results = runner.rollback_last(2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].version, 3);
        assert_eq!(results[1].version, 2);
        assert_eq!(runner.applied_versions(), vec![1]);
    }

    #[test]
    fn test_reapply_after_rollback() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.apply(1).unwrap();
        runner.rollback(1).unwrap();
        let result = runner.apply(1).unwrap();
        assert_eq!(result.status, MigrationStatus::Applied);
    }

    #[test]
    fn test_out_of_order_rejected() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.register(m(3, "comments")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        runner.apply(3).unwrap();
        let err = runner.apply(2).unwrap_err();
        assert!(matches!(err, MigrationError::OutOfOrder { .. }));
    }

    #[test]
    fn test_out_of_order_allowed() {
        let mut runner = MigrationRunner::new().allow_out_of_order(true);
        runner.register(m(1, "users")).unwrap();
        runner.register(m(3, "comments")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        runner.apply(3).unwrap();
        runner.apply(2).unwrap();
        assert_eq!(runner.applied_versions().len(), 2);
    }

    #[test]
    fn test_checksum_verification() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.apply(1).unwrap();
        assert!(runner.verify_checksum(1).unwrap());
    }

    #[test]
    fn test_checksum_mismatch() {
        let mut runner = MigrationRunner::new();
        let migration = m(1, "users");
        let checksum = migration.checksum;
        runner.register(migration).unwrap();
        runner.apply(1).unwrap();

        // Tamper with applied checksum.
        if let Some(a) = runner.applied.get_mut(&1) {
            a.checksum = checksum.wrapping_add(1);
        }
        assert!(!runner.verify_checksum(1).unwrap());
    }

    #[test]
    fn test_verify_all_checksums() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        runner.apply_all().unwrap();
        let errors = runner.verify_all_checksums();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_dry_run_pending() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        let steps = runner.dry_run_pending();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].version, 1);
        assert_eq!(steps[0].direction, MigrationDirection::Up);
        assert!(steps[0].script.contains("CREATE TABLE"));
    }

    #[test]
    fn test_dry_run_rollback() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        runner.apply_all().unwrap();
        let steps = runner.dry_run_rollback(1);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].version, 2);
        assert_eq!(steps[0].direction, MigrationDirection::Down);
    }

    #[test]
    fn test_status() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        assert_eq!(runner.status(1).unwrap(), MigrationStatus::Pending);
        runner.apply(1).unwrap();
        assert_eq!(runner.status(1).unwrap(), MigrationStatus::Applied);
        runner.rollback(1).unwrap();
        assert_eq!(runner.status(1).unwrap(), MigrationStatus::RolledBack);
    }

    #[test]
    fn test_last_applied_version() {
        let mut runner = MigrationRunner::new();
        assert!(runner.last_applied_version().is_none());
        runner.register(m(1, "users")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        runner.apply(1).unwrap();
        assert_eq!(runner.last_applied_version(), Some(1));
        runner.apply(2).unwrap();
        assert_eq!(runner.last_applied_version(), Some(2));
    }

    #[test]
    fn test_report() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        runner.register(m(2, "posts")).unwrap();
        runner.apply(1).unwrap();
        let report = runner.report();
        assert_eq!(report.total, 2);
        assert_eq!(report.applied, 1);
        assert_eq!(report.pending, 1);
        assert_eq!(report.entries.len(), 2);
    }

    #[test]
    fn test_custom_executor_success() {
        let mut runner = MigrationRunner::new();
        runner.set_executor(Box::new(|_script| ExecutionResult::ok(42)));
        runner.register(m(1, "users")).unwrap();
        let result = runner.apply(1).unwrap();
        assert_eq!(result.execution_time_ms, 42);
    }

    #[test]
    fn test_custom_executor_failure() {
        let mut runner = MigrationRunner::new();
        runner.set_executor(Box::new(|_script| {
            ExecutionResult::fail(10, "syntax error")
        }));
        runner.register(m(1, "users")).unwrap();
        let err = runner.apply(1).unwrap_err();
        match err {
            MigrationError::ExecutionFailed { version, reason } => {
                assert_eq!(version, 1);
                assert_eq!(reason, "syntax error");
            }
            _ => panic!("expected ExecutionFailed"),
        }
    }

    #[test]
    fn test_checksum_deterministic() {
        let m1 = m(1, "test");
        let m2 = Migration::new(1, "test", "CREATE TABLE test", "DROP TABLE test");
        assert_eq!(m1.checksum, m2.checksum);
    }

    #[test]
    fn test_checksum_differs_on_content() {
        let m1 = Migration::new(1, "a", "CREATE TABLE a", "DROP TABLE a");
        let m2 = Migration::new(1, "a", "CREATE TABLE b", "DROP TABLE a");
        assert_ne!(m1.checksum, m2.checksum);
    }

    #[test]
    fn test_rollback_empty_down_script() {
        let mut runner = MigrationRunner::new();
        runner.register(Migration::new(1, "irreversible", "CREATE", "")).unwrap();
        runner.apply(1).unwrap();
        let err = runner.rollback(1).unwrap_err();
        assert!(matches!(err, MigrationError::EmptyScript { version: 1, direction: "down" }));
    }

    #[test]
    fn test_get_applied() {
        let mut runner = MigrationRunner::new();
        runner.register(m(1, "users")).unwrap();
        assert!(runner.get_applied(1).is_none());
        runner.apply(1).unwrap();
        let applied = runner.get_applied(1).unwrap();
        assert_eq!(applied.name, "users");
    }
}
