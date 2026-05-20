//! Database migration system — versioned schema changes with up/down support.
//!
//! Replaces Knex migrations, Prisma Migrate, and Flyway with pure Rust.

use std::collections::BTreeSet;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Migration errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationError {
    /// Migration already applied.
    AlreadyApplied(u64),
    /// Migration not found.
    NotFound(u64),
    /// Gap detected in migration versions.
    VersionGap { expected: u64, found: u64 },
    /// No migrations to rollback.
    NothingToRollback,
    /// Duplicate version number.
    DuplicateVersion(u64),
    /// Validation failed.
    ValidationError(String),
}

impl fmt::Display for MigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyApplied(v) => write!(f, "migration {v} already applied"),
            Self::NotFound(v) => write!(f, "migration {v} not found"),
            Self::VersionGap { expected, found } => {
                write!(f, "version gap: expected {expected}, found {found}")
            }
            Self::NothingToRollback => write!(f, "no migrations to rollback"),
            Self::DuplicateVersion(v) => write!(f, "duplicate migration version: {v}"),
            Self::ValidationError(msg) => write!(f, "validation error: {msg}"),
        }
    }
}

impl std::error::Error for MigrationError {}

// ── Migration ───────────────────────────────────────────────────

/// A single migration with up and down SQL.
#[derive(Debug, Clone)]
pub struct Migration {
    pub version: u64,
    pub name: String,
    pub up_sql: String,
    pub down_sql: String,
}

impl Migration {
    pub fn new(
        version: u64,
        name: impl Into<String>,
        up_sql: impl Into<String>,
        down_sql: impl Into<String>,
    ) -> Self {
        Self {
            version,
            name: name.into(),
            up_sql: up_sql.into(),
            down_sql: down_sql.into(),
        }
    }
}

// ── Migration Status ────────────────────────────────────────────

/// Status of a migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationStatus {
    Pending,
    Applied,
}

/// Info about a migration and its status.
#[derive(Debug, Clone)]
pub struct MigrationInfo {
    pub version: u64,
    pub name: String,
    pub status: MigrationStatus,
}

// ── Applied record ──────────────────────────────────────────────

/// Record of an applied migration step (for dry-run and logging).
#[derive(Debug, Clone)]
pub struct AppliedStep {
    pub version: u64,
    pub name: String,
    pub sql: String,
    pub direction: Direction,
}

/// Direction of a migration step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
}

// ── Runner ──────────────────────────────────────────────────────

/// Manages and runs migrations.
#[derive(Debug)]
pub struct MigrationRunner {
    migrations: Vec<Migration>,
    applied_versions: BTreeSet<u64>,
}

impl Default for MigrationRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl MigrationRunner {
    pub fn new() -> Self {
        Self {
            migrations: Vec::new(),
            applied_versions: BTreeSet::new(),
        }
    }

    /// Register a migration. Rejects duplicates.
    pub fn add_migration(&mut self, migration: Migration) -> Result<(), MigrationError> {
        if self.migrations.iter().any(|m| m.version == migration.version) {
            return Err(MigrationError::DuplicateVersion(migration.version));
        }
        self.migrations.push(migration);
        self.migrations.sort_by_key(|m| m.version);
        Ok(())
    }

    /// Validate that versions form a gap-free sequence starting at the lowest.
    pub fn validate(&self) -> Result<(), MigrationError> {
        if self.migrations.is_empty() {
            return Ok(());
        }
        let versions: Vec<u64> = self.migrations.iter().map(|m| m.version).collect();
        for window in versions.windows(2) {
            if window[1] != window[0] + 1 {
                return Err(MigrationError::VersionGap {
                    expected: window[0] + 1,
                    found: window[1],
                });
            }
        }
        Ok(())
    }

    /// Return the status of all migrations.
    pub fn status(&self) -> Vec<MigrationInfo> {
        self.migrations
            .iter()
            .map(|m| MigrationInfo {
                version: m.version,
                name: m.name.clone(),
                status: if self.applied_versions.contains(&m.version) {
                    MigrationStatus::Applied
                } else {
                    MigrationStatus::Pending
                },
            })
            .collect()
    }

    /// Return pending migration versions.
    pub fn pending(&self) -> Vec<u64> {
        self.migrations
            .iter()
            .filter(|m| !self.applied_versions.contains(&m.version))
            .map(|m| m.version)
            .collect()
    }

    /// Run all pending migrations in order. Returns applied steps.
    pub fn run_pending(&mut self) -> Result<Vec<AppliedStep>, MigrationError> {
        let pending: Vec<Migration> = self
            .migrations
            .iter()
            .filter(|m| !self.applied_versions.contains(&m.version))
            .cloned()
            .collect();
        let mut steps = Vec::new();
        for m in pending {
            self.applied_versions.insert(m.version);
            steps.push(AppliedStep {
                version: m.version,
                name: m.name.clone(),
                sql: m.up_sql.clone(),
                direction: Direction::Up,
            });
        }
        Ok(steps)
    }

    /// Rollback the last N applied migrations (most recent first).
    pub fn rollback(&mut self, count: usize) -> Result<Vec<AppliedStep>, MigrationError> {
        if self.applied_versions.is_empty() {
            return Err(MigrationError::NothingToRollback);
        }
        let mut rolled: Vec<u64> = self.applied_versions.iter().rev().take(count).copied().collect();
        rolled.sort_unstable();
        rolled.reverse(); // highest first

        let mut steps = Vec::new();
        for version in rolled {
            let migration = self
                .migrations
                .iter()
                .find(|m| m.version == version)
                .ok_or(MigrationError::NotFound(version))?;
            steps.push(AppliedStep {
                version,
                name: migration.name.clone(),
                sql: migration.down_sql.clone(),
                direction: Direction::Down,
            });
            self.applied_versions.remove(&version);
        }
        Ok(steps)
    }

    /// Dry run: return what would be applied without actually applying.
    pub fn dry_run(&self) -> Vec<AppliedStep> {
        self.migrations
            .iter()
            .filter(|m| !self.applied_versions.contains(&m.version))
            .map(|m| AppliedStep {
                version: m.version,
                name: m.name.clone(),
                sql: m.up_sql.clone(),
                direction: Direction::Up,
            })
            .collect()
    }

    /// Generate the next sequential version number.
    pub fn next_version(&self) -> u64 {
        self.migrations
            .iter()
            .map(|m| m.version)
            .max()
            .map(|v| v + 1)
            .unwrap_or(1)
    }

    /// Generate a new migration skeleton with the next version.
    pub fn generate_migration(&self, name: impl Into<String>) -> Migration {
        Migration::new(self.next_version(), name, "", "")
    }

    /// Number of applied migrations.
    pub fn applied_count(&self) -> usize {
        self.applied_versions.len()
    }

    /// Total number of registered migrations.
    pub fn total_count(&self) -> usize {
        self.migrations.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_migrations() -> Vec<Migration> {
        vec![
            Migration::new(1, "create_users", "CREATE TABLE users (id INT)", "DROP TABLE users"),
            Migration::new(2, "add_email", "ALTER TABLE users ADD email TEXT", "ALTER TABLE users DROP email"),
            Migration::new(3, "create_posts", "CREATE TABLE posts (id INT)", "DROP TABLE posts"),
        ]
    }

    fn runner_with_migrations() -> MigrationRunner {
        let mut runner = MigrationRunner::new();
        for m in sample_migrations() {
            runner.add_migration(m).unwrap();
        }
        runner
    }

    #[test]
    fn add_migrations_sorted() {
        let mut runner = MigrationRunner::new();
        runner.add_migration(Migration::new(3, "c", "", "")).unwrap();
        runner.add_migration(Migration::new(1, "a", "", "")).unwrap();
        runner.add_migration(Migration::new(2, "b", "", "")).unwrap();
        let statuses = runner.status();
        assert_eq!(statuses[0].version, 1);
        assert_eq!(statuses[1].version, 2);
        assert_eq!(statuses[2].version, 3);
    }

    #[test]
    fn duplicate_version_rejected() {
        let mut runner = MigrationRunner::new();
        runner.add_migration(Migration::new(1, "a", "", "")).unwrap();
        let err = runner.add_migration(Migration::new(1, "b", "", "")).unwrap_err();
        assert_eq!(err, MigrationError::DuplicateVersion(1));
    }

    #[test]
    fn validate_no_gaps() {
        let runner = runner_with_migrations();
        assert!(runner.validate().is_ok());
    }

    #[test]
    fn validate_detects_gap() {
        let mut runner = MigrationRunner::new();
        runner.add_migration(Migration::new(1, "a", "", "")).unwrap();
        runner.add_migration(Migration::new(3, "c", "", "")).unwrap();
        let err = runner.validate().unwrap_err();
        assert_eq!(err, MigrationError::VersionGap { expected: 2, found: 3 });
    }

    #[test]
    fn run_pending_applies_all() {
        let mut runner = runner_with_migrations();
        let steps = runner.run_pending().unwrap();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].version, 1);
        assert_eq!(steps[2].version, 3);
        assert!(runner.pending().is_empty());
    }

    #[test]
    fn run_pending_idempotent() {
        let mut runner = runner_with_migrations();
        runner.run_pending().unwrap();
        let steps = runner.run_pending().unwrap();
        assert!(steps.is_empty());
    }

    #[test]
    fn rollback_last() {
        let mut runner = runner_with_migrations();
        runner.run_pending().unwrap();
        let steps = runner.rollback(1).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].version, 3);
        assert_eq!(steps[0].direction, Direction::Down);
        assert_eq!(runner.applied_count(), 2);
    }

    #[test]
    fn rollback_multiple() {
        let mut runner = runner_with_migrations();
        runner.run_pending().unwrap();
        let steps = runner.rollback(2).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].version, 3);
        assert_eq!(steps[1].version, 2);
        assert_eq!(runner.applied_count(), 1);
    }

    #[test]
    fn rollback_empty_fails() {
        let mut runner = runner_with_migrations();
        let err = runner.rollback(1).unwrap_err();
        assert_eq!(err, MigrationError::NothingToRollback);
    }

    #[test]
    fn status_shows_pending_and_applied() {
        let mut runner = runner_with_migrations();
        runner.run_pending().unwrap();
        runner.rollback(1).unwrap();
        let statuses = runner.status();
        assert_eq!(statuses[0].status, MigrationStatus::Applied);
        assert_eq!(statuses[1].status, MigrationStatus::Applied);
        assert_eq!(statuses[2].status, MigrationStatus::Pending);
    }

    #[test]
    fn dry_run_does_not_apply() {
        let runner = runner_with_migrations();
        let steps = runner.dry_run();
        assert_eq!(steps.len(), 3);
        assert_eq!(runner.applied_count(), 0);
    }

    #[test]
    fn generate_migration_sequential() {
        let runner = runner_with_migrations();
        let m = runner.generate_migration("add_index");
        assert_eq!(m.version, 4);
        assert_eq!(m.name, "add_index");
    }

    #[test]
    fn generate_migration_from_empty() {
        let runner = MigrationRunner::new();
        let m = runner.generate_migration("init");
        assert_eq!(m.version, 1);
    }

    #[test]
    fn next_version_after_partial_apply() {
        let mut runner = runner_with_migrations();
        runner.run_pending().unwrap();
        assert_eq!(runner.next_version(), 4);
    }
}
