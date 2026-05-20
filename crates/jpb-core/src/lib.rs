//! JoulesPerBit shared types — canonical definitions used across the workspace.
//!
//! All decision-path values are integer-only: no floats.
//! - [`Energy`]: micro-watt-hours (µWh), u64
//! - [`Score`]: basis points (0–10 000 = 0.00%–100.00%), u32
//! - [`TaskId`] / [`ArtifactId`]: UUID newtypes for type-safe identifiers
//! - [`EnergyReceipt`]: standard receipt returned from metered operations
//! - [`DomainStore`]: trait for energy-metered domain stores
//! - [`constants`]: shared numeric constants used across crates

pub mod constants;
pub mod domain;
pub mod domain_types;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a task node in the DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskId(pub uuid::Uuid);

/// Unique identifier for an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ArtifactId(pub uuid::Uuid);

/// Energy cost in micro-watt-hours (µWh). Integer only — no floats.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct Energy(pub u64);

impl std::ops::Add for Energy {
    type Output = Energy;

    fn add(self, other: Energy) -> Energy {
        Energy(self.0.saturating_add(other.0))
    }
}

impl std::ops::AddAssign for Energy {
    fn add_assign(&mut self, other: Energy) {
        self.0 = self.0.saturating_add(other.0);
    }
}

impl fmt::Display for Energy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}µWh", self.0)
    }
}

/// UTC timestamp for provenance records.
pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// Relevance score: integer basis points (0–10 000 = 0.00%–100.00%).
/// No floats in decision paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Score(pub u32);

impl Score {
    /// Maximum score: 10 000 basis points = 100.00%
    pub const MAX: Score = Score(10_000);
}

// Re-export domain types at crate root for convenience.
pub use domain::{DomainError, DomainStore, EnergyAccumulator, EnergyReceipt};
pub use domain_types::{
    ActivityKind, DeliveryChannel, EmailAddress, EmploymentType, InvoiceLine, InvoiceStatus,
    PayMethodKind, PayPeriod, SalaryRange, SourceModule,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_add_saturating() {
        let a = Energy(100);
        let b = Energy(200);
        assert_eq!(a + b, Energy(300));

        let max = Energy(u64::MAX);
        assert_eq!(max + Energy(1), Energy(u64::MAX));
    }

    #[test]
    fn energy_add_assign() {
        let mut e = Energy(100);
        e += Energy(200);
        assert_eq!(e, Energy(300));
    }

    #[test]
    fn energy_display() {
        assert_eq!(format!("{}", Energy(42)), "42µWh");
    }

    #[test]
    fn score_max() {
        assert_eq!(Score::MAX, Score(10_000));
        assert!(Score(5000) < Score::MAX);
    }

    #[test]
    fn task_id_ordering() {
        let a = TaskId(uuid::Uuid::nil());
        let b = TaskId(uuid::Uuid::max());
        assert!(a < b);
        assert_eq!(a, a);
    }

    #[test]
    fn cbor_roundtrip_energy() {
        let e = Energy(12345);
        let mut buf = Vec::new();
        ciborium::into_writer(&e, &mut buf).unwrap();
        let decoded: Energy = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(decoded, e);
    }

    #[test]
    fn cbor_roundtrip_task_id() {
        let id = TaskId(uuid::Uuid::new_v4());
        let mut buf = Vec::new();
        ciborium::into_writer(&id, &mut buf).unwrap();
        let decoded: TaskId = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(decoded, id);
    }

    #[test]
    fn cbor_roundtrip_score() {
        let s = Score(7500);
        let mut buf = Vec::new();
        ciborium::into_writer(&s, &mut buf).unwrap();
        let decoded: Score = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(decoded, s);
    }
}
