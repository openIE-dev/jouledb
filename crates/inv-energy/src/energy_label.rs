//! EU-style A-G energy labeling for workloads.
//!
//! Classifies workload energy efficiency into A through G categories
//! based on energy per functional unit relative to a baseline.

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// EnergyLabel
// ---------------------------------------------------------------------------

/// EU-style energy efficiency label from A (most efficient) to G (least
/// efficient).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EnergyLabel {
    /// Most efficient.
    A,
    /// Very efficient.
    B,
    /// Efficient.
    C,
    /// Average.
    D,
    /// Below average.
    E,
    /// Inefficient.
    F,
    /// Least efficient.
    G,
}

impl fmt::Display for EnergyLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::C => write!(f, "C"),
            Self::D => write!(f, "D"),
            Self::E => write!(f, "E"),
            Self::F => write!(f, "F"),
            Self::G => write!(f, "G"),
        }
    }
}

impl EnergyLabel {
    /// Returns a human-readable description of this label's efficiency tier.
    pub fn description(&self) -> &str {
        match self {
            Self::A => "Most efficient",
            Self::B => "Very efficient",
            Self::C => "Efficient",
            Self::D => "Average",
            Self::E => "Below average",
            Self::F => "Inefficient",
            Self::G => "Least efficient",
        }
    }
}

// ---------------------------------------------------------------------------
// LabelThresholds
// ---------------------------------------------------------------------------

/// Defines the ratio boundaries for each energy label.
///
/// The ratio is computed as `energy_per_unit / baseline`.
///
/// Default thresholds:
/// - **A**: ratio < 0.50
/// - **B**: 0.50 <= ratio < 0.70
/// - **C**: 0.70 <= ratio < 0.90
/// - **D**: 0.90 <= ratio < 1.10
/// - **E**: 1.10 <= ratio < 1.30
/// - **F**: 1.30 <= ratio < 1.50
/// - **G**: ratio >= 1.50
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelThresholds {
    /// Upper bound for label A (exclusive).
    pub a_max: f64,
    /// Upper bound for label B (exclusive).
    pub b_max: f64,
    /// Upper bound for label C (exclusive).
    pub c_max: f64,
    /// Upper bound for label D (exclusive).
    pub d_max: f64,
    /// Upper bound for label E (exclusive).
    pub e_max: f64,
    /// Upper bound for label F (exclusive).
    pub f_max: f64,
}

impl Default for LabelThresholds {
    fn default() -> Self {
        Self {
            a_max: 0.50,
            b_max: 0.70,
            c_max: 0.90,
            d_max: 1.10,
            e_max: 1.30,
            f_max: 1.50,
        }
    }
}

impl LabelThresholds {
    /// Classifies a ratio (energy_per_unit / baseline) into an
    /// [`EnergyLabel`].
    pub fn classify(&self, ratio: f64) -> EnergyLabel {
        if ratio < self.a_max {
            EnergyLabel::A
        } else if ratio < self.b_max {
            EnergyLabel::B
        } else if ratio < self.c_max {
            EnergyLabel::C
        } else if ratio < self.d_max {
            EnergyLabel::D
        } else if ratio < self.e_max {
            EnergyLabel::E
        } else if ratio < self.f_max {
            EnergyLabel::F
        } else {
            EnergyLabel::G
        }
    }
}

// ---------------------------------------------------------------------------
// LabelReport
// ---------------------------------------------------------------------------

/// A complete energy-label report for a workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelReport {
    /// Identifier of the workload being labelled.
    pub workload_id: String,
    /// Assigned energy label.
    pub label: EnergyLabel,
    /// Measured energy per functional unit.
    pub energy_per_unit: f64,
    /// Baseline energy per functional unit.
    pub baseline: f64,
    /// Ratio of measured to baseline (`energy_per_unit / baseline`).
    pub ratio: f64,
}

// ---------------------------------------------------------------------------
// Public convenience API
// ---------------------------------------------------------------------------

/// Classifies energy efficiency using the default thresholds.
///
/// `energy_per_unit` is the measured energy consumption per functional unit.
/// `baseline` is the reference energy per functional unit.
pub fn classify(energy_per_unit: f64, baseline: f64) -> EnergyLabel {
    let ratio = if baseline == 0.0 {
        f64::MAX
    } else {
        energy_per_unit / baseline
    };
    LabelThresholds::default().classify(ratio)
}

/// Generates a full [`LabelReport`] using the default thresholds.
pub fn generate_label_report(
    workload_id: &str,
    energy_per_unit: f64,
    baseline: f64,
) -> LabelReport {
    let ratio = if baseline == 0.0 {
        f64::MAX
    } else {
        energy_per_unit / baseline
    };
    let label = LabelThresholds::default().classify(ratio);

    LabelReport {
        workload_id: workload_id.to_string(),
        label,
        energy_per_unit,
        baseline,
        ratio,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_a() {
        // ratio < 0.50 => A
        assert_eq!(classify(0.3, 1.0), EnergyLabel::A);
    }

    #[test]
    fn classify_b() {
        // 0.50 <= ratio < 0.70 => B
        assert_eq!(classify(0.6, 1.0), EnergyLabel::B);
    }

    #[test]
    fn classify_c() {
        // 0.70 <= ratio < 0.90 => C
        assert_eq!(classify(0.8, 1.0), EnergyLabel::C);
    }

    #[test]
    fn classify_d() {
        // 0.90 <= ratio < 1.10 => D
        assert_eq!(classify(1.0, 1.0), EnergyLabel::D);
    }

    #[test]
    fn classify_e() {
        // 1.10 <= ratio < 1.30 => E
        assert_eq!(classify(1.2, 1.0), EnergyLabel::E);
    }

    #[test]
    fn classify_f() {
        // 1.30 <= ratio < 1.50 => F
        assert_eq!(classify(1.4, 1.0), EnergyLabel::F);
    }

    #[test]
    fn classify_g() {
        // ratio >= 1.50 => G
        assert_eq!(classify(2.0, 1.0), EnergyLabel::G);
    }

    #[test]
    fn label_display() {
        assert_eq!(format!("{}", EnergyLabel::A), "A");
        assert_eq!(format!("{}", EnergyLabel::B), "B");
        assert_eq!(format!("{}", EnergyLabel::C), "C");
        assert_eq!(format!("{}", EnergyLabel::D), "D");
        assert_eq!(format!("{}", EnergyLabel::E), "E");
        assert_eq!(format!("{}", EnergyLabel::F), "F");
        assert_eq!(format!("{}", EnergyLabel::G), "G");
    }

    #[test]
    fn label_ordering() {
        // A < B < C < D < E < F < G
        assert!(EnergyLabel::A < EnergyLabel::B);
        assert!(EnergyLabel::B < EnergyLabel::C);
        assert!(EnergyLabel::C < EnergyLabel::D);
        assert!(EnergyLabel::D < EnergyLabel::E);
        assert!(EnergyLabel::E < EnergyLabel::F);
        assert!(EnergyLabel::F < EnergyLabel::G);
    }

    #[test]
    fn label_description() {
        assert_eq!(EnergyLabel::A.description(), "Most efficient");
        assert_eq!(EnergyLabel::B.description(), "Very efficient");
        assert_eq!(EnergyLabel::C.description(), "Efficient");
        assert_eq!(EnergyLabel::D.description(), "Average");
        assert_eq!(EnergyLabel::E.description(), "Below average");
        assert_eq!(EnergyLabel::F.description(), "Inefficient");
        assert_eq!(EnergyLabel::G.description(), "Least efficient");
    }

    #[test]
    fn generate_report() {
        let report = generate_label_report("wl-42", 0.6, 1.0);
        assert_eq!(report.workload_id, "wl-42");
        assert_eq!(report.label, EnergyLabel::B);
        assert!((report.energy_per_unit - 0.6).abs() < 1e-10);
        assert!((report.baseline - 1.0).abs() < 1e-10);
        assert!((report.ratio - 0.6).abs() < 1e-10);
    }

    #[test]
    fn report_serialization() {
        let report = generate_label_report("wl-99", 1.4, 1.0);
        let json = serde_json::to_string(&report).expect("serialize");
        let d: LabelReport = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(d.workload_id, "wl-99");
        assert_eq!(d.label, EnergyLabel::F);
        assert!((d.energy_per_unit - 1.4).abs() < 1e-10);
        assert!((d.baseline - 1.0).abs() < 1e-10);
        assert!((d.ratio - 1.4).abs() < 1e-10);
    }
}
