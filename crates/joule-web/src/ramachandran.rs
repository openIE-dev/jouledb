//! Ramachandran plot analysis, allowed/disallowed regions, and amino acid preferences.
//!
//! Models the distribution of backbone phi/psi dihedral angles,
//! classifies residues into core, allowed, generous, and disallowed
//! regions, and provides amino-acid-specific preferences (glycine,
//! proline, pre-proline, general).

use std::fmt;

// ── Region Classification ───────────────────────────────────────────

/// Ramachandran region classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RamaRegion {
    /// Core (most-favoured) region.
    Core,
    /// Allowed region.
    Allowed,
    /// Generously allowed region.
    Generous,
    /// Disallowed region.
    Disallowed,
}

impl RamaRegion {
    /// Numeric quality score (3=core, 2=allowed, 1=generous, 0=disallowed).
    pub fn score(&self) -> u8 {
        match self {
            Self::Core => 3,
            Self::Allowed => 2,
            Self::Generous => 1,
            Self::Disallowed => 0,
        }
    }
}

impl fmt::Display for RamaRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core => write!(f, "core"),
            Self::Allowed => write!(f, "allowed"),
            Self::Generous => write!(f, "generous"),
            Self::Disallowed => write!(f, "disallowed"),
        }
    }
}

// ── Amino Acid Class ────────────────────────────────────────────────

/// Amino acid class for Ramachandran preferences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AminoAcidClass {
    /// General case (most amino acids).
    General,
    /// Glycine (achiral, extended allowed regions).
    Glycine,
    /// Proline (restricted phi ~-60°).
    Proline,
    /// Pre-proline (residue before proline, restricted psi).
    PreProline,
}

impl fmt::Display for AminoAcidClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::General => write!(f, "general"),
            Self::Glycine => write!(f, "glycine"),
            Self::Proline => write!(f, "proline"),
            Self::PreProline => write!(f, "pre-proline"),
        }
    }
}

// ── Rama Point ──────────────────────────────────────────────────────

/// A single point on the Ramachandran plot.
#[derive(Debug, Clone, PartialEq)]
pub struct RamaPoint {
    /// Residue index.
    pub residue: usize,
    /// Residue name (3-letter code).
    pub name: String,
    /// Phi angle in degrees.
    pub phi: f64,
    /// Psi angle in degrees.
    pub psi: f64,
    /// Amino acid class.
    pub aa_class: AminoAcidClass,
}

impl RamaPoint {
    pub fn new(residue: usize, name: &str, phi: f64, psi: f64) -> Self {
        let aa_class = match name.to_uppercase().as_str() {
            "GLY" => AminoAcidClass::Glycine,
            "PRO" => AminoAcidClass::Proline,
            _ => AminoAcidClass::General,
        };
        Self { residue, name: name.to_string(), phi, psi, aa_class }
    }

    /// Create with explicit amino acid class override.
    pub fn with_class(mut self, class: AminoAcidClass) -> Self {
        self.aa_class = class;
        self
    }
}

impl fmt::Display for RamaPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} φ={:.1}° ψ={:.1}° [{}]", self.residue, self.name, self.phi, self.psi, self.aa_class)
    }
}

// ── Elliptical Region ───────────────────────────────────────────────

/// An elliptical region on the Ramachandran plot.
#[derive(Debug, Clone)]
struct EllipseRegion {
    phi_center: f64,
    psi_center: f64,
    phi_radius: f64,
    psi_radius: f64,
}

impl EllipseRegion {
    fn contains(&self, phi: f64, psi: f64) -> bool {
        let dp = (phi - self.phi_center) / self.phi_radius;
        let ds = (psi - self.psi_center) / self.psi_radius;
        dp * dp + ds * ds <= 1.0
    }
}

// ── Rama Classifier ─────────────────────────────────────────────────

/// Classifies backbone angles into Ramachandran regions.
#[derive(Debug, Clone)]
pub struct RamaClassifier {
    /// Scale factor for core region (smaller = stricter).
    core_scale: f64,
    /// Scale factor for allowed region.
    allowed_scale: f64,
    /// Scale factor for generous region.
    generous_scale: f64,
}

impl RamaClassifier {
    /// Default classifier with standard region boundaries.
    pub fn new() -> Self {
        Self { core_scale: 1.0, allowed_scale: 1.6, generous_scale: 2.2 }
    }

    /// Adjust core region stringency.
    pub fn with_core_scale(mut self, s: f64) -> Self {
        self.core_scale = s;
        self
    }

    /// Adjust allowed region size.
    pub fn with_allowed_scale(mut self, s: f64) -> Self {
        self.allowed_scale = s;
        self
    }

    /// Adjust generous region size.
    pub fn with_generous_scale(mut self, s: f64) -> Self {
        self.generous_scale = s;
        self
    }

    /// Classify a single phi/psi pair for a given amino acid class.
    pub fn classify(&self, phi: f64, psi: f64, class: AminoAcidClass) -> RamaRegion {
        let regions = Self::base_regions(class);

        for r in &regions {
            let core = EllipseRegion {
                phi_center: r.phi_center,
                psi_center: r.psi_center,
                phi_radius: r.phi_radius * self.core_scale,
                psi_radius: r.psi_radius * self.core_scale,
            };
            if core.contains(phi, psi) {
                return RamaRegion::Core;
            }
        }

        for r in &regions {
            let allowed = EllipseRegion {
                phi_center: r.phi_center,
                psi_center: r.psi_center,
                phi_radius: r.phi_radius * self.allowed_scale,
                psi_radius: r.psi_radius * self.allowed_scale,
            };
            if allowed.contains(phi, psi) {
                return RamaRegion::Allowed;
            }
        }

        for r in &regions {
            let generous = EllipseRegion {
                phi_center: r.phi_center,
                psi_center: r.psi_center,
                phi_radius: r.phi_radius * self.generous_scale,
                psi_radius: r.psi_radius * self.generous_scale,
            };
            if generous.contains(phi, psi) {
                return RamaRegion::Generous;
            }
        }

        RamaRegion::Disallowed
    }

    /// Classify a `RamaPoint`.
    pub fn classify_point(&self, pt: &RamaPoint) -> RamaRegion {
        self.classify(pt.phi, pt.psi, pt.aa_class)
    }

    fn base_regions(class: AminoAcidClass) -> Vec<EllipseRegion> {
        match class {
            AminoAcidClass::General => vec![
                // Alpha-helix region
                EllipseRegion { phi_center: -63.0, psi_center: -43.0, phi_radius: 25.0, psi_radius: 25.0 },
                // Beta-sheet region
                EllipseRegion { phi_center: -120.0, psi_center: 135.0, phi_radius: 30.0, psi_radius: 25.0 },
                // Left-handed helix (small)
                EllipseRegion { phi_center: 57.0, psi_center: 47.0, phi_radius: 15.0, psi_radius: 15.0 },
            ],
            AminoAcidClass::Glycine => vec![
                EllipseRegion { phi_center: -63.0, psi_center: -43.0, phi_radius: 35.0, psi_radius: 35.0 },
                EllipseRegion { phi_center: -120.0, psi_center: 135.0, phi_radius: 40.0, psi_radius: 35.0 },
                EllipseRegion { phi_center: 63.0, psi_center: 43.0, phi_radius: 35.0, psi_radius: 35.0 },
                EllipseRegion { phi_center: 120.0, psi_center: -135.0, phi_radius: 40.0, psi_radius: 35.0 },
            ],
            AminoAcidClass::Proline => vec![
                EllipseRegion { phi_center: -63.0, psi_center: -30.0, phi_radius: 15.0, psi_radius: 20.0 },
                EllipseRegion { phi_center: -63.0, psi_center: 150.0, phi_radius: 15.0, psi_radius: 25.0 },
            ],
            AminoAcidClass::PreProline => vec![
                EllipseRegion { phi_center: -63.0, psi_center: -25.0, phi_radius: 25.0, psi_radius: 20.0 },
                EllipseRegion { phi_center: -120.0, psi_center: 150.0, phi_radius: 30.0, psi_radius: 20.0 },
            ],
        }
    }
}

impl fmt::Display for RamaClassifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RamaClassifier(core={:.1}x, allowed={:.1}x, generous={:.1}x)",
            self.core_scale, self.allowed_scale, self.generous_scale,
        )
    }
}

// ── Rama Analysis ───────────────────────────────────────────────────

/// Summary of a Ramachandran analysis.
#[derive(Debug, Clone)]
pub struct RamaAnalysis {
    pub total: usize,
    pub core_count: usize,
    pub allowed_count: usize,
    pub generous_count: usize,
    pub disallowed_count: usize,
    pub outliers: Vec<usize>,
}

impl RamaAnalysis {
    /// Run analysis on a set of points.
    pub fn analyze(classifier: &RamaClassifier, points: &[RamaPoint]) -> Self {
        let mut result = Self {
            total: points.len(),
            core_count: 0,
            allowed_count: 0,
            generous_count: 0,
            disallowed_count: 0,
            outliers: Vec::new(),
        };

        for pt in points {
            match classifier.classify_point(pt) {
                RamaRegion::Core => result.core_count += 1,
                RamaRegion::Allowed => result.allowed_count += 1,
                RamaRegion::Generous => result.generous_count += 1,
                RamaRegion::Disallowed => {
                    result.disallowed_count += 1;
                    result.outliers.push(pt.residue);
                }
            }
        }

        result
    }

    /// Fraction in core region.
    pub fn core_fraction(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.core_count as f64 / self.total as f64 }
    }

    /// Fraction in allowed or better.
    pub fn favoured_fraction(&self) -> f64 {
        if self.total == 0 { 0.0 } else { (self.core_count + self.allowed_count) as f64 / self.total as f64 }
    }

    /// True if the structure passes validation (>= 98% in favoured, 0 disallowed).
    pub fn passes_validation(&self) -> bool {
        self.favoured_fraction() >= 0.98 && self.disallowed_count == 0
    }
}

impl fmt::Display for RamaAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rama(n={}, core={:.1}%, favoured={:.1}%, outliers={})",
            self.total,
            self.core_fraction() * 100.0,
            self.favoured_fraction() * 100.0,
            self.disallowed_count,
        )
    }
}

// ── Z-Score ─────────────────────────────────────────────────────────

/// Ramachandran Z-score for overall model quality.
pub fn rama_z_score(analysis: &RamaAnalysis, expected_core: f64, std_dev: f64) -> f64 {
    if std_dev <= 0.0 {
        return 0.0;
    }
    (analysis.core_fraction() - expected_core) / std_dev
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_region_scores() {
        assert_eq!(RamaRegion::Core.score(), 3);
        assert_eq!(RamaRegion::Allowed.score(), 2);
        assert_eq!(RamaRegion::Generous.score(), 1);
        assert_eq!(RamaRegion::Disallowed.score(), 0);
    }

    #[test]
    fn test_region_display() {
        assert_eq!(RamaRegion::Core.to_string(), "core");
        assert_eq!(RamaRegion::Disallowed.to_string(), "disallowed");
    }

    #[test]
    fn test_aa_class_auto_detect() {
        let gly = RamaPoint::new(0, "GLY", -60.0, -40.0);
        assert_eq!(gly.aa_class, AminoAcidClass::Glycine);

        let pro = RamaPoint::new(1, "PRO", -63.0, -30.0);
        assert_eq!(pro.aa_class, AminoAcidClass::Proline);

        let ala = RamaPoint::new(2, "ALA", -60.0, -45.0);
        assert_eq!(ala.aa_class, AminoAcidClass::General);
    }

    #[test]
    fn test_rama_point_with_class() {
        let pt = RamaPoint::new(5, "ALA", -63.0, -25.0)
            .with_class(AminoAcidClass::PreProline);
        assert_eq!(pt.aa_class, AminoAcidClass::PreProline);
    }

    #[test]
    fn test_rama_point_display() {
        let pt = RamaPoint::new(3, "LEU", -65.0, -40.0);
        let s = pt.to_string();
        assert!(s.contains("LEU"));
        assert!(s.contains("general"));
    }

    #[test]
    fn test_classify_core_alpha() {
        let c = RamaClassifier::new();
        assert_eq!(c.classify(-63.0, -43.0, AminoAcidClass::General), RamaRegion::Core);
    }

    #[test]
    fn test_classify_core_beta() {
        let c = RamaClassifier::new();
        assert_eq!(c.classify(-120.0, 135.0, AminoAcidClass::General), RamaRegion::Core);
    }

    #[test]
    fn test_classify_disallowed() {
        let c = RamaClassifier::new();
        assert_eq!(c.classify(0.0, 0.0, AminoAcidClass::General), RamaRegion::Disallowed);
    }

    #[test]
    fn test_classify_glycine_mirror() {
        let c = RamaClassifier::new();
        // Glycine allows left-handed helix region
        let region = c.classify(63.0, 43.0, AminoAcidClass::Glycine);
        assert_ne!(region, RamaRegion::Disallowed);
    }

    #[test]
    fn test_classify_proline_restricted() {
        let c = RamaClassifier::new();
        // Proline core near -63, -30
        assert_eq!(c.classify(-63.0, -30.0, AminoAcidClass::Proline), RamaRegion::Core);
    }

    #[test]
    fn test_classifier_display() {
        let c = RamaClassifier::new();
        assert!(c.to_string().contains("RamaClassifier"));
    }

    #[test]
    fn test_classifier_builders() {
        let c = RamaClassifier::new()
            .with_core_scale(0.8)
            .with_allowed_scale(1.4)
            .with_generous_scale(2.0);
        assert!(c.to_string().contains("0.8"));
    }

    #[test]
    fn test_analysis_all_core() {
        let c = RamaClassifier::new();
        let points = vec![
            RamaPoint::new(0, "ALA", -63.0, -43.0),
            RamaPoint::new(1, "ALA", -63.0, -43.0),
        ];
        let analysis = RamaAnalysis::analyze(&c, &points);
        assert_eq!(analysis.core_count, 2);
        assert_eq!(analysis.disallowed_count, 0);
        assert!(analysis.passes_validation());
    }

    #[test]
    fn test_analysis_with_outlier() {
        let c = RamaClassifier::new();
        let points = vec![
            RamaPoint::new(0, "ALA", -63.0, -43.0),
            RamaPoint::new(1, "ALA", 0.0, 0.0),
        ];
        let analysis = RamaAnalysis::analyze(&c, &points);
        assert_eq!(analysis.disallowed_count, 1);
        assert!(analysis.outliers.contains(&1));
        assert!(!analysis.passes_validation());
    }

    #[test]
    fn test_analysis_fractions() {
        let c = RamaClassifier::new();
        let points = vec![
            RamaPoint::new(0, "ALA", -63.0, -43.0),
            RamaPoint::new(1, "ALA", -63.0, -43.0),
            RamaPoint::new(2, "ALA", -63.0, -43.0),
            RamaPoint::new(3, "ALA", 0.0, 0.0),
        ];
        let analysis = RamaAnalysis::analyze(&c, &points);
        assert!(approx(analysis.core_fraction(), 0.75, 1e-10));
    }

    #[test]
    fn test_analysis_display() {
        let c = RamaClassifier::new();
        let points = vec![RamaPoint::new(0, "ALA", -63.0, -43.0)];
        let analysis = RamaAnalysis::analyze(&c, &points);
        assert!(analysis.to_string().contains("Rama("));
    }

    #[test]
    fn test_z_score() {
        let analysis = RamaAnalysis {
            total: 100, core_count: 90, allowed_count: 8,
            generous_count: 2, disallowed_count: 0, outliers: vec![],
        };
        let z = rama_z_score(&analysis, 0.85, 0.05);
        assert!(approx(z, 1.0, 1e-10)); // (0.9 - 0.85) / 0.05 = 1.0
    }

    #[test]
    fn test_z_score_zero_stddev() {
        let analysis = RamaAnalysis {
            total: 10, core_count: 8, allowed_count: 2,
            generous_count: 0, disallowed_count: 0, outliers: vec![],
        };
        assert!(approx(rama_z_score(&analysis, 0.8, 0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_empty_analysis() {
        let c = RamaClassifier::new();
        let analysis = RamaAnalysis::analyze(&c, &[]);
        assert_eq!(analysis.total, 0);
        assert!(approx(analysis.core_fraction(), 0.0, 1e-10));
        assert!(approx(analysis.favoured_fraction(), 0.0, 1e-10));
    }
}
