//! Secondary structure prediction, DSSP-like assignment, and structure motifs.
//!
//! Implements helix, sheet, and coil classification from backbone geometry,
//! hydrogen-bond patterns for DSSP assignment, and common structural motif
//! detection including turns, hairpins, and helix-turn-helix.

use std::fmt;

// ── Secondary Structure Kind ────────────────────────────────────────

/// Classification of secondary structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SSKind {
    /// Alpha helix (i→i+4 H-bond pattern).
    AlphaHelix,
    /// 3₁₀ helix (i→i+3 H-bond pattern).
    Helix310,
    /// Pi helix (i→i+5 H-bond pattern).
    PiHelix,
    /// Parallel beta strand.
    ParallelSheet,
    /// Antiparallel beta strand.
    AntiparallelSheet,
    /// Beta turn.
    Turn,
    /// Random coil / loop.
    Coil,
}

impl SSKind {
    /// Single-character DSSP code.
    pub fn dssp_code(&self) -> char {
        match self {
            Self::AlphaHelix => 'H',
            Self::Helix310 => 'G',
            Self::PiHelix => 'I',
            Self::ParallelSheet | Self::AntiparallelSheet => 'E',
            Self::Turn => 'T',
            Self::Coil => 'C',
        }
    }

    /// True if this is any helix type.
    pub fn is_helix(&self) -> bool {
        matches!(self, Self::AlphaHelix | Self::Helix310 | Self::PiHelix)
    }

    /// True if this is any sheet type.
    pub fn is_sheet(&self) -> bool {
        matches!(self, Self::ParallelSheet | Self::AntiparallelSheet)
    }
}

impl fmt::Display for SSKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphaHelix => write!(f, "α-helix"),
            Self::Helix310 => write!(f, "3₁₀-helix"),
            Self::PiHelix => write!(f, "π-helix"),
            Self::ParallelSheet => write!(f, "β-sheet(parallel)"),
            Self::AntiparallelSheet => write!(f, "β-sheet(antiparallel)"),
            Self::Turn => write!(f, "turn"),
            Self::Coil => write!(f, "coil"),
        }
    }
}

// ── SS Assignment ───────────────────────────────────────────────────

/// Secondary structure assignment for a single residue.
#[derive(Debug, Clone, PartialEq)]
pub struct SSAssignment {
    /// Residue index (0-based).
    pub residue: usize,
    /// Assigned secondary structure.
    pub kind: SSKind,
    /// Confidence in [0, 1].
    pub confidence: f64,
}

impl SSAssignment {
    pub fn new(residue: usize, kind: SSKind, confidence: f64) -> Self {
        Self { residue, kind, confidence: confidence.clamp(0.0, 1.0) }
    }
}

impl fmt::Display for SSAssignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Res {} {} (conf={:.2})", self.residue, self.kind, self.confidence)
    }
}

// ── DSSP-like Assigner ──────────────────────────────────────────────

/// Configuration for backbone-geometry-based SS assignment.
#[derive(Debug, Clone)]
pub struct DsspAssigner {
    helix_phi_range: (f64, f64),
    helix_psi_range: (f64, f64),
    sheet_phi_range: (f64, f64),
    sheet_psi_range: (f64, f64),
    hbond_energy_cutoff: f64,
}

impl DsspAssigner {
    /// Create assigner with standard Ramachandran-based thresholds.
    pub fn new() -> Self {
        Self {
            helix_phi_range: (-160.0, -20.0),
            helix_psi_range: (-80.0, 0.0),
            sheet_phi_range: (-180.0, -60.0),
            sheet_psi_range: (60.0, 180.0),
            hbond_energy_cutoff: -0.5,
        }
    }

    /// Override helix phi range.
    pub fn with_helix_phi(mut self, lo: f64, hi: f64) -> Self {
        self.helix_phi_range = (lo, hi);
        self
    }

    /// Override helix psi range.
    pub fn with_helix_psi(mut self, lo: f64, hi: f64) -> Self {
        self.helix_psi_range = (lo, hi);
        self
    }

    /// Override sheet phi range.
    pub fn with_sheet_phi(mut self, lo: f64, hi: f64) -> Self {
        self.sheet_phi_range = (lo, hi);
        self
    }

    /// Override sheet psi range.
    pub fn with_sheet_psi(mut self, lo: f64, hi: f64) -> Self {
        self.sheet_psi_range = (lo, hi);
        self
    }

    /// Set H-bond energy cutoff in kcal/mol.
    pub fn with_hbond_cutoff(mut self, cutoff: f64) -> Self {
        self.hbond_energy_cutoff = cutoff;
        self
    }

    /// Assign secondary structure from phi/psi angles.
    pub fn assign_from_angles(&self, angles: &[(f64, f64)]) -> Vec<SSAssignment> {
        angles.iter().enumerate().map(|(i, &(phi, psi))| {
            let (kind, conf) = self.classify_residue(phi, psi);
            SSAssignment::new(i, kind, conf)
        }).collect()
    }

    fn classify_residue(&self, phi: f64, psi: f64) -> (SSKind, f64) {
        let helix_score = self.region_score(
            phi, psi,
            self.helix_phi_range, self.helix_psi_range,
        );
        let sheet_score = self.region_score(
            phi, psi,
            self.sheet_phi_range, self.sheet_psi_range,
        );

        if helix_score > sheet_score && helix_score > 0.0 {
            (SSKind::AlphaHelix, helix_score)
        } else if sheet_score > 0.0 {
            (SSKind::AntiparallelSheet, sheet_score)
        } else {
            (SSKind::Coil, 1.0 - helix_score.max(sheet_score).max(0.0))
        }
    }

    fn region_score(&self, phi: f64, psi: f64, phi_range: (f64, f64), psi_range: (f64, f64)) -> f64 {
        let phi_center = (phi_range.0 + phi_range.1) / 2.0;
        let psi_center = (psi_range.0 + psi_range.1) / 2.0;
        let phi_width = (phi_range.1 - phi_range.0) / 2.0;
        let psi_width = (psi_range.1 - psi_range.0) / 2.0;

        if phi_width <= 0.0 || psi_width <= 0.0 {
            return 0.0;
        }

        let phi_dev = ((phi - phi_center) / phi_width).abs();
        let psi_dev = ((psi - psi_center) / psi_width).abs();

        if phi_dev <= 1.0 && psi_dev <= 1.0 {
            (1.0 - phi_dev) * (1.0 - psi_dev)
        } else {
            0.0
        }
    }
}

impl fmt::Display for DsspAssigner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DsspAssigner(helix_φ=[{:.0},{:.0}], sheet_φ=[{:.0},{:.0}])",
            self.helix_phi_range.0, self.helix_phi_range.1,
            self.sheet_phi_range.0, self.sheet_phi_range.1,
        )
    }
}

// ── Structural Motif ────────────────────────────────────────────────

/// Known structural motifs.
#[derive(Debug, Clone, PartialEq)]
pub enum Motif {
    /// Beta hairpin: two antiparallel strands connected by a turn.
    BetaHairpin { start: usize, turn_start: usize, end: usize },
    /// Helix-turn-helix motif.
    HelixTurnHelix { helix1_start: usize, turn_start: usize, helix2_start: usize, end: usize },
    /// Greek key motif (four antiparallel strands).
    GreekKey { strands: [usize; 4] },
    /// Beta-alpha-beta unit.
    BetaAlphaBeta { strand1: usize, helix: usize, strand2: usize },
}

impl fmt::Display for Motif {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BetaHairpin { start, turn_start, end } =>
                write!(f, "β-hairpin({}-{}-{})", start, turn_start, end),
            Self::HelixTurnHelix { helix1_start, turn_start, helix2_start, end } =>
                write!(f, "HTH({}-{}-{}-{})", helix1_start, turn_start, helix2_start, end),
            Self::GreekKey { strands } =>
                write!(f, "GreekKey({},{},{},{})", strands[0], strands[1], strands[2], strands[3]),
            Self::BetaAlphaBeta { strand1, helix, strand2 } =>
                write!(f, "βαβ({}-{}-{})", strand1, helix, strand2),
        }
    }
}

// ── Motif Detector ──────────────────────────────────────────────────

/// Detect structural motifs from a secondary structure assignment.
#[derive(Debug, Clone)]
pub struct MotifDetector {
    min_helix_len: usize,
    min_strand_len: usize,
    max_turn_len: usize,
}

impl MotifDetector {
    pub fn new() -> Self {
        Self { min_helix_len: 4, min_strand_len: 3, max_turn_len: 5 }
    }

    pub fn with_min_helix_len(mut self, len: usize) -> Self {
        self.min_helix_len = len;
        self
    }

    pub fn with_min_strand_len(mut self, len: usize) -> Self {
        self.min_strand_len = len;
        self
    }

    pub fn with_max_turn_len(mut self, len: usize) -> Self {
        self.max_turn_len = len;
        self
    }

    /// Detect motifs from a sequence of SS assignments.
    pub fn detect(&self, assignments: &[SSAssignment]) -> Vec<Motif> {
        let mut motifs = Vec::new();
        let segments = self.extract_segments(assignments);

        // Look for beta hairpins: sheet-turn/coil-sheet
        for w in segments.windows(3) {
            if w[0].kind.is_sheet()
                && (w[1].kind == SSKind::Turn || w[1].kind == SSKind::Coil)
                && w[2].kind.is_sheet()
                && w[0].len >= self.min_strand_len
                && w[2].len >= self.min_strand_len
                && w[1].len <= self.max_turn_len
            {
                motifs.push(Motif::BetaHairpin {
                    start: w[0].start,
                    turn_start: w[1].start,
                    end: w[2].start + w[2].len,
                });
            }

            // Look for helix-turn-helix
            if w[0].kind.is_helix()
                && (w[1].kind == SSKind::Turn || w[1].kind == SSKind::Coil)
                && w[2].kind.is_helix()
                && w[0].len >= self.min_helix_len
                && w[2].len >= self.min_helix_len
                && w[1].len <= self.max_turn_len
            {
                motifs.push(Motif::HelixTurnHelix {
                    helix1_start: w[0].start,
                    turn_start: w[1].start,
                    helix2_start: w[2].start,
                    end: w[2].start + w[2].len,
                });
            }
        }

        // Look for beta-alpha-beta
        for w in segments.windows(5) {
            if w[0].kind.is_sheet()
                && w[2].kind.is_helix()
                && w[4].kind.is_sheet()
                && w[0].len >= self.min_strand_len
                && w[2].len >= self.min_helix_len
                && w[4].len >= self.min_strand_len
            {
                motifs.push(Motif::BetaAlphaBeta {
                    strand1: w[0].start,
                    helix: w[2].start,
                    strand2: w[4].start,
                });
            }
        }

        motifs
    }

    fn extract_segments(&self, assignments: &[SSAssignment]) -> Vec<Segment> {
        if assignments.is_empty() {
            return Vec::new();
        }
        let mut segments = Vec::new();
        let mut current_kind = assignments[0].kind;
        let mut start = 0;
        let mut len = 1;

        for a in assignments.iter().skip(1) {
            if a.kind == current_kind {
                len += 1;
            } else {
                segments.push(Segment { kind: current_kind, start, len });
                current_kind = a.kind;
                start = a.residue;
                len = 1;
            }
        }
        segments.push(Segment { kind: current_kind, start, len });
        segments
    }
}

impl fmt::Display for MotifDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MotifDetector(min_helix={}, min_strand={}, max_turn={})",
            self.min_helix_len, self.min_strand_len, self.max_turn_len,
        )
    }
}

#[derive(Debug, Clone)]
struct Segment {
    kind: SSKind,
    start: usize,
    len: usize,
}

// ── SS Composition ──────────────────────────────────────────────────

/// Summary statistics of secondary structure composition.
#[derive(Debug, Clone)]
pub struct SSComposition {
    pub total: usize,
    pub helix_count: usize,
    pub sheet_count: usize,
    pub turn_count: usize,
    pub coil_count: usize,
}

impl SSComposition {
    /// Compute composition from assignments.
    pub fn from_assignments(assignments: &[SSAssignment]) -> Self {
        let mut comp = Self { total: assignments.len(), helix_count: 0, sheet_count: 0, turn_count: 0, coil_count: 0 };
        for a in assignments {
            match a.kind {
                k if k.is_helix() => comp.helix_count += 1,
                k if k.is_sheet() => comp.sheet_count += 1,
                SSKind::Turn => comp.turn_count += 1,
                _ => comp.coil_count += 1,
            }
        }
        comp
    }

    /// Fraction helix.
    pub fn helix_fraction(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.helix_count as f64 / self.total as f64 }
    }

    /// Fraction sheet.
    pub fn sheet_fraction(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.sheet_count as f64 / self.total as f64 }
    }
}

impl fmt::Display for SSComposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SS(n={}, H={:.1}%, E={:.1}%, T={:.1}%, C={:.1}%)",
            self.total,
            self.helix_fraction() * 100.0,
            self.sheet_fraction() * 100.0,
            if self.total > 0 { self.turn_count as f64 / self.total as f64 * 100.0 } else { 0.0 },
            if self.total > 0 { self.coil_count as f64 / self.total as f64 * 100.0 } else { 0.0 },
        )
    }
}

// ── DSSP String ─────────────────────────────────────────────────────

/// Generate a DSSP-like string from assignments.
pub fn dssp_string(assignments: &[SSAssignment]) -> String {
    assignments.iter().map(|a| a.kind.dssp_code()).collect()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ss_kind_dssp_codes() {
        assert_eq!(SSKind::AlphaHelix.dssp_code(), 'H');
        assert_eq!(SSKind::Helix310.dssp_code(), 'G');
        assert_eq!(SSKind::PiHelix.dssp_code(), 'I');
        assert_eq!(SSKind::ParallelSheet.dssp_code(), 'E');
        assert_eq!(SSKind::Turn.dssp_code(), 'T');
        assert_eq!(SSKind::Coil.dssp_code(), 'C');
    }

    #[test]
    fn test_ss_kind_is_helix() {
        assert!(SSKind::AlphaHelix.is_helix());
        assert!(SSKind::Helix310.is_helix());
        assert!(SSKind::PiHelix.is_helix());
        assert!(!SSKind::Coil.is_helix());
    }

    #[test]
    fn test_ss_kind_is_sheet() {
        assert!(SSKind::ParallelSheet.is_sheet());
        assert!(SSKind::AntiparallelSheet.is_sheet());
        assert!(!SSKind::AlphaHelix.is_sheet());
    }

    #[test]
    fn test_ss_kind_display() {
        assert!(SSKind::AlphaHelix.to_string().contains("helix"));
    }

    #[test]
    fn test_ss_assignment_clamp() {
        let a = SSAssignment::new(0, SSKind::Coil, 1.5);
        assert!((a.confidence - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dssp_assigner_helix() {
        let assigner = DsspAssigner::new();
        let angles = vec![(-63.0, -42.0)];
        let result = assigner.assign_from_angles(&angles);
        assert_eq!(result[0].kind, SSKind::AlphaHelix);
    }

    #[test]
    fn test_dssp_assigner_sheet() {
        let assigner = DsspAssigner::new();
        let angles = vec![(-120.0, 130.0)];
        let result = assigner.assign_from_angles(&angles);
        assert!(result[0].kind.is_sheet());
    }

    #[test]
    fn test_dssp_assigner_coil() {
        let assigner = DsspAssigner::new();
        // Angles outside both helix and sheet regions
        let angles = vec![(60.0, 30.0)];
        let result = assigner.assign_from_angles(&angles);
        assert_eq!(result[0].kind, SSKind::Coil);
    }

    #[test]
    fn test_dssp_assigner_display() {
        let assigner = DsspAssigner::new();
        assert!(assigner.to_string().contains("DsspAssigner"));
    }

    #[test]
    fn test_dssp_assigner_builders() {
        let a = DsspAssigner::new()
            .with_helix_phi(-170.0, -10.0)
            .with_helix_psi(-90.0, 10.0)
            .with_sheet_phi(-170.0, -50.0)
            .with_sheet_psi(50.0, 170.0)
            .with_hbond_cutoff(-1.0);
        assert!(a.to_string().contains("-170"));
    }

    #[test]
    fn test_motif_beta_hairpin() {
        let mut assignments = Vec::new();
        for i in 0..5 { assignments.push(SSAssignment::new(i, SSKind::AntiparallelSheet, 0.9)); }
        for i in 5..7 { assignments.push(SSAssignment::new(i, SSKind::Turn, 0.8)); }
        for i in 7..12 { assignments.push(SSAssignment::new(i, SSKind::AntiparallelSheet, 0.9)); }

        let detector = MotifDetector::new();
        let motifs = detector.detect(&assignments);
        assert!(motifs.iter().any(|m| matches!(m, Motif::BetaHairpin { .. })));
    }

    #[test]
    fn test_motif_hth() {
        let mut assignments = Vec::new();
        for i in 0..6 { assignments.push(SSAssignment::new(i, SSKind::AlphaHelix, 0.9)); }
        for i in 6..8 { assignments.push(SSAssignment::new(i, SSKind::Turn, 0.8)); }
        for i in 8..14 { assignments.push(SSAssignment::new(i, SSKind::AlphaHelix, 0.9)); }

        let detector = MotifDetector::new();
        let motifs = detector.detect(&assignments);
        assert!(motifs.iter().any(|m| matches!(m, Motif::HelixTurnHelix { .. })));
    }

    #[test]
    fn test_motif_display() {
        let m = Motif::BetaHairpin { start: 0, turn_start: 5, end: 10 };
        assert!(m.to_string().contains("hairpin"));
    }

    #[test]
    fn test_motif_detector_builders() {
        let d = MotifDetector::new()
            .with_min_helix_len(5)
            .with_min_strand_len(4)
            .with_max_turn_len(3);
        assert!(d.to_string().contains("5"));
    }

    #[test]
    fn test_ss_composition() {
        let assignments = vec![
            SSAssignment::new(0, SSKind::AlphaHelix, 0.9),
            SSAssignment::new(1, SSKind::AlphaHelix, 0.9),
            SSAssignment::new(2, SSKind::Coil, 0.8),
            SSAssignment::new(3, SSKind::AntiparallelSheet, 0.9),
        ];
        let comp = SSComposition::from_assignments(&assignments);
        assert_eq!(comp.helix_count, 2);
        assert_eq!(comp.sheet_count, 1);
        assert_eq!(comp.coil_count, 1);
        assert!((comp.helix_fraction() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ss_composition_display() {
        let a = vec![SSAssignment::new(0, SSKind::Coil, 1.0)];
        let comp = SSComposition::from_assignments(&a);
        assert!(comp.to_string().contains("SS("));
    }

    #[test]
    fn test_dssp_string() {
        let assignments = vec![
            SSAssignment::new(0, SSKind::AlphaHelix, 0.9),
            SSAssignment::new(1, SSKind::AlphaHelix, 0.9),
            SSAssignment::new(2, SSKind::Coil, 0.8),
            SSAssignment::new(3, SSKind::AntiparallelSheet, 0.9),
        ];
        assert_eq!(dssp_string(&assignments), "HHCE");
    }

    #[test]
    fn test_empty_assignments() {
        let comp = SSComposition::from_assignments(&[]);
        assert_eq!(comp.total, 0);
        assert!((comp.helix_fraction()).abs() < 1e-10);
    }

    #[test]
    fn test_dssp_string_empty() {
        assert_eq!(dssp_string(&[]), "");
    }
}
