//! Disulfide bridge detection, cysteine pairing, and oxidation state analysis.
//!
//! Detects disulfide bonds (S-S bridges) between cysteine residues from
//! 3D coordinates, validates bridge geometry, and classifies oxidation
//! states for structural analysis and protein engineering.

use std::fmt;

// ── Oxidation State ─────────────────────────────────────────────────

/// Oxidation state of a cysteine residue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OxidationState {
    /// Reduced (free thiol, -SH).
    Reduced,
    /// Oxidized (part of a disulfide bridge, -S-S-).
    Oxidized,
    /// Unknown / not determined.
    Unknown,
}

impl OxidationState {
    /// Three-letter residue code.
    pub fn residue_code(&self) -> &str {
        match self {
            Self::Reduced => "CYS",
            Self::Oxidized => "CYX",
            Self::Unknown => "CYS",
        }
    }
}

impl fmt::Display for OxidationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reduced => write!(f, "reduced(-SH)"),
            Self::Oxidized => write!(f, "oxidized(-S-S-)"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ── Cysteine Residue ────────────────────────────────────────────────

/// A cysteine residue with its sulfur atom coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct CysteineResidue {
    /// Residue index.
    pub residue: usize,
    /// Chain identifier.
    pub chain: char,
    /// Sγ (sulfur gamma) atom coordinates in angstroms.
    pub sg_coords: [f64; 3],
    /// Cβ atom coordinates (if known).
    pub cb_coords: Option<[f64; 3]>,
    /// Cα atom coordinates (if known).
    pub ca_coords: Option<[f64; 3]>,
    /// Current oxidation state.
    pub state: OxidationState,
}

impl CysteineResidue {
    pub fn new(residue: usize, chain: char, sg_coords: [f64; 3]) -> Self {
        Self {
            residue, chain, sg_coords,
            cb_coords: None, ca_coords: None,
            state: OxidationState::Unknown,
        }
    }

    /// Set the Cβ coordinates.
    pub fn with_cb(mut self, coords: [f64; 3]) -> Self {
        self.cb_coords = Some(coords);
        self
    }

    /// Set the Cα coordinates.
    pub fn with_ca(mut self, coords: [f64; 3]) -> Self {
        self.ca_coords = Some(coords);
        self
    }

    /// Set the oxidation state.
    pub fn with_state(mut self, state: OxidationState) -> Self {
        self.state = state;
        self
    }

    /// Distance of Sγ to another cysteine's Sγ in angstroms.
    pub fn sg_distance(&self, other: &CysteineResidue) -> f64 {
        let dx = self.sg_coords[0] - other.sg_coords[0];
        let dy = self.sg_coords[1] - other.sg_coords[1];
        let dz = self.sg_coords[2] - other.sg_coords[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Cβ-Sγ bond length (if Cβ is known).
    pub fn cb_sg_distance(&self) -> Option<f64> {
        self.cb_coords.map(|cb| {
            let dx = cb[0] - self.sg_coords[0];
            let dy = cb[1] - self.sg_coords[1];
            let dz = cb[2] - self.sg_coords[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
    }
}

impl fmt::Display for CysteineResidue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cys {} {} SG=({:.2},{:.2},{:.2}) [{}]",
            self.chain, self.residue,
            self.sg_coords[0], self.sg_coords[1], self.sg_coords[2],
            self.state,
        )
    }
}

// ── Disulfide Bridge ────────────────────────────────────────────────

/// A disulfide bridge between two cysteine residues.
#[derive(Debug, Clone, PartialEq)]
pub struct DisulfideBridge {
    /// First cysteine (residue index, chain).
    pub cys1_residue: usize,
    pub cys1_chain: char,
    /// Second cysteine (residue index, chain).
    pub cys2_residue: usize,
    pub cys2_chain: char,
    /// Sγ-Sγ distance in angstroms.
    pub ss_distance: f64,
    /// Cβ-Sγ-Sγ-Cβ dihedral angle in degrees (χ₃, if available).
    pub chi3_angle: Option<f64>,
    /// Bridge classification.
    pub bridge_type: BridgeType,
}

impl DisulfideBridge {
    /// True if this is an inter-chain (inter-molecular) bridge.
    pub fn is_interchain(&self) -> bool {
        self.cys1_chain != self.cys2_chain
    }

    /// True if this is an intra-chain bridge.
    pub fn is_intrachain(&self) -> bool {
        self.cys1_chain == self.cys2_chain
    }

    /// Sequence separation (only meaningful for intrachain bridges).
    pub fn sequence_separation(&self) -> Option<usize> {
        if self.is_intrachain() {
            Some(if self.cys1_residue > self.cys2_residue {
                self.cys1_residue - self.cys2_residue
            } else {
                self.cys2_residue - self.cys1_residue
            })
        } else {
            None
        }
    }

    /// Validate bridge geometry.
    pub fn is_valid(&self) -> bool {
        self.ss_distance >= 1.8 && self.ss_distance <= 2.2
    }
}

impl fmt::Display for DisulfideBridge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SS({}{}-{}{}, d={:.2}Å, {})",
            self.cys1_chain, self.cys1_residue,
            self.cys2_chain, self.cys2_residue,
            self.ss_distance, self.bridge_type,
        )
    }
}

// ── Bridge Type ─────────────────────────────────────────────────────

/// Classification of disulfide bridge geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BridgeType {
    /// Spiral (χ₃ ~ ±90°): most common.
    Spiral,
    /// Hook (χ₃ ~ ±60°).
    Hook,
    /// Staple (χ₃ ~ ±120°).
    Staple,
    /// Short-range: sequence separation < 10.
    ShortRange,
    /// Long-range: sequence separation >= 25.
    LongRange,
    /// Unknown geometry.
    Unknown,
}

impl BridgeType {
    /// Classify from χ₃ dihedral angle.
    pub fn from_chi3(chi3: f64) -> Self {
        let abs_chi3 = chi3.abs();
        if (abs_chi3 - 90.0).abs() < 20.0 {
            Self::Spiral
        } else if (abs_chi3 - 60.0).abs() < 15.0 {
            Self::Hook
        } else if (abs_chi3 - 120.0).abs() < 15.0 {
            Self::Staple
        } else {
            Self::Unknown
        }
    }
}

impl fmt::Display for BridgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spiral => write!(f, "spiral"),
            Self::Hook => write!(f, "hook"),
            Self::Staple => write!(f, "staple"),
            Self::ShortRange => write!(f, "short-range"),
            Self::LongRange => write!(f, "long-range"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ── Bridge Detector ─────────────────────────────────────────────────

/// Configuration for disulfide bridge detection.
#[derive(Debug, Clone)]
pub struct BridgeDetector {
    /// Maximum Sγ-Sγ distance to consider a bridge.
    max_ss_distance: f64,
    /// Minimum Sγ-Sγ distance (too close = clash).
    min_ss_distance: f64,
    /// Ideal S-S bond length.
    ideal_ss_distance: f64,
}

impl BridgeDetector {
    pub fn new() -> Self {
        Self {
            max_ss_distance: 2.5,
            min_ss_distance: 1.5,
            ideal_ss_distance: 2.03,
        }
    }

    pub fn with_max_distance(mut self, d: f64) -> Self {
        self.max_ss_distance = d;
        self
    }

    pub fn with_min_distance(mut self, d: f64) -> Self {
        self.min_ss_distance = d;
        self
    }

    /// Detect disulfide bridges from a list of cysteine residues.
    pub fn detect(&self, cysteines: &[CysteineResidue]) -> Vec<DisulfideBridge> {
        let mut bridges = Vec::new();
        let n = cysteines.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = cysteines[i].sg_distance(&cysteines[j]);
                if dist >= self.min_ss_distance && dist <= self.max_ss_distance {
                    let chi3 = compute_chi3(&cysteines[i], &cysteines[j]);
                    let bridge_type = chi3.map(BridgeType::from_chi3).unwrap_or(BridgeType::Unknown);

                    bridges.push(DisulfideBridge {
                        cys1_residue: cysteines[i].residue,
                        cys1_chain: cysteines[i].chain,
                        cys2_residue: cysteines[j].residue,
                        cys2_chain: cysteines[j].chain,
                        ss_distance: dist,
                        chi3_angle: chi3,
                        bridge_type,
                    });
                }
            }
        }

        bridges
    }

    /// Detect and assign oxidation states to cysteines.
    pub fn detect_and_assign(&self, cysteines: &mut [CysteineResidue]) -> Vec<DisulfideBridge> {
        let bridges = self.detect(cysteines);

        // Mark bridged cysteines as oxidized
        for b in &bridges {
            for c in cysteines.iter_mut() {
                if (c.residue == b.cys1_residue && c.chain == b.cys1_chain)
                    || (c.residue == b.cys2_residue && c.chain == b.cys2_chain)
                {
                    c.state = OxidationState::Oxidized;
                }
            }
        }

        // Remaining cysteines are reduced
        for c in cysteines.iter_mut() {
            if c.state == OxidationState::Unknown {
                c.state = OxidationState::Reduced;
            }
        }

        bridges
    }
}

impl fmt::Display for BridgeDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BridgeDetector(SS=[{:.1},{:.1}]Å, ideal={:.2}Å)",
            self.min_ss_distance, self.max_ss_distance, self.ideal_ss_distance,
        )
    }
}

// ── Bridge Summary ──────────────────────────────────────────────────

/// Summary statistics for disulfide bridges in a structure.
#[derive(Debug, Clone)]
pub struct BridgeSummary {
    pub total_cysteines: usize,
    pub bridge_count: usize,
    pub intrachain_count: usize,
    pub interchain_count: usize,
    pub free_cysteines: usize,
    pub average_ss_distance: f64,
}

impl BridgeSummary {
    /// Compute summary from detected bridges and total cysteine count.
    pub fn from_bridges(bridges: &[DisulfideBridge], total_cysteines: usize) -> Self {
        let intra = bridges.iter().filter(|b| b.is_intrachain()).count();
        let inter = bridges.iter().filter(|b| b.is_interchain()).count();
        let bonded = bridges.len() * 2;
        let free = if total_cysteines >= bonded { total_cysteines - bonded } else { 0 };

        let avg_d = if bridges.is_empty() {
            0.0
        } else {
            bridges.iter().map(|b| b.ss_distance).sum::<f64>() / bridges.len() as f64
        };

        Self {
            total_cysteines,
            bridge_count: bridges.len(),
            intrachain_count: intra,
            interchain_count: inter,
            free_cysteines: free,
            average_ss_distance: avg_d,
        }
    }
}

impl fmt::Display for BridgeSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bridges(n={}, intra={}, inter={}, free_cys={}, avg_d={:.2}Å)",
            self.bridge_count, self.intrachain_count, self.interchain_count,
            self.free_cysteines, self.average_ss_distance,
        )
    }
}

// ── χ₃ Dihedral ─────────────────────────────────────────────────────

/// Compute the Cβ-Sγ-Sγ-Cβ dihedral (χ₃) if Cβ coordinates are available.
fn compute_chi3(cys1: &CysteineResidue, cys2: &CysteineResidue) -> Option<f64> {
    let cb1 = cys1.cb_coords?;
    let cb2 = cys2.cb_coords?;
    Some(dihedral_angle(cb1, cys1.sg_coords, cys2.sg_coords, cb2))
}

/// Compute dihedral angle (in degrees) from four 3D points.
fn dihedral_angle(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> f64 {
    let b1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let b2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
    let b3 = [p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]];

    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);

    let b2_len = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
    if b2_len < 1e-12 {
        return 0.0;
    }
    let b2u = [b2[0] / b2_len, b2[1] / b2_len, b2[2] / b2_len];
    let m1 = cross(n1, b2u);

    let x = dot(n1, n2);
    let y = dot(m1, n2);
    (-y.atan2(x)).to_degrees()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_oxidation_state_display() {
        assert!(OxidationState::Reduced.to_string().contains("SH"));
        assert!(OxidationState::Oxidized.to_string().contains("S-S"));
    }

    #[test]
    fn test_oxidation_state_residue_code() {
        assert_eq!(OxidationState::Reduced.residue_code(), "CYS");
        assert_eq!(OxidationState::Oxidized.residue_code(), "CYX");
    }

    #[test]
    fn test_cysteine_new() {
        let c = CysteineResidue::new(10, 'A', [1.0, 2.0, 3.0]);
        assert_eq!(c.residue, 10);
        assert_eq!(c.chain, 'A');
        assert_eq!(c.state, OxidationState::Unknown);
    }

    #[test]
    fn test_cysteine_builders() {
        let c = CysteineResidue::new(5, 'B', [0.0; 3])
            .with_cb([1.0, 0.0, 0.0])
            .with_ca([2.0, 0.0, 0.0])
            .with_state(OxidationState::Reduced);
        assert!(c.cb_coords.is_some());
        assert!(c.ca_coords.is_some());
        assert_eq!(c.state, OxidationState::Reduced);
    }

    #[test]
    fn test_cysteine_sg_distance() {
        let c1 = CysteineResidue::new(0, 'A', [0.0, 0.0, 0.0]);
        let c2 = CysteineResidue::new(1, 'A', [2.03, 0.0, 0.0]);
        assert!(approx(c1.sg_distance(&c2), 2.03, 1e-10));
    }

    #[test]
    fn test_cysteine_cb_sg_distance() {
        let c = CysteineResidue::new(0, 'A', [0.0, 0.0, 0.0])
            .with_cb([1.81, 0.0, 0.0]);
        assert!(approx(c.cb_sg_distance().unwrap(), 1.81, 1e-10));
    }

    #[test]
    fn test_cysteine_display() {
        let c = CysteineResidue::new(42, 'A', [1.0, 2.0, 3.0]);
        assert!(c.to_string().contains("Cys"));
        assert!(c.to_string().contains("42"));
    }

    #[test]
    fn test_bridge_detection() {
        let cysteines = vec![
            CysteineResidue::new(10, 'A', [0.0, 0.0, 0.0]),
            CysteineResidue::new(50, 'A', [2.03, 0.0, 0.0]),
            CysteineResidue::new(80, 'A', [20.0, 20.0, 20.0]),
        ];
        let detector = BridgeDetector::new();
        let bridges = detector.detect(&cysteines);
        assert_eq!(bridges.len(), 1);
        assert!(approx(bridges[0].ss_distance, 2.03, 1e-10));
    }

    #[test]
    fn test_bridge_interchain() {
        let b = DisulfideBridge {
            cys1_residue: 10, cys1_chain: 'A',
            cys2_residue: 20, cys2_chain: 'B',
            ss_distance: 2.03, chi3_angle: None, bridge_type: BridgeType::Unknown,
        };
        assert!(b.is_interchain());
        assert!(!b.is_intrachain());
        assert!(b.sequence_separation().is_none());
    }

    #[test]
    fn test_bridge_intrachain() {
        let b = DisulfideBridge {
            cys1_residue: 10, cys1_chain: 'A',
            cys2_residue: 50, cys2_chain: 'A',
            ss_distance: 2.03, chi3_angle: None, bridge_type: BridgeType::Unknown,
        };
        assert!(b.is_intrachain());
        assert_eq!(b.sequence_separation(), Some(40));
    }

    #[test]
    fn test_bridge_valid() {
        let b = DisulfideBridge {
            cys1_residue: 0, cys1_chain: 'A',
            cys2_residue: 1, cys2_chain: 'A',
            ss_distance: 2.03, chi3_angle: None, bridge_type: BridgeType::Unknown,
        };
        assert!(b.is_valid());

        let bad = DisulfideBridge {
            cys1_residue: 0, cys1_chain: 'A',
            cys2_residue: 1, cys2_chain: 'A',
            ss_distance: 3.0, chi3_angle: None, bridge_type: BridgeType::Unknown,
        };
        assert!(!bad.is_valid());
    }

    #[test]
    fn test_bridge_display() {
        let b = DisulfideBridge {
            cys1_residue: 10, cys1_chain: 'A',
            cys2_residue: 50, cys2_chain: 'A',
            ss_distance: 2.03, chi3_angle: None, bridge_type: BridgeType::Spiral,
        };
        let s = b.to_string();
        assert!(s.contains("SS("));
        assert!(s.contains("spiral"));
    }

    #[test]
    fn test_bridge_type_from_chi3() {
        assert_eq!(BridgeType::from_chi3(87.0), BridgeType::Spiral);
        assert_eq!(BridgeType::from_chi3(-93.0), BridgeType::Spiral);
        assert_eq!(BridgeType::from_chi3(58.0), BridgeType::Hook);
        assert_eq!(BridgeType::from_chi3(118.0), BridgeType::Staple);
        assert_eq!(BridgeType::from_chi3(0.0), BridgeType::Unknown);
    }

    #[test]
    fn test_bridge_type_display() {
        assert_eq!(BridgeType::Spiral.to_string(), "spiral");
        assert_eq!(BridgeType::Hook.to_string(), "hook");
    }

    #[test]
    fn test_detect_and_assign() {
        let mut cysteines = vec![
            CysteineResidue::new(10, 'A', [0.0, 0.0, 0.0]),
            CysteineResidue::new(50, 'A', [2.03, 0.0, 0.0]),
            CysteineResidue::new(80, 'A', [20.0, 20.0, 20.0]),
        ];
        let detector = BridgeDetector::new();
        let bridges = detector.detect_and_assign(&mut cysteines);
        assert_eq!(bridges.len(), 1);
        assert_eq!(cysteines[0].state, OxidationState::Oxidized);
        assert_eq!(cysteines[1].state, OxidationState::Oxidized);
        assert_eq!(cysteines[2].state, OxidationState::Reduced);
    }

    #[test]
    fn test_bridge_summary() {
        let bridges = vec![DisulfideBridge {
            cys1_residue: 10, cys1_chain: 'A',
            cys2_residue: 50, cys2_chain: 'A',
            ss_distance: 2.03, chi3_angle: None, bridge_type: BridgeType::Spiral,
        }];
        let summary = BridgeSummary::from_bridges(&bridges, 5);
        assert_eq!(summary.bridge_count, 1);
        assert_eq!(summary.intrachain_count, 1);
        assert_eq!(summary.free_cysteines, 3);
    }

    #[test]
    fn test_bridge_summary_display() {
        let summary = BridgeSummary::from_bridges(&[], 4);
        assert!(summary.to_string().contains("Bridges("));
    }

    #[test]
    fn test_detector_builders() {
        let d = BridgeDetector::new()
            .with_max_distance(2.8)
            .with_min_distance(1.6);
        assert!(d.to_string().contains("2.8"));
    }

    #[test]
    fn test_detector_display() {
        let d = BridgeDetector::new();
        assert!(d.to_string().contains("BridgeDetector"));
    }
}
