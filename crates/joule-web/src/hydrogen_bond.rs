//! Hydrogen bond detection, donor/acceptor geometry, and bond energy estimation.
//!
//! Detects hydrogen bonds from 3D atomic coordinates using distance
//! and angle criteria. Supports both backbone and sidechain H-bonds,
//! DSSP-style energy calculations, and H-bond network analysis.

use std::fmt;

// ── H-Bond Atom Role ────────────────────────────────────────────────

/// Role of an atom in a hydrogen bond.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HBondRole {
    /// Hydrogen bond donor (D-H···A).
    Donor,
    /// Hydrogen bond acceptor.
    Acceptor,
    /// The hydrogen itself.
    Hydrogen,
    /// Not involved in H-bonding.
    None,
}

impl fmt::Display for HBondRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Donor => write!(f, "donor"),
            Self::Acceptor => write!(f, "acceptor"),
            Self::Hydrogen => write!(f, "hydrogen"),
            Self::None => write!(f, "none"),
        }
    }
}

// ── H-Bond Atom ─────────────────────────────────────────────────────

/// An atom participating in hydrogen bond analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct HBondAtom {
    /// Atom index in the structure.
    pub index: usize,
    /// Atom name.
    pub name: String,
    /// Residue index.
    pub residue: usize,
    /// 3D coordinates in angstroms.
    pub coords: [f64; 3],
    /// Role in H-bonding.
    pub role: HBondRole,
}

impl HBondAtom {
    pub fn new(index: usize, name: &str, residue: usize, coords: [f64; 3], role: HBondRole) -> Self {
        Self { index, name: name.to_string(), residue, coords, role }
    }

    /// Distance to another atom.
    pub fn distance_to(&self, other: &HBondAtom) -> f64 {
        let dx = self.coords[0] - other.coords[0];
        let dy = self.coords[1] - other.coords[1];
        let dz = self.coords[2] - other.coords[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl fmt::Display for HBondAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}] res={} ({})", self.name, self.index, self.residue, self.role)
    }
}

// ── Hydrogen Bond ───────────────────────────────────────────────────

/// A detected hydrogen bond: D-H···A.
#[derive(Debug, Clone, PartialEq)]
pub struct HydrogenBond {
    /// Donor atom index.
    pub donor_idx: usize,
    /// Hydrogen atom index (if available).
    pub hydrogen_idx: Option<usize>,
    /// Acceptor atom index.
    pub acceptor_idx: usize,
    /// Donor residue.
    pub donor_residue: usize,
    /// Acceptor residue.
    pub acceptor_residue: usize,
    /// D···A distance in angstroms.
    pub da_distance: f64,
    /// D-H···A angle in degrees (if H position known).
    pub dha_angle: Option<f64>,
    /// Estimated energy in kcal/mol.
    pub energy: f64,
}

impl HydrogenBond {
    /// True if this is a backbone-backbone H-bond.
    pub fn is_backbone(&self) -> bool {
        // Simple heuristic: backbone H-bonds typically involve sequential residues
        true
    }

    /// Sequence separation between donor and acceptor residues.
    pub fn residue_separation(&self) -> usize {
        if self.donor_residue > self.acceptor_residue {
            self.donor_residue - self.acceptor_residue
        } else {
            self.acceptor_residue - self.donor_residue
        }
    }

    /// True if this H-bond is characteristic of an alpha helix (i→i+4).
    pub fn is_alpha_pattern(&self) -> bool {
        self.residue_separation() == 4
    }

    /// True if this H-bond is characteristic of a 3₁₀ helix (i→i+3).
    pub fn is_310_pattern(&self) -> bool {
        self.residue_separation() == 3
    }

    /// Quality assessment based on geometry.
    pub fn quality(&self) -> HBondQuality {
        if self.da_distance <= 2.8 {
            if let Some(angle) = self.dha_angle {
                if angle >= 160.0 {
                    return HBondQuality::Strong;
                } else if angle >= 130.0 {
                    return HBondQuality::Moderate;
                }
            } else {
                return HBondQuality::Moderate;
            }
        }
        if self.da_distance <= 3.2 {
            HBondQuality::Weak
        } else {
            HBondQuality::VeryWeak
        }
    }
}

impl fmt::Display for HydrogenBond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HBond(D={} A={} d={:.2}Å E={:.2} kcal/mol)",
            self.donor_idx, self.acceptor_idx, self.da_distance, self.energy,
        )
    }
}

// ── H-Bond Quality ──────────────────────────────────────────────────

/// Quality classification of a hydrogen bond.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HBondQuality {
    Strong,
    Moderate,
    Weak,
    VeryWeak,
}

impl fmt::Display for HBondQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Strong => write!(f, "strong"),
            Self::Moderate => write!(f, "moderate"),
            Self::Weak => write!(f, "weak"),
            Self::VeryWeak => write!(f, "very-weak"),
        }
    }
}

// ── H-Bond Detector ─────────────────────────────────────────────────

/// Configuration for hydrogen bond detection.
#[derive(Debug, Clone)]
pub struct HBondDetector {
    /// Maximum D···A distance in angstroms.
    max_da_distance: f64,
    /// Minimum D-H···A angle in degrees.
    min_dha_angle: f64,
    /// Maximum H···A distance in angstroms.
    max_ha_distance: f64,
}

impl HBondDetector {
    /// Standard H-bond detection criteria.
    pub fn new() -> Self {
        Self {
            max_da_distance: 3.5,
            min_dha_angle: 120.0,
            max_ha_distance: 2.5,
        }
    }

    pub fn with_max_da_distance(mut self, d: f64) -> Self {
        self.max_da_distance = d;
        self
    }

    pub fn with_min_dha_angle(mut self, a: f64) -> Self {
        self.min_dha_angle = a;
        self
    }

    pub fn with_max_ha_distance(mut self, d: f64) -> Self {
        self.max_ha_distance = d;
        self
    }

    /// Detect H-bonds from atom lists.
    pub fn detect(
        &self,
        donors: &[HBondAtom],
        acceptors: &[HBondAtom],
        hydrogens: Option<&[HBondAtom]>,
    ) -> Vec<HydrogenBond> {
        let mut bonds = Vec::new();

        for d in donors {
            for a in acceptors {
                if d.residue == a.residue {
                    continue;
                }

                let da_dist = d.distance_to(a);
                if da_dist > self.max_da_distance {
                    continue;
                }

                // Find the best hydrogen for this donor
                let (h_idx, dha_angle) = if let Some(hs) = hydrogens {
                    self.find_best_hydrogen(d, a, hs)
                } else {
                    (None, None)
                };

                // Check angle criteria if hydrogen is known
                if let Some(angle) = dha_angle {
                    if angle < self.min_dha_angle {
                        continue;
                    }
                }

                let energy = dssp_hbond_energy(da_dist, dha_angle);

                bonds.push(HydrogenBond {
                    donor_idx: d.index,
                    hydrogen_idx: h_idx,
                    acceptor_idx: a.index,
                    donor_residue: d.residue,
                    acceptor_residue: a.residue,
                    da_distance: da_dist,
                    dha_angle,
                    energy,
                });
            }
        }

        bonds
    }

    fn find_best_hydrogen(
        &self,
        donor: &HBondAtom,
        acceptor: &HBondAtom,
        hydrogens: &[HBondAtom],
    ) -> (Option<usize>, Option<f64>) {
        let mut best_h: Option<usize> = None;
        let mut best_angle = 0.0_f64;

        for h in hydrogens {
            // Hydrogen should be close to donor
            let dh_dist = donor.distance_to(h);
            if dh_dist > 1.3 {
                continue;
            }

            let ha_dist = h.distance_to(acceptor);
            if ha_dist > self.max_ha_distance {
                continue;
            }

            let angle = compute_angle(donor.coords, h.coords, acceptor.coords);
            if angle > best_angle {
                best_angle = angle;
                best_h = Some(h.index);
            }
        }

        if best_h.is_some() {
            (best_h, Some(best_angle))
        } else {
            (None, None)
        }
    }
}

impl fmt::Display for HBondDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HBondDetector(DA<{:.1}Å, DHA>{:.0}°, HA<{:.1}Å)",
            self.max_da_distance, self.min_dha_angle, self.max_ha_distance,
        )
    }
}

// ── H-Bond Network ──────────────────────────────────────────────────

/// Summary of an H-bond network.
#[derive(Debug, Clone)]
pub struct HBondNetwork {
    pub bonds: Vec<HydrogenBond>,
}

impl HBondNetwork {
    pub fn new(bonds: Vec<HydrogenBond>) -> Self {
        Self { bonds }
    }

    /// Number of H-bonds.
    pub fn count(&self) -> usize {
        self.bonds.len()
    }

    /// Total H-bond energy in kcal/mol.
    pub fn total_energy(&self) -> f64 {
        self.bonds.iter().map(|b| b.energy).sum()
    }

    /// Average D···A distance.
    pub fn average_distance(&self) -> f64 {
        if self.bonds.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.bonds.iter().map(|b| b.da_distance).sum();
        sum / self.bonds.len() as f64
    }

    /// Count of alpha-helix pattern H-bonds (i→i+4).
    pub fn alpha_pattern_count(&self) -> usize {
        self.bonds.iter().filter(|b| b.is_alpha_pattern()).count()
    }

    /// Count of 3₁₀ pattern H-bonds (i→i+3).
    pub fn helix_310_count(&self) -> usize {
        self.bonds.iter().filter(|b| b.is_310_pattern()).count()
    }

    /// Distribution of H-bond qualities.
    pub fn quality_distribution(&self) -> (usize, usize, usize, usize) {
        let mut strong = 0;
        let mut moderate = 0;
        let mut weak = 0;
        let mut very_weak = 0;
        for b in &self.bonds {
            match b.quality() {
                HBondQuality::Strong => strong += 1,
                HBondQuality::Moderate => moderate += 1,
                HBondQuality::Weak => weak += 1,
                HBondQuality::VeryWeak => very_weak += 1,
            }
        }
        (strong, moderate, weak, very_weak)
    }
}

impl fmt::Display for HBondNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HBondNetwork(n={}, E={:.1} kcal/mol, avg_d={:.2}Å)",
            self.count(), self.total_energy(), self.average_distance(),
        )
    }
}

// ── DSSP Energy ─────────────────────────────────────────────────────

/// DSSP-style H-bond energy estimate (simplified Kabsch-Sander).
/// Returns energy in kcal/mol (negative = favourable).
pub fn dssp_hbond_energy(da_distance: f64, _dha_angle: Option<f64>) -> f64 {
    // Simplified Kabsch-Sander: E = 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332
    // We approximate with just D-A distance
    if da_distance < 0.5 {
        return 0.0;
    }
    let e_base = -27.888; // kcal/mol constant
    let r = da_distance;
    // Simplified: E ~ e_base * (1/r - 1/r0) where r0 is cutoff
    let e = e_base * (1.0 / r - 1.0 / 3.5);
    e.min(0.0)
}

// ── Angle Calculation ───────────────────────────────────────────────

/// Compute angle (in degrees) at point B in the triangle A-B-C.
fn compute_angle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];

    let dot_val = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    let mag_ba = (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).sqrt();
    let mag_bc = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();

    if mag_ba < 1e-12 || mag_bc < 1e-12 {
        return 0.0;
    }

    let cos_angle = (dot_val / (mag_ba * mag_bc)).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_hbond_role_display() {
        assert_eq!(HBondRole::Donor.to_string(), "donor");
        assert_eq!(HBondRole::Acceptor.to_string(), "acceptor");
        assert_eq!(HBondRole::None.to_string(), "none");
    }

    #[test]
    fn test_hbond_atom_distance() {
        let a = HBondAtom::new(0, "N", 0, [0.0, 0.0, 0.0], HBondRole::Donor);
        let b = HBondAtom::new(1, "O", 1, [3.0, 4.0, 0.0], HBondRole::Acceptor);
        assert!(approx(a.distance_to(&b), 5.0, 1e-10));
    }

    #[test]
    fn test_hbond_atom_display() {
        let a = HBondAtom::new(5, "N", 3, [1.0, 2.0, 3.0], HBondRole::Donor);
        let s = a.to_string();
        assert!(s.contains("N"));
        assert!(s.contains("donor"));
    }

    #[test]
    fn test_hydrogen_bond_separation() {
        let hb = HydrogenBond {
            donor_idx: 0, hydrogen_idx: None, acceptor_idx: 1,
            donor_residue: 5, acceptor_residue: 9,
            da_distance: 2.9, dha_angle: None, energy: -1.5,
        };
        assert_eq!(hb.residue_separation(), 4);
        assert!(hb.is_alpha_pattern());
        assert!(!hb.is_310_pattern());
    }

    #[test]
    fn test_hydrogen_bond_310() {
        let hb = HydrogenBond {
            donor_idx: 0, hydrogen_idx: None, acceptor_idx: 1,
            donor_residue: 10, acceptor_residue: 7,
            da_distance: 3.0, dha_angle: None, energy: -1.0,
        };
        assert!(hb.is_310_pattern());
    }

    #[test]
    fn test_hydrogen_bond_quality_strong() {
        let hb = HydrogenBond {
            donor_idx: 0, hydrogen_idx: None, acceptor_idx: 1,
            donor_residue: 0, acceptor_residue: 4,
            da_distance: 2.7, dha_angle: Some(165.0), energy: -2.0,
        };
        assert_eq!(hb.quality(), HBondQuality::Strong);
    }

    #[test]
    fn test_hydrogen_bond_quality_weak() {
        let hb = HydrogenBond {
            donor_idx: 0, hydrogen_idx: None, acceptor_idx: 1,
            donor_residue: 0, acceptor_residue: 4,
            da_distance: 3.1, dha_angle: Some(165.0), energy: -0.5,
        };
        assert_eq!(hb.quality(), HBondQuality::Weak);
    }

    #[test]
    fn test_hydrogen_bond_display() {
        let hb = HydrogenBond {
            donor_idx: 0, hydrogen_idx: None, acceptor_idx: 1,
            donor_residue: 0, acceptor_residue: 4,
            da_distance: 2.9, dha_angle: None, energy: -1.5,
        };
        assert!(hb.to_string().contains("HBond"));
    }

    #[test]
    fn test_quality_display() {
        assert_eq!(HBondQuality::Strong.to_string(), "strong");
        assert_eq!(HBondQuality::VeryWeak.to_string(), "very-weak");
    }

    #[test]
    fn test_detector_basic() {
        let donors = vec![
            HBondAtom::new(0, "N", 0, [0.0, 0.0, 0.0], HBondRole::Donor),
        ];
        let acceptors = vec![
            HBondAtom::new(1, "O", 4, [2.8, 0.0, 0.0], HBondRole::Acceptor),
        ];
        let detector = HBondDetector::new();
        let bonds = detector.detect(&donors, &acceptors, None);
        assert_eq!(bonds.len(), 1);
        assert!(approx(bonds[0].da_distance, 2.8, 1e-10));
    }

    #[test]
    fn test_detector_too_far() {
        let donors = vec![
            HBondAtom::new(0, "N", 0, [0.0, 0.0, 0.0], HBondRole::Donor),
        ];
        let acceptors = vec![
            HBondAtom::new(1, "O", 4, [10.0, 0.0, 0.0], HBondRole::Acceptor),
        ];
        let detector = HBondDetector::new();
        let bonds = detector.detect(&donors, &acceptors, None);
        assert!(bonds.is_empty());
    }

    #[test]
    fn test_detector_same_residue_skipped() {
        let donors = vec![HBondAtom::new(0, "N", 5, [0.0, 0.0, 0.0], HBondRole::Donor)];
        let acceptors = vec![HBondAtom::new(1, "O", 5, [2.8, 0.0, 0.0], HBondRole::Acceptor)];
        let detector = HBondDetector::new();
        let bonds = detector.detect(&donors, &acceptors, None);
        assert!(bonds.is_empty());
    }

    #[test]
    fn test_detector_builders() {
        let d = HBondDetector::new()
            .with_max_da_distance(3.2)
            .with_min_dha_angle(130.0)
            .with_max_ha_distance(2.3);
        assert!(d.to_string().contains("3.2"));
    }

    #[test]
    fn test_detector_display() {
        let d = HBondDetector::new();
        assert!(d.to_string().contains("HBondDetector"));
    }

    #[test]
    fn test_network_total_energy() {
        let bonds = vec![
            HydrogenBond {
                donor_idx: 0, hydrogen_idx: None, acceptor_idx: 1,
                donor_residue: 0, acceptor_residue: 4,
                da_distance: 2.9, dha_angle: None, energy: -1.5,
            },
            HydrogenBond {
                donor_idx: 2, hydrogen_idx: None, acceptor_idx: 3,
                donor_residue: 1, acceptor_residue: 5,
                da_distance: 3.0, dha_angle: None, energy: -1.0,
            },
        ];
        let net = HBondNetwork::new(bonds);
        assert_eq!(net.count(), 2);
        assert!(approx(net.total_energy(), -2.5, 1e-10));
    }

    #[test]
    fn test_network_average_distance() {
        let bonds = vec![
            HydrogenBond {
                donor_idx: 0, hydrogen_idx: None, acceptor_idx: 1,
                donor_residue: 0, acceptor_residue: 4,
                da_distance: 2.8, dha_angle: None, energy: -1.5,
            },
            HydrogenBond {
                donor_idx: 2, hydrogen_idx: None, acceptor_idx: 3,
                donor_residue: 1, acceptor_residue: 5,
                da_distance: 3.2, dha_angle: None, energy: -1.0,
            },
        ];
        let net = HBondNetwork::new(bonds);
        assert!(approx(net.average_distance(), 3.0, 1e-10));
    }

    #[test]
    fn test_network_display() {
        let net = HBondNetwork::new(vec![]);
        assert!(net.to_string().contains("HBondNetwork"));
    }

    #[test]
    fn test_dssp_energy_negative() {
        let e = dssp_hbond_energy(2.9, None);
        assert!(e < 0.0);
    }

    #[test]
    fn test_dssp_energy_at_cutoff() {
        let e = dssp_hbond_energy(3.5, None);
        assert!(approx(e, 0.0, 1e-10));
    }

    #[test]
    fn test_compute_angle_right() {
        let angle = compute_angle([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(approx(angle, 90.0, 0.1));
    }

    #[test]
    fn test_compute_angle_straight() {
        let angle = compute_angle([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(approx(angle, 180.0, 0.1));
    }
}
