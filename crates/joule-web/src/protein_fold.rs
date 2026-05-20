//! Protein folding energy landscape, contact maps, and backbone dihedral angles.
//!
//! Models the energy surface that governs protein folding, including
//! residue-residue contact maps, phi/psi backbone torsion angles, and
//! simplified energy potentials based on hydrophobic collapse and
//! hydrogen-bonding contributions.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────────

/// Default contact distance threshold in angstroms (Cα-Cα).
const DEFAULT_CONTACT_THRESHOLD: f64 = 8.0;

/// Boltzmann constant in kcal/(mol·K).
const KB_KCAL: f64 = 0.001987204;

// ── Backbone Dihedral ───────────────────────────────────────────────

/// Backbone dihedral angles (phi, psi) for a single residue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackboneDihedral {
    /// Residue index (0-based).
    pub residue: usize,
    /// Phi angle in degrees (C-N-Cα-C).
    pub phi: f64,
    /// Psi angle in degrees (N-Cα-C-N).
    pub psi: f64,
}

impl BackboneDihedral {
    /// Create a new backbone dihedral record.
    pub fn new(residue: usize, phi: f64, psi: f64) -> Self {
        Self { residue, phi, psi }
    }

    /// Returns true if the angles lie in the alpha-helix region.
    pub fn is_alpha_helix(&self) -> bool {
        (-160.0..=-20.0).contains(&self.phi) && (-80.0..=0.0).contains(&self.psi)
    }

    /// Returns true if the angles lie in the beta-sheet region.
    pub fn is_beta_sheet(&self) -> bool {
        (-180.0..=-60.0).contains(&self.phi) && (60.0..=180.0).contains(&self.psi)
    }

    /// Convert phi to radians.
    pub fn phi_rad(&self) -> f64 {
        self.phi.to_radians()
    }

    /// Convert psi to radians.
    pub fn psi_rad(&self) -> f64 {
        self.psi.to_radians()
    }
}

impl fmt::Display for BackboneDihedral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Res {} φ={:.1}° ψ={:.1}°", self.residue, self.phi, self.psi)
    }
}

// ── Contact ─────────────────────────────────────────────────────────

/// A residue-residue contact in a contact map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contact {
    /// Index of the first residue.
    pub i: usize,
    /// Index of the second residue.
    pub j: usize,
    /// Distance in angstroms.
    pub distance: f64,
}

impl Contact {
    pub fn new(i: usize, j: usize, distance: f64) -> Self {
        Self { i, j, distance }
    }

    /// True if this is a long-range contact (sequence separation >= 12).
    pub fn is_long_range(&self) -> bool {
        let sep = if self.i > self.j { self.i - self.j } else { self.j - self.i };
        sep >= 12
    }

    /// Sequence separation between the two residues.
    pub fn sequence_separation(&self) -> usize {
        if self.i > self.j { self.i - self.j } else { self.j - self.i }
    }
}

impl fmt::Display for Contact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Contact({}-{}, {:.2} Å)", self.i, self.j, self.distance)
    }
}

// ── Contact Map ─────────────────────────────────────────────────────

/// Binary contact map for a protein structure.
#[derive(Debug, Clone)]
pub struct ContactMap {
    size: usize,
    threshold: f64,
    contacts: Vec<Contact>,
    matrix: Vec<bool>,
}

impl ContactMap {
    /// Build a contact map from Cα coordinates with default threshold.
    pub fn from_coordinates(coords: &[[f64; 3]]) -> Self {
        Self::from_coordinates_with_threshold(coords, DEFAULT_CONTACT_THRESHOLD)
    }

    /// Build a contact map with a custom distance threshold.
    pub fn from_coordinates_with_threshold(coords: &[[f64; 3]], threshold: f64) -> Self {
        let n = coords.len();
        let mut contacts = Vec::new();
        let mut matrix = vec![false; n * n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist <= threshold {
                    contacts.push(Contact::new(i, j, dist));
                    matrix[i * n + j] = true;
                    matrix[j * n + i] = true;
                }
            }
        }

        Self { size: n, threshold, contacts, matrix }
    }

    /// Number of residues.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Distance threshold used.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// All contacts found.
    pub fn contacts(&self) -> &[Contact] {
        &self.contacts
    }

    /// Check whether residues i and j are in contact.
    pub fn in_contact(&self, i: usize, j: usize) -> bool {
        if i >= self.size || j >= self.size {
            return false;
        }
        self.matrix[i * self.size + j]
    }

    /// Contact density (fraction of possible contacts that exist).
    pub fn density(&self) -> f64 {
        if self.size < 2 {
            return 0.0;
        }
        let possible = self.size * (self.size - 1) / 2;
        self.contacts.len() as f64 / possible as f64
    }

    /// Number of long-range contacts (separation >= 12).
    pub fn long_range_count(&self) -> usize {
        self.contacts.iter().filter(|c| c.is_long_range()).count()
    }

    /// Contact order: average sequence separation of contacts divided by chain length.
    pub fn contact_order(&self) -> f64 {
        if self.contacts.is_empty() || self.size == 0 {
            return 0.0;
        }
        let total_sep: usize = self.contacts.iter().map(|c| c.sequence_separation()).sum();
        total_sep as f64 / (self.contacts.len() as f64 * self.size as f64)
    }
}

impl fmt::Display for ContactMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ContactMap(n={}, contacts={}, threshold={:.1} Å)",
            self.size,
            self.contacts.len(),
            self.threshold,
        )
    }
}

// ── Energy Potential ────────────────────────────────────────────────

/// Type of residue for coarse-grained energy calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidueType {
    Hydrophobic,
    Polar,
    Charged,
}

impl fmt::Display for ResidueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hydrophobic => write!(f, "H"),
            Self::Polar => write!(f, "P"),
            Self::Charged => write!(f, "C"),
        }
    }
}

/// Simple contact energy potential.
#[derive(Debug, Clone)]
pub struct ContactEnergy {
    /// Interaction matrix: [ResidueType pair] -> energy in kcal/mol.
    hh: f64,
    hp: f64,
    hc: f64,
    pp: f64,
    pc: f64,
    cc: f64,
}

impl ContactEnergy {
    /// Default Miyazawa-Jernigan-like contact energies.
    pub fn default_potential() -> Self {
        Self { hh: -3.0, hp: -1.0, hc: -1.5, pp: -1.0, pc: -2.0, cc: -4.0 }
    }

    /// Custom contact energy potential.
    pub fn new(hh: f64, hp: f64, hc: f64, pp: f64, pc: f64, cc: f64) -> Self {
        Self { hh, hp, hc, pp, pc, cc }
    }

    /// Get interaction energy between two residue types.
    pub fn energy(&self, a: ResidueType, b: ResidueType) -> f64 {
        use ResidueType::*;
        match (a, b) {
            (Hydrophobic, Hydrophobic) => self.hh,
            (Hydrophobic, Polar) | (Polar, Hydrophobic) => self.hp,
            (Hydrophobic, Charged) | (Charged, Hydrophobic) => self.hc,
            (Polar, Polar) => self.pp,
            (Polar, Charged) | (Charged, Polar) => self.pc,
            (Charged, Charged) => self.cc,
        }
    }

    /// Total energy of a conformation given contacts and residue types.
    pub fn total_energy(&self, contacts: &[Contact], types: &[ResidueType]) -> f64 {
        contacts.iter().map(|c| {
            if c.i < types.len() && c.j < types.len() {
                self.energy(types[c.i], types[c.j])
            } else {
                0.0
            }
        }).sum()
    }
}

impl fmt::Display for ContactEnergy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContactEnergy(HH={:.1}, HP={:.1}, CC={:.1})", self.hh, self.hp, self.cc)
    }
}

// ── Folding Landscape ───────────────────────────────────────────────

/// A point on the folding energy landscape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandscapePoint {
    /// Reaction coordinate (e.g., fraction of native contacts).
    pub reaction_coord: f64,
    /// Free energy in kcal/mol.
    pub free_energy: f64,
}

impl LandscapePoint {
    pub fn new(reaction_coord: f64, free_energy: f64) -> Self {
        Self { reaction_coord, free_energy }
    }
}

impl fmt::Display for LandscapePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.2} kcal/mol)", self.reaction_coord, self.free_energy)
    }
}

/// Simplified two-state folding landscape.
#[derive(Debug, Clone)]
pub struct FoldingLandscape {
    temperature: f64,
    native_energy: f64,
    unfolded_energy: f64,
    barrier_height: f64,
    points: Vec<LandscapePoint>,
}

impl FoldingLandscape {
    /// Create a new folding landscape.
    pub fn new(native_energy: f64, unfolded_energy: f64, barrier_height: f64) -> Self {
        Self {
            temperature: 300.0,
            native_energy,
            unfolded_energy,
            barrier_height,
            points: Vec::new(),
        }
    }

    /// Set the temperature in Kelvin.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Temperature in Kelvin.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Stability: ΔG = G_unfolded - G_native (negative means native is stable).
    pub fn stability(&self) -> f64 {
        self.unfolded_energy - self.native_energy
    }

    /// Folding rate estimate from Arrhenius-like expression (relative units).
    pub fn folding_rate(&self) -> f64 {
        let barrier = self.barrier_height - self.unfolded_energy;
        (-barrier / (KB_KCAL * self.temperature)).exp()
    }

    /// Unfolding rate estimate.
    pub fn unfolding_rate(&self) -> f64 {
        let barrier = self.barrier_height - self.native_energy;
        (-barrier / (KB_KCAL * self.temperature)).exp()
    }

    /// Equilibrium constant K_eq = k_fold / k_unfold.
    pub fn equilibrium_constant(&self) -> f64 {
        let k_unfold = self.unfolding_rate();
        if k_unfold == 0.0 {
            return f64::INFINITY;
        }
        self.folding_rate() / k_unfold
    }

    /// Fraction folded at equilibrium.
    pub fn fraction_folded(&self) -> f64 {
        let keq = self.equilibrium_constant();
        keq / (1.0 + keq)
    }

    /// Sample the landscape at uniform intervals along the reaction coordinate.
    pub fn sample(&mut self, n_points: usize) {
        self.points.clear();
        for i in 0..n_points {
            let q = i as f64 / (n_points - 1).max(1) as f64;
            // Simple double-well potential: G(q) = a*q^4 - b*q^2 + c*q + d
            let energy = self.double_well_energy(q);
            self.points.push(LandscapePoint::new(q, energy));
        }
    }

    fn double_well_energy(&self, q: f64) -> f64 {
        // Polynomial that passes through unfolded (q=0), barrier (q~0.5), native (q=1)
        let a = 4.0 * (self.barrier_height - 0.5 * (self.native_energy + self.unfolded_energy));
        let linear = self.native_energy - self.unfolded_energy;
        self.unfolded_energy + a * q * q * (1.0 - q) * (1.0 - q)
            / 0.0625_f64.max(0.0001)
            * 0.0625
            + linear * q
    }

    /// Points sampled on the landscape.
    pub fn points(&self) -> &[LandscapePoint] {
        &self.points
    }
}

impl fmt::Display for FoldingLandscape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FoldingLandscape(ΔG={:.2} kcal/mol, T={:.0} K, frac_folded={:.3})",
            self.stability(),
            self.temperature,
            self.fraction_folded(),
        )
    }
}

// ── Fraction of Native Contacts ─────────────────────────────────────

/// Compute fraction of native contacts (Q) present in a decoy structure.
pub fn fraction_native_contacts(
    native: &ContactMap,
    decoy_coords: &[[f64; 3]],
    tolerance: f64,
) -> f64 {
    if native.contacts().is_empty() {
        return 0.0;
    }
    let mut matched = 0usize;
    for c in native.contacts() {
        if c.i >= decoy_coords.len() || c.j >= decoy_coords.len() {
            continue;
        }
        let dx = decoy_coords[c.i][0] - decoy_coords[c.j][0];
        let dy = decoy_coords[c.i][1] - decoy_coords[c.j][1];
        let dz = decoy_coords[c.i][2] - decoy_coords[c.j][2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist <= native.threshold() + tolerance {
            matched += 1;
        }
    }
    matched as f64 / native.contacts().len() as f64
}

// ── Dihedral Angle from 4 Points ────────────────────────────────────

/// Compute a dihedral angle (in degrees) from four 3D points.
pub fn dihedral_angle(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> f64 {
    let b1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let b2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
    let b3 = [p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]];

    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let m1 = cross(n1, b2_normalized(b2));

    let x = dot(n1, n2);
    let y = dot(m1, n2);
    (-y.atan2(x)).to_degrees()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn b2_normalized(b: [f64; 3]) -> [f64; 3] {
    let len = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [b[0] / len, b[1] / len, b[2] / len]
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_backbone_dihedral_alpha() {
        let d = BackboneDihedral::new(0, -60.0, -45.0);
        assert!(d.is_alpha_helix());
        assert!(!d.is_beta_sheet());
    }

    #[test]
    fn test_backbone_dihedral_beta() {
        let d = BackboneDihedral::new(1, -120.0, 130.0);
        assert!(d.is_beta_sheet());
        assert!(!d.is_alpha_helix());
    }

    #[test]
    fn test_backbone_radians() {
        let d = BackboneDihedral::new(0, 180.0, -90.0);
        assert!(approx(d.phi_rad(), std::f64::consts::PI, 1e-10));
        assert!(approx(d.psi_rad(), -std::f64::consts::FRAC_PI_2, 1e-10));
    }

    #[test]
    fn test_backbone_display() {
        let d = BackboneDihedral::new(5, -63.0, -42.0);
        assert!(d.to_string().contains("Res 5"));
    }

    #[test]
    fn test_contact_long_range() {
        let c = Contact::new(2, 20, 6.5);
        assert!(c.is_long_range());
        assert_eq!(c.sequence_separation(), 18);
    }

    #[test]
    fn test_contact_short_range() {
        let c = Contact::new(5, 8, 5.0);
        assert!(!c.is_long_range());
        assert_eq!(c.sequence_separation(), 3);
    }

    #[test]
    fn test_contact_map_simple() {
        let coords = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [20.0, 0.0, 0.0]];
        let cm = ContactMap::from_coordinates(&coords);
        assert!(cm.in_contact(0, 1));
        assert!(!cm.in_contact(0, 2));
        assert_eq!(cm.size(), 3);
    }

    #[test]
    fn test_contact_map_density() {
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let cm = ContactMap::from_coordinates(&coords);
        // All three are within 8 Å: density = 3/3 = 1.0
        assert!(approx(cm.density(), 1.0, 1e-10));
    }

    #[test]
    fn test_contact_map_contact_order() {
        let coords = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let cm = ContactMap::from_coordinates(&coords);
        let co = cm.contact_order();
        // 3 contacts: (0,1)=1, (0,2)=2, (1,2)=1 -> sum=4, N=3, C=3 -> 4/9
        assert!(approx(co, 4.0 / 9.0, 1e-10));
    }

    #[test]
    fn test_contact_map_display() {
        let coords = vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let cm = ContactMap::from_coordinates(&coords);
        assert!(cm.to_string().contains("ContactMap"));
    }

    #[test]
    fn test_contact_energy_hh() {
        let ce = ContactEnergy::default_potential();
        assert!(approx(ce.energy(ResidueType::Hydrophobic, ResidueType::Hydrophobic), -3.0, 1e-10));
    }

    #[test]
    fn test_contact_energy_symmetric() {
        let ce = ContactEnergy::default_potential();
        assert_eq!(
            ce.energy(ResidueType::Hydrophobic, ResidueType::Polar),
            ce.energy(ResidueType::Polar, ResidueType::Hydrophobic),
        );
    }

    #[test]
    fn test_total_energy() {
        let ce = ContactEnergy::default_potential();
        let contacts = vec![Contact::new(0, 1, 5.0)];
        let types = vec![ResidueType::Hydrophobic, ResidueType::Hydrophobic];
        assert!(approx(ce.total_energy(&contacts, &types), -3.0, 1e-10));
    }

    #[test]
    fn test_folding_landscape_stability() {
        let fl = FoldingLandscape::new(-10.0, 0.0, 5.0);
        assert!(approx(fl.stability(), 10.0, 1e-10));
    }

    #[test]
    fn test_folding_landscape_fraction_folded() {
        let fl = FoldingLandscape::new(-15.0, 0.0, 10.0);
        let ff = fl.fraction_folded();
        // Very stable protein should be mostly folded
        assert!(ff > 0.9);
    }

    #[test]
    fn test_folding_landscape_with_temperature() {
        let fl = FoldingLandscape::new(-10.0, 0.0, 5.0).with_temperature(400.0);
        assert!(approx(fl.temperature(), 400.0, 1e-10));
    }

    #[test]
    fn test_folding_landscape_sample() {
        let mut fl = FoldingLandscape::new(-10.0, 0.0, 5.0);
        fl.sample(50);
        assert_eq!(fl.points().len(), 50);
    }

    #[test]
    fn test_fraction_native_contacts() {
        let coords = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let native = ContactMap::from_coordinates(&coords);
        let q = fraction_native_contacts(&native, &coords, 0.5);
        assert!(approx(q, 1.0, 1e-10));
    }

    #[test]
    fn test_fraction_native_contacts_partial() {
        let coords = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let native = ContactMap::from_coordinates(&coords);
        // Move residue 2 far away
        let decoy = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [60.0, 0.0, 0.0]];
        let q = fraction_native_contacts(&native, &decoy, 0.5);
        assert!(q < 1.0);
        assert!(q > 0.0);
    }

    #[test]
    fn test_dihedral_angle_planar() {
        // Four coplanar points forming a known dihedral
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];
        let p4 = [0.0, 1.0, 1.0];
        let angle = dihedral_angle(p1, p2, p3, p4);
        // Should be close to 90 degrees (or -90)
        assert!(approx(angle.abs(), 90.0, 1.0));
    }

    #[test]
    fn test_residue_type_display() {
        assert_eq!(ResidueType::Hydrophobic.to_string(), "H");
        assert_eq!(ResidueType::Polar.to_string(), "P");
        assert_eq!(ResidueType::Charged.to_string(), "C");
    }
}
