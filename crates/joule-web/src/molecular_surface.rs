//! Solvent-accessible surface area, van der Waals radii, and surface calculations.
//!
//! Implements Shrake-Rupley and Lee-Richards style algorithms for
//! computing solvent-accessible surface area (SASA) from 3D atomic
//! coordinates and element-specific van der Waals radii.

use std::fmt;

// ── Van der Waals Radii ─────────────────────────────────────────────

/// Van der Waals radius lookup for common biological elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    Carbon,
    Nitrogen,
    Oxygen,
    Sulfur,
    Hydrogen,
    Phosphorus,
    Selenium,
    Other,
}

impl Element {
    /// Parse from a 1-2 character element symbol.
    pub fn from_symbol(sym: &str) -> Self {
        match sym.trim().to_uppercase().as_str() {
            "C" => Self::Carbon,
            "N" => Self::Nitrogen,
            "O" => Self::Oxygen,
            "S" => Self::Sulfur,
            "H" | "D" => Self::Hydrogen,
            "P" => Self::Phosphorus,
            "SE" => Self::Selenium,
            _ => Self::Other,
        }
    }

    /// Van der Waals radius in angstroms.
    pub fn vdw_radius(&self) -> f64 {
        match self {
            Self::Carbon => 1.70,
            Self::Nitrogen => 1.55,
            Self::Oxygen => 1.52,
            Self::Sulfur => 1.80,
            Self::Hydrogen => 1.20,
            Self::Phosphorus => 1.80,
            Self::Selenium => 1.90,
            Self::Other => 1.70,
        }
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Carbon => write!(f, "C"),
            Self::Nitrogen => write!(f, "N"),
            Self::Oxygen => write!(f, "O"),
            Self::Sulfur => write!(f, "S"),
            Self::Hydrogen => write!(f, "H"),
            Self::Phosphorus => write!(f, "P"),
            Self::Selenium => write!(f, "Se"),
            Self::Other => write!(f, "X"),
        }
    }
}

// ── Surface Atom ────────────────────────────────────────────────────

/// Atom with surface area information.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceAtom {
    /// Atom index.
    pub index: usize,
    /// 3D coordinates in angstroms.
    pub coords: [f64; 3],
    /// Element type.
    pub element: Element,
    /// Van der Waals radius in angstroms.
    pub vdw_radius: f64,
    /// Computed solvent-accessible surface area in Å².
    pub sasa: f64,
}

impl SurfaceAtom {
    pub fn new(index: usize, coords: [f64; 3], element: Element) -> Self {
        let vdw_radius = element.vdw_radius();
        Self { index, coords, element, vdw_radius, sasa: 0.0 }
    }

    /// Override the van der Waals radius.
    pub fn with_radius(mut self, radius: f64) -> Self {
        self.vdw_radius = radius;
        self
    }

    /// Effective radius = vdW + probe radius.
    pub fn effective_radius(&self, probe: f64) -> f64 {
        self.vdw_radius + probe
    }

    /// Distance to another atom.
    pub fn distance_to(&self, other: &SurfaceAtom) -> f64 {
        let dx = self.coords[0] - other.coords[0];
        let dy = self.coords[1] - other.coords[1];
        let dz = self.coords[2] - other.coords[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl fmt::Display for SurfaceAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Atom {} {} r={:.2}Å SASA={:.1}Å²",
            self.index, self.element, self.vdw_radius, self.sasa,
        )
    }
}

// ── SASA Calculator ─────────────────────────────────────────────────

/// Shrake-Rupley SASA calculator.
#[derive(Debug, Clone)]
pub struct SasaCalculator {
    /// Probe radius in angstroms (default: 1.4 Å for water).
    probe_radius: f64,
    /// Number of test points per atom sphere.
    n_points: usize,
}

impl SasaCalculator {
    /// Default calculator with water probe and 960 test points.
    pub fn new() -> Self {
        Self { probe_radius: 1.4, n_points: 960 }
    }

    /// Set probe radius.
    pub fn with_probe_radius(mut self, r: f64) -> Self {
        self.probe_radius = r;
        self
    }

    /// Set number of sphere test points.
    pub fn with_n_points(mut self, n: usize) -> Self {
        self.n_points = n.max(12);
        self
    }

    /// Compute SASA for all atoms (Shrake-Rupley algorithm).
    pub fn compute(&self, atoms: &mut [SurfaceAtom]) -> SasaResult {
        let n = atoms.len();
        let sphere_points = golden_spiral_points(self.n_points);
        let mut total_sasa = 0.0;

        for i in 0..n {
            let ri = atoms[i].effective_radius(self.probe_radius);
            let ci = atoms[i].coords;
            let mut accessible = 0usize;

            for sp in &sphere_points {
                let test_x = ci[0] + ri * sp[0];
                let test_y = ci[1] + ri * sp[1];
                let test_z = ci[2] + ri * sp[2];

                let mut buried = false;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let rj = atoms[j].effective_radius(self.probe_radius);
                    let dx = test_x - atoms[j].coords[0];
                    let dy = test_y - atoms[j].coords[1];
                    let dz = test_z - atoms[j].coords[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < rj * rj {
                        buried = true;
                        break;
                    }
                }
                if !buried {
                    accessible += 1;
                }
            }

            let fraction = accessible as f64 / self.n_points as f64;
            let area = 4.0 * std::f64::consts::PI * ri * ri * fraction;
            atoms[i].sasa = area;
            total_sasa += area;
        }

        SasaResult {
            total_sasa,
            atom_count: n,
            probe_radius: self.probe_radius,
        }
    }
}

impl fmt::Display for SasaCalculator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SasaCalculator(probe={:.2}Å, points={})",
            self.probe_radius, self.n_points,
        )
    }
}

// ── SASA Result ─────────────────────────────────────────────────────

/// Result of a SASA computation.
#[derive(Debug, Clone)]
pub struct SasaResult {
    /// Total SASA in Å².
    pub total_sasa: f64,
    /// Number of atoms.
    pub atom_count: usize,
    /// Probe radius used.
    pub probe_radius: f64,
}

impl SasaResult {
    /// Average SASA per atom.
    pub fn average_sasa(&self) -> f64 {
        if self.atom_count == 0 { 0.0 } else { self.total_sasa / self.atom_count as f64 }
    }
}

impl fmt::Display for SasaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SASA(total={:.1}Å², n={}, probe={:.2}Å)",
            self.total_sasa, self.atom_count, self.probe_radius,
        )
    }
}

// ── Surface Classification ──────────────────────────────────────────

/// Classification of atom surface exposure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceExposure {
    /// Fully buried (SASA = 0).
    Buried,
    /// Partially exposed.
    Partial,
    /// Fully exposed.
    Exposed,
}

impl SurfaceExposure {
    /// Classify from SASA value and a maximum reference.
    pub fn classify(sasa: f64, max_sasa: f64) -> Self {
        if max_sasa <= 0.0 {
            return Self::Buried;
        }
        let ratio = sasa / max_sasa;
        if ratio < 0.05 {
            Self::Buried
        } else if ratio < 0.5 {
            Self::Partial
        } else {
            Self::Exposed
        }
    }
}

impl fmt::Display for SurfaceExposure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buried => write!(f, "buried"),
            Self::Partial => write!(f, "partial"),
            Self::Exposed => write!(f, "exposed"),
        }
    }
}

// ── Relative SASA ───────────────────────────────────────────────────

/// Maximum SASA reference values for common amino acids (Gly-X-Gly tripeptide, Å²).
pub fn max_sasa_reference(res_name: &str) -> f64 {
    match res_name.trim().to_uppercase().as_str() {
        "ALA" => 129.0,
        "ARG" => 274.0,
        "ASN" => 195.0,
        "ASP" => 193.0,
        "CYS" => 167.0,
        "GLN" => 225.0,
        "GLU" => 223.0,
        "GLY" => 104.0,
        "HIS" => 224.0,
        "ILE" => 197.0,
        "LEU" => 201.0,
        "LYS" => 236.0,
        "MET" => 224.0,
        "PHE" => 240.0,
        "PRO" => 159.0,
        "SER" => 155.0,
        "THR" => 172.0,
        "TRP" => 285.0,
        "TYR" => 263.0,
        "VAL" => 174.0,
        _ => 200.0,
    }
}

// ── Golden Spiral Point Generation ──────────────────────────────────

/// Generate approximately uniformly distributed points on a unit sphere.
fn golden_spiral_points(n: usize) -> Vec<[f64; 3]> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        let y = 1.0 - (i as f64 / (n - 1).max(1) as f64) * 2.0;
        let radius = (1.0 - y * y).sqrt();
        let theta = golden_angle * i as f64;
        points.push([radius * theta.cos(), y, radius * theta.sin()]);
    }

    points
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_element_from_symbol() {
        assert_eq!(Element::from_symbol("C"), Element::Carbon);
        assert_eq!(Element::from_symbol("N"), Element::Nitrogen);
        assert_eq!(Element::from_symbol("O"), Element::Oxygen);
        assert_eq!(Element::from_symbol("S"), Element::Sulfur);
        assert_eq!(Element::from_symbol("SE"), Element::Selenium);
        assert_eq!(Element::from_symbol("ZZ"), Element::Other);
    }

    #[test]
    fn test_vdw_radii() {
        assert!(approx(Element::Carbon.vdw_radius(), 1.70, 1e-2));
        assert!(approx(Element::Oxygen.vdw_radius(), 1.52, 1e-2));
        assert!(approx(Element::Hydrogen.vdw_radius(), 1.20, 1e-2));
    }

    #[test]
    fn test_element_display() {
        assert_eq!(Element::Carbon.to_string(), "C");
        assert_eq!(Element::Selenium.to_string(), "Se");
    }

    #[test]
    fn test_surface_atom_new() {
        let a = SurfaceAtom::new(0, [1.0, 2.0, 3.0], Element::Carbon);
        assert!(approx(a.vdw_radius, 1.70, 1e-2));
        assert!(approx(a.sasa, 0.0, 1e-10));
    }

    #[test]
    fn test_surface_atom_with_radius() {
        let a = SurfaceAtom::new(0, [0.0; 3], Element::Carbon).with_radius(2.0);
        assert!(approx(a.vdw_radius, 2.0, 1e-10));
    }

    #[test]
    fn test_surface_atom_effective_radius() {
        let a = SurfaceAtom::new(0, [0.0; 3], Element::Carbon);
        assert!(approx(a.effective_radius(1.4), 3.1, 1e-2));
    }

    #[test]
    fn test_surface_atom_distance() {
        let a = SurfaceAtom::new(0, [0.0, 0.0, 0.0], Element::Carbon);
        let b = SurfaceAtom::new(1, [3.0, 4.0, 0.0], Element::Nitrogen);
        assert!(approx(a.distance_to(&b), 5.0, 1e-10));
    }

    #[test]
    fn test_surface_atom_display() {
        let a = SurfaceAtom::new(0, [0.0; 3], Element::Oxygen);
        assert!(a.to_string().contains("O"));
    }

    #[test]
    fn test_sasa_single_atom() {
        let mut atoms = vec![SurfaceAtom::new(0, [0.0, 0.0, 0.0], Element::Carbon)];
        let calc = SasaCalculator::new().with_n_points(200);
        let result = calc.compute(&mut atoms);
        // Isolated atom: SASA = 4π(r+probe)²
        let expected = 4.0 * std::f64::consts::PI * (1.7 + 1.4) * (1.7 + 1.4);
        assert!(approx(result.total_sasa, expected, expected * 0.05));
    }

    #[test]
    fn test_sasa_two_atoms_less_than_isolated() {
        let mut isolated = vec![SurfaceAtom::new(0, [0.0, 0.0, 0.0], Element::Carbon)];
        let calc = SasaCalculator::new().with_n_points(200);
        let iso_result = calc.compute(&mut isolated);

        let mut pair = vec![
            SurfaceAtom::new(0, [0.0, 0.0, 0.0], Element::Carbon),
            SurfaceAtom::new(1, [3.0, 0.0, 0.0], Element::Carbon),
        ];
        let pair_result = calc.compute(&mut pair);
        // Two close atoms have less total SASA than 2x isolated
        assert!(pair_result.total_sasa < 2.0 * iso_result.total_sasa);
    }

    #[test]
    fn test_sasa_calculator_display() {
        let calc = SasaCalculator::new();
        assert!(calc.to_string().contains("probe=1.40"));
    }

    #[test]
    fn test_sasa_calculator_builders() {
        let calc = SasaCalculator::new().with_probe_radius(1.5).with_n_points(500);
        assert!(calc.to_string().contains("1.50"));
    }

    #[test]
    fn test_sasa_result_average() {
        let r = SasaResult { total_sasa: 100.0, atom_count: 10, probe_radius: 1.4 };
        assert!(approx(r.average_sasa(), 10.0, 1e-10));
    }

    #[test]
    fn test_sasa_result_display() {
        let r = SasaResult { total_sasa: 1234.5, atom_count: 50, probe_radius: 1.4 };
        assert!(r.to_string().contains("1234.5"));
    }

    #[test]
    fn test_surface_exposure_buried() {
        assert_eq!(SurfaceExposure::classify(0.0, 100.0), SurfaceExposure::Buried);
    }

    #[test]
    fn test_surface_exposure_partial() {
        assert_eq!(SurfaceExposure::classify(25.0, 100.0), SurfaceExposure::Partial);
    }

    #[test]
    fn test_surface_exposure_exposed() {
        assert_eq!(SurfaceExposure::classify(80.0, 100.0), SurfaceExposure::Exposed);
    }

    #[test]
    fn test_surface_exposure_display() {
        assert_eq!(SurfaceExposure::Buried.to_string(), "buried");
        assert_eq!(SurfaceExposure::Exposed.to_string(), "exposed");
    }

    #[test]
    fn test_max_sasa_reference() {
        assert!(approx(max_sasa_reference("ALA"), 129.0, 1e-1));
        assert!(approx(max_sasa_reference("TRP"), 285.0, 1e-1));
        assert!(approx(max_sasa_reference("XYZ"), 200.0, 1e-1));
    }

    #[test]
    fn test_golden_spiral_unit_sphere() {
        let pts = golden_spiral_points(100);
        assert_eq!(pts.len(), 100);
        for p in &pts {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(approx(r, 1.0, 0.01));
        }
    }
}
