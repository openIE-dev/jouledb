//! Peptide fragmentation, b/y ion series, and CID/HCD models.
//!
//! Computes theoretical fragment ions from a peptide sequence
//! including b-ion (N-terminal) and y-ion (C-terminal) series,
//! a-ions, neutral losses (water, ammonia), and internal fragments.
//! Supports CID and HCD fragmentation energy models.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────

/// Proton mass in Daltons.
pub const PROTON_MASS: f64 = 1.007_276_47;

/// Water loss (H₂O) in Daltons.
pub const WATER_LOSS: f64 = 18.010_565;

/// Ammonia loss (NH₃) in Daltons.
pub const AMMONIA_LOSS: f64 = 17.026_549;

/// CO loss (for a-ions from b-ions) in Daltons.
pub const CO_LOSS: f64 = 27.994_915;

// ── Standard amino acid residue masses ──────────────────────────

/// Monoisotopic residue masses for the 20 standard amino acids.
pub fn residue_mass(aa: char) -> Option<f64> {
    match aa {
        'G' => Some(57.021_464),
        'A' => Some(71.037_114),
        'V' => Some(99.068_414),
        'L' => Some(113.084_064),
        'I' => Some(113.084_064),
        'P' => Some(97.052_764),
        'F' => Some(147.068_414),
        'W' => Some(186.079_313),
        'M' => Some(131.040_485),
        'S' => Some(87.032_028),
        'T' => Some(101.047_679),
        'C' => Some(103.009_185),
        'Y' => Some(163.063_329),
        'H' => Some(137.058_912),
        'D' => Some(115.026_943),
        'E' => Some(129.042_593),
        'N' => Some(114.042_927),
        'Q' => Some(128.058_578),
        'K' => Some(128.094_963),
        'R' => Some(156.101_111),
        _ => None,
    }
}

/// Compute the total residue mass of a peptide sequence.
pub fn peptide_mass(sequence: &str) -> Option<f64> {
    let mut total = 0.0;
    for ch in sequence.chars() {
        total += residue_mass(ch)?;
    }
    // Add water for the intact peptide (H + OH terminus).
    Some(total + WATER_LOSS)
}

// ── FragmentType ────────────────────────────────────────────────

/// Type of a fragment ion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentType {
    B,
    Y,
    A,
    BMinusWater,
    BMinusAmmonia,
    YMinusWater,
    YMinusAmmonia,
}

impl fmt::Display for FragmentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::B => write!(f, "b"),
            Self::Y => write!(f, "y"),
            Self::A => write!(f, "a"),
            Self::BMinusWater => write!(f, "b-H₂O"),
            Self::BMinusAmmonia => write!(f, "b-NH₃"),
            Self::YMinusWater => write!(f, "y-H₂O"),
            Self::YMinusAmmonia => write!(f, "y-NH₃"),
        }
    }
}

// ── FragmentIon ─────────────────────────────────────────────────

/// A theoretical fragment ion with its m/z and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentIon {
    pub ion_type: FragmentType,
    pub index: usize,
    pub charge: u8,
    pub mz: f64,
    pub neutral_mass: f64,
}

impl FragmentIon {
    pub fn new(ion_type: FragmentType, index: usize, charge: u8, neutral_mass: f64) -> Self {
        let mz = (neutral_mass + charge as f64 * PROTON_MASS) / charge as f64;
        Self { ion_type, index, charge, mz, neutral_mass }
    }
}

impl fmt::Display for FragmentIon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.charge == 1 {
            write!(f, "{}{}  m/z={:.4}", self.ion_type, self.index, self.mz)
        } else {
            write!(
                f,
                "{}{}^{}+  m/z={:.4}",
                self.ion_type, self.index, self.charge, self.mz
            )
        }
    }
}

// ── FragmentationModel ──────────────────────────────────────────

/// Fragmentation energy model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FragmentationModel {
    /// Collision-induced dissociation — favours b/y ions.
    Cid,
    /// Higher-energy collisional dissociation — more complete fragmentation.
    Hcd,
    /// Electron transfer dissociation — favours c/z ions (simplified here).
    Etd,
}

impl FragmentationModel {
    /// Whether this model produces strong b-ion series.
    pub fn produces_b_ions(&self) -> bool {
        matches!(self, Self::Cid | Self::Hcd)
    }

    /// Whether neutral losses are prominent.
    pub fn strong_neutral_losses(&self) -> bool {
        matches!(self, Self::Cid)
    }
}

impl fmt::Display for FragmentationModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cid => write!(f, "CID"),
            Self::Hcd => write!(f, "HCD"),
            Self::Etd => write!(f, "ETD"),
        }
    }
}

// ── FragmentConfig ──────────────────────────────────────────────

/// Configuration for theoretical fragment generation.
#[derive(Debug, Clone)]
pub struct FragmentConfig {
    pub model: FragmentationModel,
    pub max_charge: u8,
    pub include_a_ions: bool,
    pub include_neutral_losses: bool,
    pub min_mz: f64,
    pub max_mz: f64,
}

impl FragmentConfig {
    pub fn new(model: FragmentationModel) -> Self {
        Self {
            model,
            max_charge: 1,
            include_a_ions: false,
            include_neutral_losses: true,
            min_mz: 100.0,
            max_mz: 2000.0,
        }
    }

    pub fn with_max_charge(mut self, c: u8) -> Self {
        self.max_charge = c.max(1);
        self
    }

    pub fn with_a_ions(mut self, yes: bool) -> Self {
        self.include_a_ions = yes;
        self
    }

    pub fn with_neutral_losses(mut self, yes: bool) -> Self {
        self.include_neutral_losses = yes;
        self
    }

    pub fn with_mz_range(mut self, lo: f64, hi: f64) -> Self {
        self.min_mz = lo;
        self.max_mz = hi;
        self
    }
}

// ── Fragment generation ─────────────────────────────────────────

/// Compute cumulative residue masses from N-terminus.
fn cumulative_masses(sequence: &str) -> Option<Vec<f64>> {
    let mut masses = Vec::with_capacity(sequence.len());
    let mut running = 0.0;
    for ch in sequence.chars() {
        running += residue_mass(ch)?;
        masses.push(running);
    }
    Some(masses)
}

/// Generate all theoretical fragment ions for a peptide.
pub fn generate_fragments(sequence: &str, config: &FragmentConfig) -> Option<Vec<FragmentIon>> {
    let n = sequence.len();
    if n < 2 {
        return Some(Vec::new());
    }
    let cum = cumulative_masses(sequence)?;
    let total = cum[n - 1] + WATER_LOSS; // intact peptide neutral mass.
    let mut ions = Vec::new();

    for charge in 1..=config.max_charge {
        for i in 1..n {
            // b-ion: sum of first i residues.
            let b_mass = cum[i - 1];
            let b = FragmentIon::new(FragmentType::B, i, charge, b_mass);
            if b.mz >= config.min_mz && b.mz <= config.max_mz {
                ions.push(b);
            }

            // y-ion: intact - b + H₂O.
            let y_mass = total - cum[i - 1];
            let y = FragmentIon::new(FragmentType::Y, n - i, charge, y_mass);
            if y.mz >= config.min_mz && y.mz <= config.max_mz {
                ions.push(y);
            }

            // a-ion: b - CO.
            if config.include_a_ions {
                let a_mass = b_mass - CO_LOSS;
                if a_mass > 0.0 {
                    let a = FragmentIon::new(FragmentType::A, i, charge, a_mass);
                    if a.mz >= config.min_mz && a.mz <= config.max_mz {
                        ions.push(a);
                    }
                }
            }

            // Neutral losses.
            if config.include_neutral_losses && config.model.strong_neutral_losses() {
                let bw = FragmentIon::new(FragmentType::BMinusWater, i, charge, b_mass - WATER_LOSS);
                if bw.mz >= config.min_mz && bw.mz <= config.max_mz && bw.neutral_mass > 0.0 {
                    ions.push(bw);
                }
                let ba = FragmentIon::new(FragmentType::BMinusAmmonia, i, charge, b_mass - AMMONIA_LOSS);
                if ba.mz >= config.min_mz && ba.mz <= config.max_mz && ba.neutral_mass > 0.0 {
                    ions.push(ba);
                }
                let yw = FragmentIon::new(FragmentType::YMinusWater, n - i, charge, y_mass - WATER_LOSS);
                if yw.mz >= config.min_mz && yw.mz <= config.max_mz && yw.neutral_mass > 0.0 {
                    ions.push(yw);
                }
                let ya = FragmentIon::new(FragmentType::YMinusAmmonia, n - i, charge, y_mass - AMMONIA_LOSS);
                if ya.mz >= config.min_mz && ya.mz <= config.max_mz && ya.neutral_mass > 0.0 {
                    ions.push(ya);
                }
            }
        }
    }

    ions.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
    Some(ions)
}

/// Count the number of each ion type in a fragment list.
pub fn count_ion_types(ions: &[FragmentIon]) -> Vec<(FragmentType, usize)> {
    let types = [
        FragmentType::B,
        FragmentType::Y,
        FragmentType::A,
        FragmentType::BMinusWater,
        FragmentType::BMinusAmmonia,
        FragmentType::YMinusWater,
        FragmentType::YMinusAmmonia,
    ];
    types
        .iter()
        .filter_map(|t| {
            let c = ions.iter().filter(|i| i.ion_type == *t).count();
            if c > 0 { Some((*t, c)) } else { None }
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_mass_glycine() {
        assert!((residue_mass('G').unwrap() - 57.021_464).abs() < 1e-4);
    }

    #[test]
    fn test_residue_mass_unknown() {
        assert!(residue_mass('X').is_none());
    }

    #[test]
    fn test_peptide_mass_gly_ala() {
        // GA = G + A + H₂O
        let m = peptide_mass("GA").unwrap();
        let expected = 57.021_464 + 71.037_114 + WATER_LOSS;
        assert!((m - expected).abs() < 0.001);
    }

    #[test]
    fn test_peptide_mass_invalid() {
        assert!(peptide_mass("GXA").is_none());
    }

    #[test]
    fn test_generate_b_y_ions() {
        let cfg = FragmentConfig::new(FragmentationModel::Hcd)
            .with_neutral_losses(false)
            .with_mz_range(0.0, 5000.0);
        let ions = generate_fragments("PEPTIDE", &cfg).unwrap();
        let b_count = ions.iter().filter(|i| i.ion_type == FragmentType::B).count();
        let y_count = ions.iter().filter(|i| i.ion_type == FragmentType::Y).count();
        // 7-residue peptide ⟹ 6 b-ions and 6 y-ions.
        assert_eq!(b_count, 6);
        assert_eq!(y_count, 6);
    }

    #[test]
    fn test_a_ions() {
        let cfg = FragmentConfig::new(FragmentationModel::Hcd)
            .with_a_ions(true)
            .with_neutral_losses(false)
            .with_mz_range(0.0, 5000.0);
        let ions = generate_fragments("PEPTIDE", &cfg).unwrap();
        let a_count = ions.iter().filter(|i| i.ion_type == FragmentType::A).count();
        assert!(a_count > 0);
    }

    #[test]
    fn test_neutral_losses_cid() {
        let cfg = FragmentConfig::new(FragmentationModel::Cid)
            .with_mz_range(0.0, 5000.0);
        let ions = generate_fragments("PEPTIDE", &cfg).unwrap();
        assert!(ions.iter().any(|i| i.ion_type == FragmentType::BMinusWater));
    }

    #[test]
    fn test_no_neutral_losses_hcd() {
        let cfg = FragmentConfig::new(FragmentationModel::Hcd)
            .with_mz_range(0.0, 5000.0);
        let ions = generate_fragments("PEPTIDE", &cfg).unwrap();
        // HCD does not have strong neutral losses in our model.
        assert!(!ions.iter().any(|i| i.ion_type == FragmentType::BMinusWater));
    }

    #[test]
    fn test_multiply_charged_fragments() {
        let cfg = FragmentConfig::new(FragmentationModel::Hcd)
            .with_max_charge(2)
            .with_neutral_losses(false)
            .with_mz_range(0.0, 5000.0);
        let ions = generate_fragments("PEPTIDE", &cfg).unwrap();
        assert!(ions.iter().any(|i| i.charge == 2));
    }

    #[test]
    fn test_fragment_ion_display_singly() {
        let ion = FragmentIon::new(FragmentType::B, 3, 1, 300.0);
        let s = format!("{}", ion);
        assert!(s.starts_with("b3"));
        assert!(!s.contains('^'));
    }

    #[test]
    fn test_fragment_ion_display_doubly() {
        let ion = FragmentIon::new(FragmentType::Y, 5, 2, 600.0);
        let s = format!("{}", ion);
        assert!(s.contains("y5^2+"));
    }

    #[test]
    fn test_b1_mz_correctness() {
        let cfg = FragmentConfig::new(FragmentationModel::Hcd)
            .with_neutral_losses(false)
            .with_mz_range(0.0, 5000.0);
        let ions = generate_fragments("AG", &cfg).unwrap();
        let b1 = ions.iter().find(|i| i.ion_type == FragmentType::B && i.index == 1).unwrap();
        // b1 = A residue mass + proton
        let expected = 71.037_114 + PROTON_MASS;
        assert!((b1.mz - expected).abs() < 0.001);
    }

    #[test]
    fn test_empty_sequence() {
        let cfg = FragmentConfig::new(FragmentationModel::Hcd);
        let ions = generate_fragments("A", &cfg).unwrap();
        assert!(ions.is_empty());
    }

    #[test]
    fn test_sorted_output() {
        let cfg = FragmentConfig::new(FragmentationModel::Hcd)
            .with_neutral_losses(false)
            .with_mz_range(0.0, 5000.0);
        let ions = generate_fragments("PEPTIDE", &cfg).unwrap();
        for w in ions.windows(2) {
            assert!(w[0].mz <= w[1].mz);
        }
    }

    #[test]
    fn test_count_ion_types() {
        let cfg = FragmentConfig::new(FragmentationModel::Cid)
            .with_a_ions(true)
            .with_mz_range(0.0, 5000.0);
        let ions = generate_fragments("PEPTIDE", &cfg).unwrap();
        let counts = count_ion_types(&ions);
        assert!(counts.iter().any(|(t, _)| *t == FragmentType::B));
        assert!(counts.iter().any(|(t, _)| *t == FragmentType::Y));
    }

    #[test]
    fn test_fragmentation_model_display() {
        assert_eq!(format!("{}", FragmentationModel::Cid), "CID");
        assert_eq!(format!("{}", FragmentationModel::Hcd), "HCD");
        assert_eq!(format!("{}", FragmentationModel::Etd), "ETD");
    }

    #[test]
    fn test_proton_mass_sanity() {
        assert!(PROTON_MASS > 1.0 && PROTON_MASS < 1.01);
    }

    #[test]
    fn test_mz_range_filter() {
        let cfg = FragmentConfig::new(FragmentationModel::Hcd)
            .with_neutral_losses(false)
            .with_mz_range(200.0, 500.0);
        let ions = generate_fragments("PEPTIDE", &cfg).unwrap();
        assert!(ions.iter().all(|i| i.mz >= 200.0 && i.mz <= 500.0));
    }

    #[test]
    fn test_fragment_type_display() {
        assert_eq!(format!("{}", FragmentType::BMinusWater), "b-H₂O");
        assert_eq!(format!("{}", FragmentType::YMinusAmmonia), "y-NH₃");
    }
}
