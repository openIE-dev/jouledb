//! Post-translational modification identification, mass shifts, and
//! localisation scoring.
//!
//! Models common PTMs (phosphorylation, oxidation, acetylation, etc.),
//! computes mass shifts for modified peptides, implements Ascore-style
//! localisation scoring, and supports variable modification search
//! space enumeration.

use std::fmt;

// ── ModificationType ────────────────────────────────────────────

/// Common post-translational modification types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModificationType {
    Phosphorylation,
    Oxidation,
    Acetylation,
    Methylation,
    Dimethylation,
    Trimethylation,
    Ubiquitination,
    Carbamidomethyl,
    Deamidation,
    Pyro,
    Sulfo,
    Nitrosylation,
    Custom,
}

impl ModificationType {
    /// Monoisotopic mass shift in Daltons.
    pub fn mass_shift(&self) -> f64 {
        match self {
            Self::Phosphorylation => 79.966_331,
            Self::Oxidation => 15.994_915,
            Self::Acetylation => 42.010_565,
            Self::Methylation => 14.015_650,
            Self::Dimethylation => 28.031_300,
            Self::Trimethylation => 42.046_950,
            Self::Ubiquitination => 114.042_927, // diGly remnant
            Self::Carbamidomethyl => 57.021_464,
            Self::Deamidation => 0.984_016,
            Self::Pyro => -17.026_549,
            Self::Sulfo => 79.956_815,
            Self::Nitrosylation => 28.990_164,
            Self::Custom => 0.0,
        }
    }

    /// Residues this modification typically targets.
    pub fn target_residues(&self) -> &'static [char] {
        match self {
            Self::Phosphorylation => &['S', 'T', 'Y'],
            Self::Oxidation => &['M', 'W'],
            Self::Acetylation => &['K', 'N'],
            Self::Methylation => &['K', 'R'],
            Self::Dimethylation => &['K', 'R'],
            Self::Trimethylation => &['K'],
            Self::Ubiquitination => &['K'],
            Self::Carbamidomethyl => &['C'],
            Self::Deamidation => &['N', 'Q'],
            Self::Pyro => &['Q', 'E'],
            Self::Sulfo => &['Y'],
            Self::Nitrosylation => &['C'],
            Self::Custom => &[],
        }
    }

    /// Whether this modification is typically fixed (always present)
    /// or variable (search-space parameter).
    pub fn is_typically_fixed(&self) -> bool {
        matches!(self, Self::Carbamidomethyl)
    }
}

impl fmt::Display for ModificationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Phosphorylation => write!(f, "Phospho"),
            Self::Oxidation => write!(f, "Oxidation"),
            Self::Acetylation => write!(f, "Acetyl"),
            Self::Methylation => write!(f, "Methyl"),
            Self::Dimethylation => write!(f, "Dimethyl"),
            Self::Trimethylation => write!(f, "Trimethyl"),
            Self::Ubiquitination => write!(f, "GlyGly"),
            Self::Carbamidomethyl => write!(f, "Carbamidomethyl"),
            Self::Deamidation => write!(f, "Deamidated"),
            Self::Pyro => write!(f, "Pyro-glu"),
            Self::Sulfo => write!(f, "Sulfo"),
            Self::Nitrosylation => write!(f, "Nitrosyl"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

// ── SiteModification ────────────────────────────────────────────

/// A modification placed at a specific position on a peptide.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteModification {
    pub mod_type: ModificationType,
    pub position: usize,
    pub residue: char,
    pub mass_shift: f64,
}

impl SiteModification {
    pub fn new(mod_type: ModificationType, position: usize, residue: char) -> Self {
        Self {
            mod_type,
            position,
            residue,
            mass_shift: mod_type.mass_shift(),
        }
    }

    pub fn with_custom_mass(mut self, mass: f64) -> Self {
        self.mass_shift = mass;
        self
    }
}

impl fmt::Display for SiteModification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}{}): {:.4} Da",
            self.mod_type, self.residue, self.position + 1, self.mass_shift
        )
    }
}

// ── ModifiedPeptide ─────────────────────────────────────────────

/// A peptide sequence with zero or more modifications.
#[derive(Debug, Clone)]
pub struct ModifiedPeptide {
    pub sequence: String,
    pub modifications: Vec<SiteModification>,
    pub n_term_mod: Option<ModificationType>,
    pub c_term_mod: Option<ModificationType>,
}

impl ModifiedPeptide {
    pub fn new(sequence: &str) -> Self {
        Self {
            sequence: sequence.to_string(),
            modifications: Vec::new(),
            n_term_mod: None,
            c_term_mod: None,
        }
    }

    pub fn with_modification(mut self, site_mod: SiteModification) -> Self {
        self.modifications.push(site_mod);
        self
    }

    pub fn with_n_term(mut self, mod_type: ModificationType) -> Self {
        self.n_term_mod = Some(mod_type);
        self
    }

    pub fn with_c_term(mut self, mod_type: ModificationType) -> Self {
        self.c_term_mod = Some(mod_type);
        self
    }

    /// Total mass shift from all modifications.
    pub fn total_mass_shift(&self) -> f64 {
        let site_shift: f64 = self.modifications.iter().map(|m| m.mass_shift).sum();
        let nterm = self.n_term_mod.map(|m| m.mass_shift()).unwrap_or(0.0);
        let cterm = self.c_term_mod.map(|m| m.mass_shift()).unwrap_or(0.0);
        site_shift + nterm + cterm
    }

    /// Number of modifications.
    pub fn mod_count(&self) -> usize {
        let extra = self.n_term_mod.is_some() as usize + self.c_term_mod.is_some() as usize;
        self.modifications.len() + extra
    }

    /// Whether a specific position is modified.
    pub fn is_modified_at(&self, position: usize) -> bool {
        self.modifications.iter().any(|m| m.position == position)
    }

    /// Get all positions of a given modification type.
    pub fn positions_of(&self, mod_type: ModificationType) -> Vec<usize> {
        self.modifications
            .iter()
            .filter(|m| m.mod_type == mod_type)
            .map(|m| m.position)
            .collect()
    }
}

impl fmt::Display for ModifiedPeptide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let chars: Vec<char> = self.sequence.chars().collect();
        let mut parts = Vec::new();
        if let Some(nt) = &self.n_term_mod {
            parts.push(format!("[{}]-", nt));
        }
        for (i, &ch) in chars.iter().enumerate() {
            parts.push(ch.to_string());
            for m in &self.modifications {
                if m.position == i {
                    parts.push(format!("[{}]", m.mod_type));
                }
            }
        }
        if let Some(ct) = &self.c_term_mod {
            parts.push(format!("-[{}]", ct));
        }
        write!(f, "{}", parts.concat())
    }
}

// ── Modification search space ───────────────────────────────────

/// Configuration for variable modification search.
#[derive(Debug, Clone)]
pub struct ModSearchConfig {
    pub variable_mods: Vec<ModificationType>,
    pub fixed_mods: Vec<ModificationType>,
    pub max_variable_mods: usize,
}

impl ModSearchConfig {
    pub fn new() -> Self {
        Self {
            variable_mods: Vec::new(),
            fixed_mods: vec![ModificationType::Carbamidomethyl],
            max_variable_mods: 3,
        }
    }

    pub fn with_variable_mod(mut self, m: ModificationType) -> Self {
        self.variable_mods.push(m);
        self
    }

    pub fn with_fixed_mod(mut self, m: ModificationType) -> Self {
        self.fixed_mods.push(m);
        self
    }

    pub fn with_max_variable(mut self, n: usize) -> Self {
        self.max_variable_mods = n;
        self
    }
}

impl Default for ModSearchConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Find all positions in a peptide where a modification can occur.
pub fn candidate_sites(sequence: &str, mod_type: ModificationType) -> Vec<usize> {
    let targets = mod_type.target_residues();
    sequence
        .chars()
        .enumerate()
        .filter(|(_, ch)| targets.contains(ch))
        .map(|(i, _)| i)
        .collect()
}

/// Enumerate all modified peptide forms given variable mods.
///
/// Returns a list of modification combinations up to `max_mods`.
pub fn enumerate_modifications(
    sequence: &str,
    config: &ModSearchConfig,
) -> Vec<ModifiedPeptide> {
    // Apply fixed modifications first.
    let mut base_mods = Vec::new();
    for fixed in &config.fixed_mods {
        let sites = candidate_sites(sequence, *fixed);
        let chars: Vec<char> = sequence.chars().collect();
        for pos in sites {
            base_mods.push(SiteModification::new(*fixed, pos, chars[pos]));
        }
    }

    // Find candidate variable sites.
    let mut var_sites: Vec<(ModificationType, usize, char)> = Vec::new();
    let chars: Vec<char> = sequence.chars().collect();
    for var_mod in &config.variable_mods {
        let sites = candidate_sites(sequence, *var_mod);
        for pos in sites {
            // Don't place variable mod on a fixed-mod site.
            if !base_mods.iter().any(|m| m.position == pos) {
                var_sites.push((*var_mod, pos, chars[pos]));
            }
        }
    }

    let mut results = Vec::new();

    // Unmodified form (with only fixed mods).
    let base = ModifiedPeptide {
        sequence: sequence.to_string(),
        modifications: base_mods.clone(),
        n_term_mod: None,
        c_term_mod: None,
    };
    results.push(base);

    // Generate combinations of variable mods up to max.
    let max = config.max_variable_mods.min(var_sites.len());
    for size in 1..=max {
        for combo in combinations(&var_sites, size) {
            let mut mods = base_mods.clone();
            for &(mt, pos, res) in &combo {
                mods.push(SiteModification::new(mt, pos, res));
            }
            results.push(ModifiedPeptide {
                sequence: sequence.to_string(),
                modifications: mods,
                n_term_mod: None,
                c_term_mod: None,
            });
        }
    }

    results
}

/// Simple k-combinations of a slice.
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if items.len() < k {
        return Vec::new();
    }
    let mut result = Vec::new();
    for i in 0..=items.len() - k {
        let sub = combinations(&items[i + 1..], k - 1);
        for mut s in sub {
            s.insert(0, items[i].clone());
            result.push(s);
        }
    }
    result
}

// ── Localisation scoring ────────────────────────────────────────

/// A localisation score for a modification at a candidate position.
#[derive(Debug, Clone)]
pub struct LocalisationScore {
    pub position: usize,
    pub residue: char,
    pub score: f64,
    pub probability: f64,
}

impl LocalisationScore {
    pub fn new(position: usize, residue: char, score: f64) -> Self {
        Self { position, residue, score, probability: 0.0 }
    }

    pub fn with_probability(mut self, p: f64) -> Self {
        self.probability = p;
        self
    }
}

impl fmt::Display for LocalisationScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{} score={:.2} p={:.3}",
            self.residue,
            self.position + 1,
            self.score,
            self.probability,
        )
    }
}

/// Simplified Ascore-style localisation: given fragment ion scores
/// for each candidate site, compute the probability that each site
/// is the true modification site.
///
/// `site_scores` is a slice of (position, residue, raw_score).
pub fn compute_localisation(
    site_scores: &[(usize, char, f64)],
) -> Vec<LocalisationScore> {
    if site_scores.is_empty() {
        return Vec::new();
    }

    let max_score = site_scores
        .iter()
        .map(|s| s.2)
        .fold(f64::NEG_INFINITY, f64::max);

    // Softmax over scores for probabilities.
    let exps: Vec<f64> = site_scores
        .iter()
        .map(|s| (s.2 - max_score).exp())
        .collect();
    let total: f64 = exps.iter().sum();

    site_scores
        .iter()
        .zip(exps.iter())
        .map(|((pos, res, score), &ex)| {
            let prob = if total > 0.0 { ex / total } else { 0.0 };
            LocalisationScore::new(*pos, *res, *score).with_probability(prob)
        })
        .collect()
}

/// Determine the best localisation site.
pub fn best_site(scores: &[LocalisationScore]) -> Option<&LocalisationScore> {
    scores
        .iter()
        .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap())
}

/// Delta score between best and second-best sites.
pub fn localisation_delta(scores: &[LocalisationScore]) -> f64 {
    if scores.len() < 2 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = scores.iter().map(|s| s.score).collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    sorted[0] - sorted[1]
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phospho_mass() {
        let m = ModificationType::Phosphorylation.mass_shift();
        assert!((m - 79.966).abs() < 0.001);
    }

    #[test]
    fn test_oxidation_mass() {
        let m = ModificationType::Oxidation.mass_shift();
        assert!((m - 15.995).abs() < 0.001);
    }

    #[test]
    fn test_target_residues_phospho() {
        let targets = ModificationType::Phosphorylation.target_residues();
        assert!(targets.contains(&'S'));
        assert!(targets.contains(&'T'));
        assert!(targets.contains(&'Y'));
        assert!(!targets.contains(&'A'));
    }

    #[test]
    fn test_is_typically_fixed() {
        assert!(ModificationType::Carbamidomethyl.is_typically_fixed());
        assert!(!ModificationType::Oxidation.is_typically_fixed());
    }

    #[test]
    fn test_site_modification_display() {
        let sm = SiteModification::new(ModificationType::Phosphorylation, 2, 'S');
        let d = format!("{}", sm);
        assert!(d.contains("Phospho"));
        assert!(d.contains("S3")); // 1-indexed
    }

    #[test]
    fn test_modified_peptide_total_shift() {
        let pep = ModifiedPeptide::new("PEPTIDE")
            .with_modification(SiteModification::new(
                ModificationType::Phosphorylation, 0, 'P',
            ))
            .with_modification(SiteModification::new(
                ModificationType::Oxidation, 3, 'T',
            ));
        let shift = pep.total_mass_shift();
        let expected = 79.966_331 + 15.994_915;
        assert!((shift - expected).abs() < 0.001);
    }

    #[test]
    fn test_modified_peptide_display() {
        let pep = ModifiedPeptide::new("MST")
            .with_modification(SiteModification::new(
                ModificationType::Oxidation, 0, 'M',
            ));
        let d = format!("{}", pep);
        assert!(d.contains("M[Oxidation]"));
    }

    #[test]
    fn test_nterm_mod_display() {
        let pep = ModifiedPeptide::new("PEPTIDE")
            .with_n_term(ModificationType::Acetylation);
        let d = format!("{}", pep);
        assert!(d.contains("[Acetyl]-"));
    }

    #[test]
    fn test_mod_count() {
        let pep = ModifiedPeptide::new("PEPTIDE")
            .with_modification(SiteModification::new(
                ModificationType::Phosphorylation, 0, 'P',
            ))
            .with_n_term(ModificationType::Acetylation);
        assert_eq!(pep.mod_count(), 2);
    }

    #[test]
    fn test_is_modified_at() {
        let pep = ModifiedPeptide::new("ABC")
            .with_modification(SiteModification::new(
                ModificationType::Oxidation, 1, 'B',
            ));
        assert!(pep.is_modified_at(1));
        assert!(!pep.is_modified_at(0));
    }

    #[test]
    fn test_positions_of() {
        let pep = ModifiedPeptide::new("STSYS")
            .with_modification(SiteModification::new(ModificationType::Phosphorylation, 0, 'S'))
            .with_modification(SiteModification::new(ModificationType::Phosphorylation, 2, 'S'));
        let pos = pep.positions_of(ModificationType::Phosphorylation);
        assert_eq!(pos, vec![0, 2]);
    }

    #[test]
    fn test_candidate_sites() {
        let sites = candidate_sites("SAMSTER", ModificationType::Phosphorylation);
        // S at 0, T at 4.
        assert!(sites.contains(&0));
        assert!(sites.contains(&4));
    }

    #[test]
    fn test_enumerate_unmodified() {
        let cfg = ModSearchConfig::new();
        let forms = enumerate_modifications("AGK", &cfg);
        // At least the base (fixed-mods only) form.
        assert!(!forms.is_empty());
    }

    #[test]
    fn test_enumerate_variable() {
        let cfg = ModSearchConfig::new()
            .with_variable_mod(ModificationType::Oxidation)
            .with_max_variable(2);
        let forms = enumerate_modifications("MWM", &cfg);
        // M at pos 0,2 and W at pos 1 → 3 sites; combos of size 0,1,2.
        assert!(forms.len() > 1);
    }

    #[test]
    fn test_enumerate_respects_max() {
        let cfg = ModSearchConfig::new()
            .with_variable_mod(ModificationType::Phosphorylation)
            .with_max_variable(1);
        let forms = enumerate_modifications("STSST", &cfg);
        // 1 unmodified + C(n,1) single-mod forms.
        for form in &forms {
            let var_count = form.modifications.iter()
                .filter(|m| m.mod_type == ModificationType::Phosphorylation)
                .count();
            assert!(var_count <= 1);
        }
    }

    #[test]
    fn test_localisation_scoring() {
        let scores = compute_localisation(&[
            (0, 'S', 10.0),
            (2, 'T', 5.0),
            (4, 'S', 3.0),
        ]);
        assert_eq!(scores.len(), 3);
        let total_p: f64 = scores.iter().map(|s| s.probability).sum();
        assert!((total_p - 1.0).abs() < 1e-6);
        // Position 0 has the highest score ⟹ highest probability.
        assert!(scores[0].probability > scores[1].probability);
    }

    #[test]
    fn test_best_site() {
        let scores = compute_localisation(&[
            (0, 'S', 10.0),
            (2, 'T', 15.0),
        ]);
        let best = best_site(&scores).unwrap();
        assert_eq!(best.position, 2);
    }

    #[test]
    fn test_localisation_delta() {
        let scores = compute_localisation(&[
            (0, 'S', 10.0),
            (2, 'T', 5.0),
        ]);
        let delta = localisation_delta(&scores);
        assert!((delta - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mod_type_display() {
        assert_eq!(format!("{}", ModificationType::Phosphorylation), "Phospho");
        assert_eq!(format!("{}", ModificationType::Ubiquitination), "GlyGly");
    }

    #[test]
    fn test_custom_mass_shift() {
        let sm = SiteModification::new(ModificationType::Custom, 0, 'X')
            .with_custom_mass(42.0);
        assert!((sm.mass_shift - 42.0).abs() < 1e-9);
    }
}
