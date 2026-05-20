//! Splice site prediction with donor/acceptor signal scoring and
//! intron/exon boundary detection.
//!
//! Implements GT-AG rule scanning, branch point detection, polypyrimidine
//! tract scoring, position weight matrix models for donor (5') and
//! acceptor (3') splice sites, exon/intron boundary delineation,
//! and splice site strength classification.

use std::fmt;

// ── Splice Site Type ────────────────────────────────────────────

/// Type of splice site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpliceSiteType {
    /// 5' splice site (exon/intron boundary, typically GT).
    Donor,
    /// 3' splice site (intron/exon boundary, typically AG).
    Acceptor,
    /// Non-canonical splice site.
    NonCanonical,
}

impl fmt::Display for SpliceSiteType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Donor => write!(f, "donor(5')"),
            Self::Acceptor => write!(f, "acceptor(3')"),
            Self::NonCanonical => write!(f, "non-canonical"),
        }
    }
}

// ── Splice Site Strength ────────────────────────────────────────

/// Qualitative classification of splice site strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpliceStrength {
    Strong,
    Moderate,
    Weak,
}

impl fmt::Display for SpliceStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Strong => write!(f, "strong"),
            Self::Moderate => write!(f, "moderate"),
            Self::Weak => write!(f, "weak"),
        }
    }
}

/// Classify splice site strength from a normalized score.
pub fn classify_strength(score: f64) -> SpliceStrength {
    if score >= 0.8 {
        SpliceStrength::Strong
    } else if score >= 0.5 {
        SpliceStrength::Moderate
    } else {
        SpliceStrength::Weak
    }
}

// ── Splice Site Hit ─────────────────────────────────────────────

/// A detected splice site with position and score.
#[derive(Debug, Clone)]
pub struct SpliceSite {
    pub site_type: SpliceSiteType,
    pub position: usize,
    pub score: f64,
    pub strength: SpliceStrength,
    pub dinucleotide: String,
    pub context: String,
}

impl SpliceSite {
    pub fn new(site_type: SpliceSiteType, position: usize) -> Self {
        Self {
            site_type,
            position,
            score: 0.0,
            strength: SpliceStrength::Weak,
            dinucleotide: String::new(),
            context: String::new(),
        }
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self.strength = classify_strength(score);
        self
    }

    pub fn with_dinucleotide(mut self, dn: &str) -> Self {
        self.dinucleotide = dn.to_string();
        self
    }

    pub fn with_context(mut self, ctx: &str) -> Self {
        self.context = ctx.to_string();
        self
    }

    /// Whether this is a canonical GT or AG site.
    pub fn is_canonical(&self) -> bool {
        let dn = self.dinucleotide.to_uppercase();
        match self.site_type {
            SpliceSiteType::Donor => dn == "GT",
            SpliceSiteType::Acceptor => dn == "AG",
            SpliceSiteType::NonCanonical => false,
        }
    }
}

impl fmt::Display for SpliceSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpliceSite({}, pos={}, {}, score={:.3}, {})",
            self.site_type, self.position, self.dinucleotide,
            self.score, self.strength,
        )
    }
}

// ── Splice Site PWM ─────────────────────────────────────────────

/// Position weight matrix for splice site scoring.
#[derive(Debug, Clone)]
pub struct SplicePwm {
    pub site_type: SpliceSiteType,
    pub matrix: Vec<[f64; 4]>,
}

impl SplicePwm {
    pub fn new(site_type: SpliceSiteType) -> Self {
        Self {
            site_type,
            matrix: Vec::new(),
        }
    }

    pub fn with_position(mut self, a: f64, t: f64, g: f64, c: f64) -> Self {
        self.matrix.push([a, t, g, c]);
        self
    }

    /// Build a default donor (5') splice site PWM.
    ///
    /// Consensus: MAG|GTRAGT (M=A/C, R=A/G, |=exon/intron boundary).
    pub fn default_donor() -> Self {
        Self::new(SpliceSiteType::Donor)
            .with_position(1.0, -1.0, -0.5, 0.5)   // M (A or C)
            .with_position(1.5, -1.0, -0.5, -0.5)   // A
            .with_position(-1.0, -1.0, 2.0, -1.0)   // G
            // intron boundary
            .with_position(-2.0, -2.0, 2.5, -2.0)   // G (canonical)
            .with_position(-2.0, 2.5, -2.0, -2.0)   // T (canonical)
            .with_position(1.0, -0.5, 0.5, -1.0)    // R (A or G)
            .with_position(1.5, -0.5, -0.5, -0.5)   // A
            .with_position(-1.0, -1.0, 2.0, -1.0)   // G
            .with_position(-1.0, 1.5, -0.5, -0.5)   // T
    }

    /// Build a default acceptor (3') splice site PWM.
    ///
    /// Consensus: polypyrimidine tract + YAG| (Y=C/T).
    pub fn default_acceptor() -> Self {
        Self::new(SpliceSiteType::Acceptor)
            .with_position(-0.5, 1.0, -1.0, 1.0)    // Y (pyrimidine)
            .with_position(-0.5, 1.0, -1.0, 1.0)    // Y
            .with_position(-0.5, 1.0, -1.0, 1.0)    // Y
            .with_position(-0.5, 1.0, -1.0, 1.0)    // Y (polypyrimidine tract)
            .with_position(-1.0, 1.0, -1.0, 1.0)    // Y
            .with_position(-2.0, -2.0, -2.0, -2.0)   // N (branch point region)
            .with_position(-0.5, 1.0, -1.0, 1.0)    // Y
            // splice site
            .with_position(-2.0, -2.0, -2.0, -2.0)   // N
            .with_position(2.5, -2.0, -2.0, -2.0)   // A (canonical)
            .with_position(-2.0, -2.0, 2.5, -2.0)   // G (canonical)
    }

    fn base_index(b: u8) -> Option<usize> {
        match b {
            b'A' | b'a' => Some(0),
            b'T' | b't' => Some(1),
            b'G' | b'g' => Some(2),
            b'C' | b'c' => Some(3),
            _ => None,
        }
    }

    /// Score a subsequence.
    pub fn score(&self, seq: &[u8]) -> f64 {
        if seq.len() < self.matrix.len() {
            return f64::NEG_INFINITY;
        }
        let mut total = 0.0;
        for (i, weights) in self.matrix.iter().enumerate() {
            if let Some(idx) = Self::base_index(seq[i]) {
                total += weights[idx];
            } else {
                total += -5.0;
            }
        }
        total
    }

    /// Max and min possible scores.
    pub fn score_range(&self) -> (f64, f64) {
        let max_s: f64 = self.matrix.iter()
            .map(|r| r.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .sum();
        let min_s: f64 = self.matrix.iter()
            .map(|r| r.iter().copied().fold(f64::INFINITY, f64::min))
            .sum();
        (min_s, max_s)
    }

    /// Normalized score (0.0 to 1.0).
    pub fn normalized_score(&self, seq: &[u8]) -> f64 {
        let raw = self.score(seq);
        let (mn, mx) = self.score_range();
        if (mx - mn).abs() < 1e-12 { 0.0 } else { (raw - mn) / (mx - mn) }
    }
}

impl fmt::Display for SplicePwm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SplicePwm({}, len={})", self.site_type, self.matrix.len())
    }
}

// ── Intron/Exon Boundary ────────────────────────────────────────

/// An intron/exon structure inferred from splice sites.
#[derive(Debug, Clone)]
pub struct IntronExon {
    pub exon_start: usize,
    pub exon_end: usize,
    pub intron_start: usize,
    pub intron_end: usize,
    pub donor_score: f64,
    pub acceptor_score: f64,
}

impl IntronExon {
    pub fn new(exon_start: usize, exon_end: usize, intron_start: usize, intron_end: usize) -> Self {
        Self {
            exon_start,
            exon_end,
            intron_start,
            intron_end,
            donor_score: 0.0,
            acceptor_score: 0.0,
        }
    }

    pub fn with_donor_score(mut self, score: f64) -> Self {
        self.donor_score = score;
        self
    }

    pub fn with_acceptor_score(mut self, score: f64) -> Self {
        self.acceptor_score = score;
        self
    }

    pub fn exon_length(&self) -> usize {
        if self.exon_end >= self.exon_start { self.exon_end - self.exon_start + 1 } else { 0 }
    }

    pub fn intron_length(&self) -> usize {
        if self.intron_end >= self.intron_start { self.intron_end - self.intron_start + 1 } else { 0 }
    }
}

impl fmt::Display for IntronExon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IntronExon(exon={}-{} [{}bp], intron={}-{} [{}bp])",
            self.exon_start, self.exon_end, self.exon_length(),
            self.intron_start, self.intron_end, self.intron_length(),
        )
    }
}

// ── Polypyrimidine Tract ────────────────────────────────────────

/// Score a polypyrimidine tract upstream of an acceptor site.
pub fn polypyrimidine_score(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let pyrimidine_count = seq.iter()
        .filter(|&&b| b == b'T' || b == b't' || b == b'C' || b == b'c')
        .count();
    pyrimidine_count as f64 / seq.len() as f64
}

// ── Splice Site Scanning ────────────────────────────────────────

/// Configuration for splice site scanning.
#[derive(Debug, Clone)]
pub struct SpliceScanConfig {
    pub min_score: f64,
    pub scan_donors: bool,
    pub scan_acceptors: bool,
    pub context_size: usize,
}

impl SpliceScanConfig {
    pub fn new() -> Self {
        Self {
            min_score: 0.4,
            scan_donors: true,
            scan_acceptors: true,
            context_size: 20,
        }
    }

    pub fn with_min_score(mut self, score: f64) -> Self {
        self.min_score = score;
        self
    }

    pub fn with_scan_donors(mut self, scan: bool) -> Self {
        self.scan_donors = scan;
        self
    }

    pub fn with_scan_acceptors(mut self, scan: bool) -> Self {
        self.scan_acceptors = scan;
        self
    }

    pub fn with_context_size(mut self, size: usize) -> Self {
        self.context_size = size;
        self
    }
}

impl fmt::Display for SpliceScanConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpliceScanConfig(min_score={:.2}, donors={}, acceptors={})",
            self.min_score, self.scan_donors, self.scan_acceptors,
        )
    }
}

/// Scan for GT donor sites using a PWM.
pub fn scan_donor_sites(sequence: &str, pwm: &SplicePwm, min_score: f64) -> Vec<SpliceSite> {
    let seq = sequence.to_uppercase();
    let bytes = seq.as_bytes();
    let mut sites = Vec::new();
    let pwm_len = pwm.matrix.len();

    if bytes.len() < pwm_len {
        return sites;
    }

    for i in 0..=(bytes.len() - pwm_len) {
        let norm = pwm.normalized_score(&bytes[i..]);
        if norm >= min_score {
            // Extract the GT dinucleotide (at offset 3 in default donor PWM)
            let dn_start = i + 3;
            let dn = if dn_start + 2 <= bytes.len() {
                seq[dn_start..dn_start + 2].to_string()
            } else {
                String::new()
            };
            let ctx_end = (i + pwm_len).min(bytes.len());
            let ctx = seq[i..ctx_end].to_string();
            let site = SpliceSite::new(SpliceSiteType::Donor, i)
                .with_score(norm)
                .with_dinucleotide(&dn)
                .with_context(&ctx);
            sites.push(site);
        }
    }
    sites
}

/// Scan for AG acceptor sites using a PWM.
pub fn scan_acceptor_sites(sequence: &str, pwm: &SplicePwm, min_score: f64) -> Vec<SpliceSite> {
    let seq = sequence.to_uppercase();
    let bytes = seq.as_bytes();
    let mut sites = Vec::new();
    let pwm_len = pwm.matrix.len();

    if bytes.len() < pwm_len {
        return sites;
    }

    for i in 0..=(bytes.len() - pwm_len) {
        let norm = pwm.normalized_score(&bytes[i..]);
        if norm >= min_score {
            // AG at offset 8-9 in default acceptor PWM
            let ag_start = i + 8;
            let dn = if ag_start + 2 <= bytes.len() {
                seq[ag_start..ag_start + 2].to_string()
            } else {
                String::new()
            };
            let ctx_end = (i + pwm_len).min(bytes.len());
            let ctx = seq[i..ctx_end].to_string();
            let site = SpliceSite::new(SpliceSiteType::Acceptor, i)
                .with_score(norm)
                .with_dinucleotide(&dn)
                .with_context(&ctx);
            sites.push(site);
        }
    }
    sites
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splice_site_type_display() {
        assert_eq!(format!("{}", SpliceSiteType::Donor), "donor(5')");
        assert_eq!(format!("{}", SpliceSiteType::Acceptor), "acceptor(3')");
    }

    #[test]
    fn test_splice_strength_display() {
        assert_eq!(format!("{}", SpliceStrength::Strong), "strong");
        assert_eq!(format!("{}", SpliceStrength::Moderate), "moderate");
        assert_eq!(format!("{}", SpliceStrength::Weak), "weak");
    }

    #[test]
    fn test_classify_strength() {
        assert_eq!(classify_strength(0.9), SpliceStrength::Strong);
        assert_eq!(classify_strength(0.6), SpliceStrength::Moderate);
        assert_eq!(classify_strength(0.3), SpliceStrength::Weak);
    }

    #[test]
    fn test_splice_site_builders() {
        let site = SpliceSite::new(SpliceSiteType::Donor, 100)
            .with_score(0.85)
            .with_dinucleotide("GT")
            .with_context("AAGGTAAGT");
        assert_eq!(site.position, 100);
        assert_eq!(site.strength, SpliceStrength::Strong);
        assert!(site.is_canonical());
    }

    #[test]
    fn test_splice_site_canonical() {
        let donor = SpliceSite::new(SpliceSiteType::Donor, 0).with_dinucleotide("GT");
        assert!(donor.is_canonical());
        let acceptor = SpliceSite::new(SpliceSiteType::Acceptor, 0).with_dinucleotide("AG");
        assert!(acceptor.is_canonical());
        let non = SpliceSite::new(SpliceSiteType::Donor, 0).with_dinucleotide("GC");
        assert!(!non.is_canonical());
    }

    #[test]
    fn test_splice_site_display() {
        let site = SpliceSite::new(SpliceSiteType::Donor, 50)
            .with_score(0.7)
            .with_dinucleotide("GT");
        let s = format!("{}", site);
        assert!(s.contains("donor"));
        assert!(s.contains("GT"));
        assert!(s.contains("moderate"));
    }

    #[test]
    fn test_splice_pwm_score() {
        let pwm = SplicePwm::new(SpliceSiteType::Donor)
            .with_position(2.0, -1.0, -1.0, -1.0)   // A
            .with_position(-1.0, -1.0, 2.0, -1.0);  // G
        let score = pwm.score(b"AG");
        assert!((score - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_splice_pwm_normalized() {
        let pwm = SplicePwm::new(SpliceSiteType::Donor)
            .with_position(2.0, -1.0, -1.0, -1.0)
            .with_position(-1.0, -1.0, 2.0, -1.0);
        let norm = pwm.normalized_score(b"AG");
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_splice_pwm_display() {
        let pwm = SplicePwm::default_donor();
        let s = format!("{}", pwm);
        assert!(s.contains("donor"));
        assert!(s.contains("len=9"));
    }

    #[test]
    fn test_default_donor_pwm() {
        let pwm = SplicePwm::default_donor();
        assert_eq!(pwm.matrix.len(), 9);
    }

    #[test]
    fn test_default_acceptor_pwm() {
        let pwm = SplicePwm::default_acceptor();
        assert_eq!(pwm.matrix.len(), 10);
    }

    #[test]
    fn test_polypyrimidine_score() {
        assert!((polypyrimidine_score(b"TTTTCCCC") - 1.0).abs() < 1e-9);
        assert!((polypyrimidine_score(b"AAAGGGAA") - 0.0).abs() < 1e-9);
        assert!((polypyrimidine_score(b"TTAA") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_polypyrimidine_empty() {
        assert_eq!(polypyrimidine_score(b""), 0.0);
    }

    #[test]
    fn test_intron_exon_lengths() {
        let ie = IntronExon::new(100, 200, 201, 500)
            .with_donor_score(0.9)
            .with_acceptor_score(0.8);
        assert_eq!(ie.exon_length(), 101);
        assert_eq!(ie.intron_length(), 300);
    }

    #[test]
    fn test_intron_exon_display() {
        let ie = IntronExon::new(100, 200, 201, 500);
        let s = format!("{}", ie);
        assert!(s.contains("exon=100-200"));
        assert!(s.contains("intron=201-500"));
    }

    #[test]
    fn test_scan_donor_sites() {
        // Build a consensus-like donor: AAG GT AAG T
        let pwm = SplicePwm::default_donor();
        let seq = "NNNAAGGTAGTNNN";
        let sites = scan_donor_sites(seq, &pwm, 0.3);
        // Should find at least one hit
        assert!(!sites.is_empty() || seq.len() < pwm.matrix.len());
    }

    #[test]
    fn test_scan_config_builders() {
        let config = SpliceScanConfig::new()
            .with_min_score(0.6)
            .with_scan_donors(true)
            .with_scan_acceptors(false)
            .with_context_size(30);
        assert!((config.min_score - 0.6).abs() < 1e-9);
        assert!(!config.scan_acceptors);
        assert_eq!(config.context_size, 30);
    }

    #[test]
    fn test_scan_config_display() {
        let config = SpliceScanConfig::new();
        let s = format!("{}", config);
        assert!(s.contains("min_score"));
    }
}
