//! Variant calling engine with SNP/indel detection, quality scoring,
//! and genotype likelihood computation.
//!
//! Implements pileup-based variant detection, Phred quality score
//! handling, genotype likelihood models (diploid/haploid), allele
//! frequency estimation, variant filtering by quality and depth,
//! and VCF-style variant representation.

use std::fmt;
use std::collections::HashMap;

// ── Variant Type ────────────────────────────────────────────────

/// Classification of a genomic variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariantType {
    Snp,
    Insertion,
    Deletion,
    Mnp,
    Complex,
}

impl fmt::Display for VariantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Snp => write!(f, "SNP"),
            Self::Insertion => write!(f, "INS"),
            Self::Deletion => write!(f, "DEL"),
            Self::Mnp => write!(f, "MNP"),
            Self::Complex => write!(f, "COMPLEX"),
        }
    }
}

// ── Genotype ────────────────────────────────────────────────────

/// Diploid genotype representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Genotype {
    HomRef,
    Het,
    HomAlt,
    HemiRef,
    HemiAlt,
    Unknown,
}

impl fmt::Display for Genotype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HomRef => write!(f, "0/0"),
            Self::Het => write!(f, "0/1"),
            Self::HomAlt => write!(f, "1/1"),
            Self::HemiRef => write!(f, "0"),
            Self::HemiAlt => write!(f, "1"),
            Self::Unknown => write!(f, "./."),
        }
    }
}

// ── Phred Quality ───────────────────────────────────────────────

/// Convert a Phred quality score to an error probability.
pub fn phred_to_prob(phred: f64) -> f64 {
    10.0_f64.powf(-phred / 10.0)
}

/// Convert an error probability to a Phred quality score.
pub fn prob_to_phred(prob: f64) -> f64 {
    if prob <= 0.0 {
        return 999.0; // cap
    }
    -10.0 * prob.log10()
}

// ── Allele Observation ──────────────────────────────────────────

/// A single base observation from a read at a position.
#[derive(Debug, Clone)]
pub struct AlleleObs {
    pub base: char,
    pub quality: f64,
    pub mapping_quality: f64,
    pub is_forward: bool,
}

impl AlleleObs {
    pub fn new(base: char, quality: f64) -> Self {
        Self {
            base,
            quality,
            mapping_quality: 60.0,
            is_forward: true,
        }
    }

    pub fn with_mapping_quality(mut self, mq: f64) -> Self {
        self.mapping_quality = mq;
        self
    }

    pub fn with_strand(mut self, forward: bool) -> Self {
        self.is_forward = forward;
        self
    }

    /// Error probability for this observation.
    pub fn error_prob(&self) -> f64 {
        phred_to_prob(self.quality)
    }
}

impl fmt::Display for AlleleObs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(Q={:.0}, MQ={:.0}, {})",
            self.base, self.quality, self.mapping_quality,
            if self.is_forward { "+" } else { "-" },
        )
    }
}

// ── Pileup Column ───────────────────────────────────────────────

/// All observations at a single genomic position.
#[derive(Debug, Clone)]
pub struct PileupColumn {
    pub chrom: String,
    pub position: usize,
    pub ref_base: char,
    pub observations: Vec<AlleleObs>,
}

impl PileupColumn {
    pub fn new(chrom: &str, position: usize, ref_base: char) -> Self {
        Self {
            chrom: chrom.to_string(),
            position,
            ref_base,
            observations: Vec::new(),
        }
    }

    pub fn add_observation(&mut self, obs: AlleleObs) {
        self.observations.push(obs);
    }

    /// Total depth at this position.
    pub fn depth(&self) -> usize {
        self.observations.len()
    }

    /// Count observations for each allele.
    pub fn allele_counts(&self) -> HashMap<char, usize> {
        let mut counts = HashMap::new();
        for obs in &self.observations {
            *counts.entry(obs.base).or_insert(0) += 1;
        }
        counts
    }

    /// Most frequent non-reference allele.
    pub fn alt_allele(&self) -> Option<(char, usize)> {
        let counts = self.allele_counts();
        counts.into_iter()
            .filter(|(base, _)| *base != self.ref_base)
            .max_by_key(|(_, count)| *count)
    }

    /// Allele frequency of the alternate allele.
    pub fn alt_frequency(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let alt_count = self.alt_allele().map_or(0, |(_, c)| c);
        alt_count as f64 / self.observations.len() as f64
    }

    /// Strand bias: fraction of alt-allele reads on forward strand.
    pub fn strand_bias(&self) -> f64 {
        let alt = self.alt_allele();
        if alt.is_none() {
            return 0.5;
        }
        let alt_base = alt.unwrap().0;
        let alt_obs: Vec<&AlleleObs> = self.observations.iter()
            .filter(|o| o.base == alt_base)
            .collect();
        if alt_obs.is_empty() {
            return 0.5;
        }
        let fwd = alt_obs.iter().filter(|o| o.is_forward).count();
        fwd as f64 / alt_obs.len() as f64
    }
}

impl fmt::Display for PileupColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pileup({}:{}, ref={}, depth={})",
            self.chrom, self.position, self.ref_base, self.depth(),
        )
    }
}

// ── Genotype Likelihood ─────────────────────────────────────────

/// Compute diploid genotype likelihoods for a pileup column.
///
/// Returns (P(0/0), P(0/1), P(1/1)) as Phred-scaled likelihoods.
pub fn genotype_likelihoods(pileup: &PileupColumn) -> (f64, f64, f64) {
    let ref_base = pileup.ref_base;
    let alt = pileup.alt_allele();
    if alt.is_none() {
        return (0.0, 999.0, 999.0);
    }
    let alt_base = alt.unwrap().0;

    let mut ll_hom_ref = 0.0_f64;
    let mut ll_het = 0.0_f64;
    let mut ll_hom_alt = 0.0_f64;

    for obs in &pileup.observations {
        let e = obs.error_prob().max(1e-10);
        let correct = (1.0 - e).max(1e-10);

        if obs.base == ref_base {
            ll_hom_ref += correct.ln();
            ll_het += (0.5 * correct + 0.5 * e).ln();
            ll_hom_alt += e.ln();
        } else if obs.base == alt_base {
            ll_hom_ref += e.ln();
            ll_het += (0.5 * correct + 0.5 * e).ln();
            ll_hom_alt += correct.ln();
        } else {
            // Third allele: treat as error for all genotypes
            ll_hom_ref += e.ln();
            ll_het += e.ln();
            ll_hom_alt += e.ln();
        }
    }

    // Normalize to Phred scale relative to best
    let max_ll = ll_hom_ref.max(ll_het).max(ll_hom_alt);
    let pl_hom_ref = -10.0 * (ll_hom_ref - max_ll) / std::f64::consts::LN_10;
    let pl_het = -10.0 * (ll_het - max_ll) / std::f64::consts::LN_10;
    let pl_hom_alt = -10.0 * (ll_hom_alt - max_ll) / std::f64::consts::LN_10;

    (pl_hom_ref, pl_het, pl_hom_alt)
}

/// Call the most likely genotype from a pileup.
pub fn call_genotype(pileup: &PileupColumn) -> (Genotype, f64) {
    let (pl_rr, pl_ra, pl_aa) = genotype_likelihoods(pileup);

    if pl_rr <= pl_ra && pl_rr <= pl_aa {
        let gq = pl_ra.min(pl_aa);
        (Genotype::HomRef, gq)
    } else if pl_ra <= pl_rr && pl_ra <= pl_aa {
        let gq = pl_rr.min(pl_aa);
        (Genotype::Het, gq)
    } else {
        let gq = pl_rr.min(pl_ra);
        (Genotype::HomAlt, gq)
    }
}

// ── Variant Record ──────────────────────────────────────────────

/// A called variant in VCF-like format.
#[derive(Debug, Clone)]
pub struct VariantRecord {
    pub chrom: String,
    pub position: usize,
    pub id: String,
    pub ref_allele: String,
    pub alt_allele: String,
    pub quality: f64,
    pub filter: String,
    pub variant_type: VariantType,
    pub genotype: Genotype,
    pub genotype_quality: f64,
    pub depth: usize,
    pub alt_depth: usize,
    pub allele_frequency: f64,
}

impl VariantRecord {
    pub fn new(chrom: &str, position: usize, ref_allele: &str, alt_allele: &str) -> Self {
        let vtype = classify_variant(ref_allele, alt_allele);
        Self {
            chrom: chrom.to_string(),
            position,
            id: ".".to_string(),
            ref_allele: ref_allele.to_string(),
            alt_allele: alt_allele.to_string(),
            quality: 0.0,
            filter: "PASS".to_string(),
            variant_type: vtype,
            genotype: Genotype::Unknown,
            genotype_quality: 0.0,
            depth: 0,
            alt_depth: 0,
            allele_frequency: 0.0,
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn with_quality(mut self, q: f64) -> Self {
        self.quality = q;
        self
    }

    pub fn with_filter(mut self, f_str: &str) -> Self {
        self.filter = f_str.to_string();
        self
    }

    pub fn with_genotype(mut self, gt: Genotype, gq: f64) -> Self {
        self.genotype = gt;
        self.genotype_quality = gq;
        self
    }

    pub fn with_depth(mut self, total: usize, alt: usize) -> Self {
        self.depth = total;
        self.alt_depth = alt;
        if total > 0 {
            self.allele_frequency = alt as f64 / total as f64;
        }
        self
    }

    /// Whether this variant passes all filters.
    pub fn is_passing(&self) -> bool {
        self.filter == "PASS"
    }

    /// VCF line representation.
    pub fn to_vcf_line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\tGT:GQ:DP\t{}:{:.0}:{}",
            self.chrom, self.position, self.id,
            self.ref_allele, self.alt_allele,
            self.quality, self.filter,
            self.genotype, self.genotype_quality, self.depth,
        )
    }
}

impl fmt::Display for VariantRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{} {}>{}({}, {}, Q={:.1}, DP={})",
            self.chrom, self.position,
            self.ref_allele, self.alt_allele,
            self.variant_type, self.genotype,
            self.quality, self.depth,
        )
    }
}

// ── Variant Classification ──────────────────────────────────────

/// Classify a variant from ref/alt alleles.
pub fn classify_variant(ref_allele: &str, alt_allele: &str) -> VariantType {
    let rlen = ref_allele.len();
    let alen = alt_allele.len();
    if rlen == 1 && alen == 1 {
        VariantType::Snp
    } else if rlen < alen && alen > 1 {
        VariantType::Insertion
    } else if rlen > alen && rlen > 1 {
        VariantType::Deletion
    } else if rlen == alen && rlen > 1 {
        VariantType::Mnp
    } else {
        VariantType::Complex
    }
}

// ── Variant Filter ──────────────────────────────────────────────

/// Filter criteria for variant calls.
#[derive(Debug, Clone)]
pub struct VariantFilter {
    pub min_quality: f64,
    pub min_depth: usize,
    pub min_alt_depth: usize,
    pub min_allele_frequency: f64,
    pub max_strand_bias: f64,
    pub min_genotype_quality: f64,
}

impl VariantFilter {
    pub fn new() -> Self {
        Self {
            min_quality: 20.0,
            min_depth: 5,
            min_alt_depth: 2,
            min_allele_frequency: 0.05,
            max_strand_bias: 0.95,
            min_genotype_quality: 10.0,
        }
    }

    pub fn with_min_quality(mut self, q: f64) -> Self {
        self.min_quality = q;
        self
    }

    pub fn with_min_depth(mut self, d: usize) -> Self {
        self.min_depth = d;
        self
    }

    pub fn with_min_alt_depth(mut self, d: usize) -> Self {
        self.min_alt_depth = d;
        self
    }

    pub fn with_min_allele_frequency(mut self, af: f64) -> Self {
        self.min_allele_frequency = af;
        self
    }

    pub fn with_min_genotype_quality(mut self, gq: f64) -> Self {
        self.min_genotype_quality = gq;
        self
    }

    /// Apply filter to a variant and return the filter status string.
    pub fn apply(&self, variant: &VariantRecord) -> String {
        let mut reasons = Vec::new();
        if variant.quality < self.min_quality {
            reasons.push("LowQual");
        }
        if variant.depth < self.min_depth {
            reasons.push("LowDepth");
        }
        if variant.alt_depth < self.min_alt_depth {
            reasons.push("LowAltDepth");
        }
        if variant.allele_frequency < self.min_allele_frequency {
            reasons.push("LowAF");
        }
        if variant.genotype_quality < self.min_genotype_quality {
            reasons.push("LowGQ");
        }
        if reasons.is_empty() {
            "PASS".to_string()
        } else {
            reasons.join(";")
        }
    }
}

impl fmt::Display for VariantFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VariantFilter(Q>={:.0}, DP>={}, AD>={}, AF>={:.2}, GQ>={:.0})",
            self.min_quality, self.min_depth, self.min_alt_depth,
            self.min_allele_frequency, self.min_genotype_quality,
        )
    }
}

// ── Call Variants from Pileups ──────────────────────────────────

/// Call variants from a series of pileup columns.
pub fn call_variants(pileups: &[PileupColumn], filter: &VariantFilter) -> Vec<VariantRecord> {
    let mut variants = Vec::new();

    for pileup in pileups {
        if let Some((alt_base, alt_count)) = pileup.alt_allele() {
            let (gt, gq) = call_genotype(pileup);
            if gt == Genotype::HomRef {
                continue;
            }
            let qual = gq;
            let mut record = VariantRecord::new(
                &pileup.chrom,
                pileup.position,
                &pileup.ref_base.to_string(),
                &alt_base.to_string(),
            )
            .with_quality(qual)
            .with_genotype(gt, gq)
            .with_depth(pileup.depth(), alt_count);

            let status = filter.apply(&record);
            record.filter = status;
            variants.push(record);
        }
    }
    variants
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_type_display() {
        assert_eq!(format!("{}", VariantType::Snp), "SNP");
        assert_eq!(format!("{}", VariantType::Insertion), "INS");
        assert_eq!(format!("{}", VariantType::Deletion), "DEL");
    }

    #[test]
    fn test_genotype_display() {
        assert_eq!(format!("{}", Genotype::HomRef), "0/0");
        assert_eq!(format!("{}", Genotype::Het), "0/1");
        assert_eq!(format!("{}", Genotype::HomAlt), "1/1");
    }

    #[test]
    fn test_phred_to_prob() {
        assert!((phred_to_prob(10.0) - 0.1).abs() < 1e-9);
        assert!((phred_to_prob(20.0) - 0.01).abs() < 1e-9);
        assert!((phred_to_prob(30.0) - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_prob_to_phred() {
        assert!((prob_to_phred(0.01) - 20.0).abs() < 1e-9);
        assert!((prob_to_phred(0.001) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_allele_obs_error_prob() {
        let obs = AlleleObs::new('A', 30.0);
        assert!((obs.error_prob() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_allele_obs_display() {
        let obs = AlleleObs::new('G', 25.0).with_strand(false);
        let s = format!("{}", obs);
        assert!(s.contains("G"));
        assert!(s.contains("-"));
    }

    #[test]
    fn test_pileup_depth() {
        let mut pileup = PileupColumn::new("chr1", 100, 'A');
        pileup.add_observation(AlleleObs::new('A', 30.0));
        pileup.add_observation(AlleleObs::new('A', 30.0));
        pileup.add_observation(AlleleObs::new('G', 30.0));
        assert_eq!(pileup.depth(), 3);
    }

    #[test]
    fn test_pileup_allele_counts() {
        let mut pileup = PileupColumn::new("chr1", 100, 'A');
        pileup.add_observation(AlleleObs::new('A', 30.0));
        pileup.add_observation(AlleleObs::new('A', 30.0));
        pileup.add_observation(AlleleObs::new('G', 30.0));
        let counts = pileup.allele_counts();
        assert_eq!(counts[&'A'], 2);
        assert_eq!(counts[&'G'], 1);
    }

    #[test]
    fn test_pileup_alt_frequency() {
        let mut pileup = PileupColumn::new("chr1", 100, 'A');
        for _ in 0..5 { pileup.add_observation(AlleleObs::new('A', 30.0)); }
        for _ in 0..5 { pileup.add_observation(AlleleObs::new('G', 30.0)); }
        assert!((pileup.alt_frequency() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_classify_variant_snp() {
        assert_eq!(classify_variant("A", "G"), VariantType::Snp);
    }

    #[test]
    fn test_classify_variant_insertion() {
        assert_eq!(classify_variant("A", "ATG"), VariantType::Insertion);
    }

    #[test]
    fn test_classify_variant_deletion() {
        assert_eq!(classify_variant("ATG", "A"), VariantType::Deletion);
    }

    #[test]
    fn test_genotype_likelihood_hom_ref() {
        let mut pileup = PileupColumn::new("chr1", 100, 'A');
        for _ in 0..20 { pileup.add_observation(AlleleObs::new('A', 30.0)); }
        pileup.add_observation(AlleleObs::new('G', 10.0)); // noise
        let (pl_rr, pl_ra, pl_aa) = genotype_likelihoods(&pileup);
        assert!(pl_rr < pl_ra);
        assert!(pl_rr < pl_aa);
    }

    #[test]
    fn test_genotype_likelihood_hom_alt() {
        let mut pileup = PileupColumn::new("chr1", 100, 'A');
        for _ in 0..20 { pileup.add_observation(AlleleObs::new('G', 30.0)); }
        let (pl_rr, _pl_ra, pl_aa) = genotype_likelihoods(&pileup);
        assert!(pl_aa < pl_rr);
    }

    #[test]
    fn test_call_genotype_het() {
        let mut pileup = PileupColumn::new("chr1", 100, 'A');
        for _ in 0..10 { pileup.add_observation(AlleleObs::new('A', 30.0)); }
        for _ in 0..10 { pileup.add_observation(AlleleObs::new('G', 30.0)); }
        let (gt, _gq) = call_genotype(&pileup);
        assert_eq!(gt, Genotype::Het);
    }

    #[test]
    fn test_variant_record_builders() {
        let v = VariantRecord::new("chr1", 100, "A", "G")
            .with_id("rs123")
            .with_quality(50.0)
            .with_genotype(Genotype::Het, 30.0)
            .with_depth(20, 10);
        assert_eq!(v.id, "rs123");
        assert!(v.is_passing());
        assert!((v.allele_frequency - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_variant_record_vcf_line() {
        let v = VariantRecord::new("chr1", 100, "A", "G")
            .with_quality(30.0)
            .with_genotype(Genotype::Het, 25.0)
            .with_depth(20, 10);
        let line = v.to_vcf_line();
        assert!(line.contains("chr1\t100"));
        assert!(line.contains("A\tG"));
    }

    #[test]
    fn test_variant_filter_apply_pass() {
        let filter = VariantFilter::new();
        let v = VariantRecord::new("chr1", 100, "A", "G")
            .with_quality(30.0)
            .with_genotype(Genotype::Het, 20.0)
            .with_depth(20, 10);
        assert_eq!(filter.apply(&v), "PASS");
    }

    #[test]
    fn test_variant_filter_apply_fail() {
        let filter = VariantFilter::new();
        let v = VariantRecord::new("chr1", 100, "A", "G")
            .with_quality(5.0)
            .with_depth(2, 1);
        let status = filter.apply(&v);
        assert!(status.contains("LowQual"));
        assert!(status.contains("LowDepth"));
    }

    #[test]
    fn test_variant_filter_display() {
        let f = VariantFilter::new();
        let s = format!("{}", f);
        assert!(s.contains("Q>=20"));
    }
}
