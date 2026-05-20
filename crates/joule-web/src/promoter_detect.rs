//! Promoter region detection with TATA box scoring, CpG island identification,
//! and position weight matrix motif scanning.
//!
//! Implements prokaryotic and eukaryotic promoter element detection
//! including TATA box consensus matching, CpG island sliding-window
//! analysis, GC content profiling, and PWM-based transcription factor
//! binding site scoring with p-value estimation.

use std::fmt;

// ── Promoter Element Kind ───────────────────────────────────────

/// Type of promoter element detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromoterElement {
    TataBox,
    CpgIsland,
    CaatBox,
    GcBox,
    InrElement,
    MinusTen,
    MinusThirtyFive,
    CustomMotif,
}

impl fmt::Display for PromoterElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TataBox => write!(f, "TATA_box"),
            Self::CpgIsland => write!(f, "CpG_island"),
            Self::CaatBox => write!(f, "CAAT_box"),
            Self::GcBox => write!(f, "GC_box"),
            Self::InrElement => write!(f, "Inr"),
            Self::MinusTen => write!(f, "-10_box"),
            Self::MinusThirtyFive => write!(f, "-35_box"),
            Self::CustomMotif => write!(f, "custom_motif"),
        }
    }
}

// ── Promoter Hit ────────────────────────────────────────────────

/// A detected promoter element at a specific position.
#[derive(Debug, Clone)]
pub struct PromoterHit {
    pub element: PromoterElement,
    pub position: usize,
    pub length: usize,
    pub score: f64,
    pub p_value: f64,
    pub sequence: String,
    pub strand: char,
}

impl PromoterHit {
    pub fn new(element: PromoterElement, position: usize, length: usize) -> Self {
        Self {
            element,
            position,
            length,
            score: 0.0,
            p_value: 1.0,
            sequence: String::new(),
            strand: '+',
        }
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    pub fn with_p_value(mut self, p: f64) -> Self {
        self.p_value = p;
        self
    }

    pub fn with_sequence(mut self, seq: &str) -> Self {
        self.sequence = seq.to_string();
        self
    }

    pub fn with_strand(mut self, strand: char) -> Self {
        self.strand = strand;
        self
    }

    /// End position (inclusive).
    pub fn end(&self) -> usize {
        self.position + self.length.saturating_sub(1)
    }

    /// Whether this hit passes a significance threshold.
    pub fn is_significant(&self, p_threshold: f64) -> bool {
        self.p_value <= p_threshold
    }
}

impl fmt::Display for PromoterHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@{}-{}({}, score={:.3}, p={:.2e})",
            self.element, self.position, self.end(),
            self.strand, self.score, self.p_value,
        )
    }
}

// ── Position Weight Matrix ──────────────────────────────────────

/// A position weight matrix for motif scoring.
///
/// Each row is a position, columns are A, T, G, C log-odds scores.
#[derive(Debug, Clone)]
pub struct Pwm {
    pub name: String,
    pub matrix: Vec<[f64; 4]>,
}

impl Pwm {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            matrix: Vec::new(),
        }
    }

    pub fn with_position(mut self, a: f64, t: f64, g: f64, c: f64) -> Self {
        self.matrix.push([a, t, g, c]);
        self
    }

    /// Length of the motif.
    pub fn length(&self) -> usize {
        self.matrix.len()
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

    /// Score a subsequence against this PWM.
    pub fn score(&self, seq: &[u8]) -> f64 {
        if seq.len() < self.matrix.len() {
            return f64::NEG_INFINITY;
        }
        let mut total = 0.0;
        for (i, weights) in self.matrix.iter().enumerate() {
            if let Some(idx) = Self::base_index(seq[i]) {
                total += weights[idx];
            } else {
                total += -10.0; // penalty for N or unknown
            }
        }
        total
    }

    /// Maximum possible score for this PWM.
    pub fn max_score(&self) -> f64 {
        self.matrix.iter().map(|row| {
            row.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }).sum()
    }

    /// Minimum possible score for this PWM.
    pub fn min_score(&self) -> f64 {
        self.matrix.iter().map(|row| {
            row.iter().copied().fold(f64::INFINITY, f64::min)
        }).sum()
    }

    /// Normalized score (0.0 to 1.0) of a subsequence.
    pub fn normalized_score(&self, seq: &[u8]) -> f64 {
        let raw = self.score(seq);
        let mn = self.min_score();
        let mx = self.max_score();
        if (mx - mn).abs() < 1e-12 { 0.0 } else { (raw - mn) / (mx - mn) }
    }
}

impl fmt::Display for Pwm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PWM({}, len={})", self.name, self.length())
    }
}

// ── TATA Box Detection ──────────────────────────────────────────

/// TATA box consensus: TATA(A/T)A(A/T)(A/G).
/// Returns scored hits above a threshold.
pub fn detect_tata_boxes(sequence: &str, min_score: f64) -> Vec<PromoterHit> {
    // Build a simple TATA PWM: TATAAAA with some flexibility
    let tata_pwm = Pwm::new("TATA_box")
        .with_position(-2.0, 2.0, -2.0, -2.0)   // T
        .with_position(2.0, -2.0, -2.0, -2.0)    // A
        .with_position(-2.0, 2.0, -2.0, -2.0)    // T
        .with_position(2.0, -2.0, -2.0, -2.0)    // A
        .with_position(1.5, 1.0, -2.0, -2.0)     // A/T
        .with_position(2.0, -1.0, -2.0, -2.0)    // A
        .with_position(1.5, 1.0, -2.0, -2.0);    // A/T

    let seq = sequence.to_uppercase();
    let bytes = seq.as_bytes();
    let mut hits = Vec::new();

    if bytes.len() < tata_pwm.length() {
        return hits;
    }

    for i in 0..=(bytes.len() - tata_pwm.length()) {
        let score = tata_pwm.normalized_score(&bytes[i..]);
        if score >= min_score {
            let end = i + tata_pwm.length();
            let subseq = &seq[i..end];
            let hit = PromoterHit::new(PromoterElement::TataBox, i, tata_pwm.length())
                .with_score(score)
                .with_p_value(estimate_p_value(score, tata_pwm.length()))
                .with_sequence(subseq);
            hits.push(hit);
        }
    }
    hits
}

// ── CpG Island Detection ────────────────────────────────────────

/// Parameters for CpG island detection.
#[derive(Debug, Clone)]
pub struct CpgIslandParams {
    pub window_size: usize,
    pub step_size: usize,
    pub min_length: usize,
    pub min_gc_content: f64,
    pub min_obs_exp_cpg: f64,
}

impl CpgIslandParams {
    pub fn new() -> Self {
        Self {
            window_size: 200,
            step_size: 1,
            min_length: 200,
            min_gc_content: 0.5,
            min_obs_exp_cpg: 0.6,
        }
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn with_min_gc_content(mut self, gc: f64) -> Self {
        self.min_gc_content = gc;
        self
    }

    pub fn with_min_obs_exp_cpg(mut self, ratio: f64) -> Self {
        self.min_obs_exp_cpg = ratio;
        self
    }

    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_length = len;
        self
    }
}

impl fmt::Display for CpgIslandParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CpgIslandParams(win={}, gc>={:.0}%, obs/exp>={:.2})",
            self.window_size, self.min_gc_content * 100.0, self.min_obs_exp_cpg,
        )
    }
}

/// A detected CpG island region.
#[derive(Debug, Clone)]
pub struct CpgIsland {
    pub start: usize,
    pub end: usize,
    pub gc_content: f64,
    pub obs_exp_cpg: f64,
    pub cpg_count: usize,
    pub length: usize,
}

impl CpgIsland {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            gc_content: 0.0,
            obs_exp_cpg: 0.0,
            cpg_count: 0,
            length: if end >= start { end - start + 1 } else { 0 },
        }
    }
}

impl fmt::Display for CpgIsland {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CpgIsland({}-{}, len={}, GC={:.1}%, obs/exp={:.2})",
            self.start, self.end, self.length,
            self.gc_content * 100.0, self.obs_exp_cpg,
        )
    }
}

/// Detect CpG islands in a DNA sequence using a sliding window.
pub fn detect_cpg_islands(sequence: &str, params: &CpgIslandParams) -> Vec<CpgIsland> {
    let seq = sequence.to_uppercase();
    let bytes = seq.as_bytes();
    let mut islands = Vec::new();

    if bytes.len() < params.window_size {
        return islands;
    }

    let mut island_start: Option<usize> = None;

    let mut pos = 0;
    while pos + params.window_size <= bytes.len() {
        let window = &bytes[pos..pos + params.window_size];
        let (gc, obs_exp) = compute_cpg_stats(window);

        if gc >= params.min_gc_content && obs_exp >= params.min_obs_exp_cpg {
            if island_start.is_none() {
                island_start = Some(pos);
            }
        } else if let Some(start) = island_start {
            let end = pos + params.window_size - 2;
            let length = end - start + 1;
            if length >= params.min_length {
                let region = &bytes[start..=end];
                let (gc_final, obs_exp_final) = compute_cpg_stats(region);
                let cpg_count = count_cpg_dinucleotides(region);
                let mut island = CpgIsland::new(start, end);
                island.gc_content = gc_final;
                island.obs_exp_cpg = obs_exp_final;
                island.cpg_count = cpg_count;
                islands.push(island);
            }
            island_start = None;
        }
        pos += params.step_size;
    }

    // Close trailing island
    if let Some(start) = island_start {
        let end = bytes.len() - 1;
        let length = end - start + 1;
        if length >= params.min_length {
            let region = &bytes[start..=end];
            let (gc_final, obs_exp_final) = compute_cpg_stats(region);
            let cpg_count = count_cpg_dinucleotides(region);
            let mut island = CpgIsland::new(start, end);
            island.gc_content = gc_final;
            island.obs_exp_cpg = obs_exp_final;
            island.cpg_count = cpg_count;
            islands.push(island);
        }
    }

    islands
}

// ── GC Content ──────────────────────────────────────────────────

/// Compute GC content of a sequence.
pub fn gc_content(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let gc = seq.iter().filter(|&&b| b == b'G' || b == b'C').count();
    gc as f64 / seq.len() as f64
}

fn compute_cpg_stats(seq: &[u8]) -> (f64, f64) {
    let n = seq.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let gc = gc_content(seq);
    let c_count = seq.iter().filter(|&&b| b == b'C').count() as f64;
    let g_count = seq.iter().filter(|&&b| b == b'G').count() as f64;
    let cpg_count = count_cpg_dinucleotides(seq) as f64;
    let expected = (c_count * g_count) / n;
    let obs_exp = if expected > 0.0 { cpg_count / expected } else { 0.0 };
    (gc, obs_exp)
}

fn count_cpg_dinucleotides(seq: &[u8]) -> usize {
    if seq.len() < 2 {
        return 0;
    }
    seq.windows(2).filter(|w| w[0] == b'C' && w[1] == b'G').count()
}

// ── P-value Estimation ──────────────────────────────────────────

/// Simple p-value estimate based on motif score and length.
fn estimate_p_value(normalized_score: f64, motif_len: usize) -> f64 {
    // Approximate: assume scores follow a Gumbel distribution
    let k = motif_len as f64;
    let lambda = 0.7 * k.ln();
    let mu = 0.5;
    let z = (normalized_score - mu) * lambda;
    (1.0 - (-(-z).exp()).exp()).max(1e-300).min(1.0)
}

// ── Motif Scanning ──────────────────────────────────────────────

/// Scan a sequence with a PWM and return hits above a threshold.
pub fn scan_pwm(sequence: &str, pwm: &Pwm, min_normalized_score: f64) -> Vec<PromoterHit> {
    let seq = sequence.to_uppercase();
    let bytes = seq.as_bytes();
    let mut hits = Vec::new();

    if bytes.len() < pwm.length() {
        return hits;
    }

    for i in 0..=(bytes.len() - pwm.length()) {
        let norm = pwm.normalized_score(&bytes[i..]);
        if norm >= min_normalized_score {
            let subseq = &seq[i..i + pwm.length()];
            let hit = PromoterHit::new(PromoterElement::CustomMotif, i, pwm.length())
                .with_score(norm)
                .with_p_value(estimate_p_value(norm, pwm.length()))
                .with_sequence(subseq);
            hits.push(hit);
        }
    }
    hits
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promoter_element_display() {
        assert_eq!(format!("{}", PromoterElement::TataBox), "TATA_box");
        assert_eq!(format!("{}", PromoterElement::CpgIsland), "CpG_island");
        assert_eq!(format!("{}", PromoterElement::MinusTen), "-10_box");
    }

    #[test]
    fn test_promoter_hit_basic() {
        let hit = PromoterHit::new(PromoterElement::TataBox, 100, 7)
            .with_score(0.9)
            .with_p_value(0.001)
            .with_sequence("TATAAAA")
            .with_strand('+');
        assert_eq!(hit.end(), 106);
        assert!(hit.is_significant(0.01));
        assert!(!hit.is_significant(0.0001));
    }

    #[test]
    fn test_promoter_hit_display() {
        let hit = PromoterHit::new(PromoterElement::TataBox, 50, 7)
            .with_score(0.85)
            .with_p_value(0.005);
        let s = format!("{}", hit);
        assert!(s.contains("TATA_box"));
        assert!(s.contains("50"));
    }

    #[test]
    fn test_pwm_score() {
        let pwm = Pwm::new("test")
            .with_position(2.0, -1.0, -1.0, -1.0)   // A
            .with_position(-1.0, 2.0, -1.0, -1.0)   // T
            .with_position(-1.0, -1.0, 2.0, -1.0);  // G
        let score = pwm.score(b"ATG");
        assert!((score - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_pwm_max_min_score() {
        let pwm = Pwm::new("test")
            .with_position(2.0, -1.0, -1.0, -1.0)
            .with_position(-1.0, 2.0, -1.0, -1.0);
        assert!((pwm.max_score() - 4.0).abs() < 1e-9);
        assert!((pwm.min_score() - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_pwm_normalized_score() {
        let pwm = Pwm::new("test")
            .with_position(2.0, -1.0, -1.0, -1.0)
            .with_position(-1.0, 2.0, -1.0, -1.0);
        // Perfect match = max score => normalized = 1.0
        let norm = pwm.normalized_score(b"AT");
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pwm_display() {
        let pwm = Pwm::new("TATA").with_position(1.0, 0.0, 0.0, 0.0);
        let s = format!("{}", pwm);
        assert!(s.contains("TATA"));
        assert!(s.contains("len=1"));
    }

    #[test]
    fn test_detect_tata_boxes() {
        let seq = "NNNNNNNTATAAAANNNNN";
        let hits = detect_tata_boxes(seq, 0.5);
        assert!(!hits.is_empty());
        let best = hits.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        assert!(best.score > 0.5);
    }

    #[test]
    fn test_detect_tata_boxes_no_hit() {
        let seq = "GGGCCCCGGGCCCC";
        let hits = detect_tata_boxes(seq, 0.9);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_gc_content() {
        assert!((gc_content(b"GCGC") - 1.0).abs() < 1e-9);
        assert!((gc_content(b"ATAT") - 0.0).abs() < 1e-9);
        assert!((gc_content(b"ATGC") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_gc_content_empty() {
        assert_eq!(gc_content(b""), 0.0);
    }

    #[test]
    fn test_count_cpg_dinucleotides() {
        assert_eq!(count_cpg_dinucleotides(b"CGCGCG"), 3);
        assert_eq!(count_cpg_dinucleotides(b"AAAA"), 0);
        assert_eq!(count_cpg_dinucleotides(b"CG"), 1);
    }

    #[test]
    fn test_cpg_island_params_builders() {
        let params = CpgIslandParams::new()
            .with_window_size(100)
            .with_min_gc_content(0.6)
            .with_min_obs_exp_cpg(0.7)
            .with_min_length(150);
        assert_eq!(params.window_size, 100);
        assert!((params.min_gc_content - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_cpg_island_params_display() {
        let params = CpgIslandParams::new();
        let s = format!("{}", params);
        assert!(s.contains("win=200"));
    }

    #[test]
    fn test_detect_cpg_islands_high_cpg() {
        // 300 CG repeats = highly CpG rich
        let seq = "CG".repeat(150);
        let params = CpgIslandParams::new()
            .with_window_size(50)
            .with_min_length(50);
        let islands = detect_cpg_islands(&seq, &params);
        assert!(!islands.is_empty());
        assert!(islands[0].gc_content > 0.9);
    }

    #[test]
    fn test_detect_cpg_islands_no_cpg() {
        let seq = "A".repeat(500);
        let params = CpgIslandParams::new();
        let islands = detect_cpg_islands(&seq, &params);
        assert!(islands.is_empty());
    }

    #[test]
    fn test_scan_pwm() {
        let pwm = Pwm::new("test")
            .with_position(2.0, -2.0, -2.0, -2.0)  // A
            .with_position(-2.0, 2.0, -2.0, -2.0)  // T
            .with_position(-2.0, -2.0, 2.0, -2.0); // G
        let hits = scan_pwm("NNATGNNN", &pwm, 0.9);
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_cpg_island_display() {
        let mut island = CpgIsland::new(100, 500);
        island.gc_content = 0.65;
        island.obs_exp_cpg = 0.8;
        let s = format!("{}", island);
        assert!(s.contains("100-500"));
        assert!(s.contains("65.0%"));
    }

    #[test]
    fn test_estimate_p_value_range() {
        let p1 = estimate_p_value(0.0, 7);
        let p2 = estimate_p_value(1.0, 7);
        // Higher score should give lower p-value
        assert!(p2 <= p1 || (p2 - p1).abs() < 1e-6);
    }
}
