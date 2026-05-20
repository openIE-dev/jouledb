//! Gene prediction engine with ORF scoring, start/stop codon detection,
//! and gene model building.
//!
//! Implements ab initio gene prediction using coding potential scoring,
//! hexamer frequency statistics, start codon (ATG/GTG/TTG) context
//! evaluation, stop codon recognition (TAA/TAG/TGA), and Markov chain
//! models for coding vs. non-coding sequence discrimination.

use std::fmt;
use std::collections::HashMap;

// ── Codon Type ──────────────────────────────────────────────────

/// Classification of a trinucleotide codon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodonType {
    Start,
    Stop,
    Sense,
}

impl fmt::Display for CodonType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start => write!(f, "start"),
            Self::Stop => write!(f, "stop"),
            Self::Sense => write!(f, "sense"),
        }
    }
}

/// Classify a trinucleotide codon string.
pub fn classify_codon(codon: &str) -> CodonType {
    let upper = codon.to_uppercase();
    match upper.as_str() {
        "ATG" | "GTG" | "TTG" => CodonType::Start,
        "TAA" | "TAG" | "TGA" => CodonType::Stop,
        _ => CodonType::Sense,
    }
}

// ── Gene Strand ─────────────────────────────────────────────────

/// Strand for gene prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredStrand {
    Forward,
    Reverse,
}

impl fmt::Display for PredStrand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward => write!(f, "+"),
            Self::Reverse => write!(f, "-"),
        }
    }
}

// ── Predicted Gene ──────────────────────────────────────────────

/// A predicted gene with coding potential score and coordinates.
#[derive(Debug, Clone)]
pub struct PredictedGene {
    pub id: String,
    pub start: usize,
    pub end: usize,
    pub strand: PredStrand,
    pub frame: u8,
    pub start_codon: String,
    pub stop_codon: String,
    pub coding_score: f64,
    pub start_context_score: f64,
    pub length_bp: usize,
}

impl PredictedGene {
    pub fn new(start: usize, end: usize, strand: PredStrand, frame: u8) -> Self {
        let length_bp = if end >= start { end - start + 1 } else { 0 };
        Self {
            id: String::new(),
            start,
            end,
            strand,
            frame,
            start_codon: String::new(),
            stop_codon: String::new(),
            coding_score: 0.0,
            start_context_score: 0.0,
            length_bp,
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn with_start_codon(mut self, codon: &str) -> Self {
        self.start_codon = codon.to_uppercase();
        self
    }

    pub fn with_stop_codon(mut self, codon: &str) -> Self {
        self.stop_codon = codon.to_uppercase();
        self
    }

    pub fn with_coding_score(mut self, score: f64) -> Self {
        self.coding_score = score;
        self
    }

    pub fn with_start_context_score(mut self, score: f64) -> Self {
        self.start_context_score = score;
        self
    }

    /// Combined score weighting coding potential and start context.
    pub fn combined_score(&self, coding_weight: f64, context_weight: f64) -> f64 {
        self.coding_score * coding_weight + self.start_context_score * context_weight
    }

    /// Number of codons in this predicted gene.
    pub fn codon_count(&self) -> usize {
        self.length_bp / 3
    }
}

impl fmt::Display for PredictedGene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gene({}, {}-{}, {}, frame={}, score={:.3})",
            self.id, self.start, self.end, self.strand, self.frame, self.coding_score,
        )
    }
}

// ── Hexamer Statistics ──────────────────────────────────────────

/// Hexamer (6-mer) frequency table for coding potential evaluation.
#[derive(Debug, Clone)]
pub struct HexamerTable {
    pub coding_freqs: HashMap<String, f64>,
    pub noncoding_freqs: HashMap<String, f64>,
}

impl HexamerTable {
    pub fn new() -> Self {
        Self {
            coding_freqs: HashMap::new(),
            noncoding_freqs: HashMap::new(),
        }
    }

    /// Train the table from a coding sequence.
    pub fn train_coding(&mut self, sequence: &str) {
        let seq = sequence.to_uppercase();
        let bytes = seq.as_bytes();
        if bytes.len() < 6 {
            return;
        }
        let total = (bytes.len() - 5) as f64;
        for i in 0..=(bytes.len() - 6) {
            let hexamer = &seq[i..i + 6];
            if hexamer.bytes().all(|b| matches!(b, b'A' | b'T' | b'G' | b'C')) {
                *self.coding_freqs.entry(hexamer.to_string()).or_insert(0.0) += 1.0 / total;
            }
        }
    }

    /// Train the table from a non-coding sequence.
    pub fn train_noncoding(&mut self, sequence: &str) {
        let seq = sequence.to_uppercase();
        let bytes = seq.as_bytes();
        if bytes.len() < 6 {
            return;
        }
        let total = (bytes.len() - 5) as f64;
        for i in 0..=(bytes.len() - 6) {
            let hexamer = &seq[i..i + 6];
            if hexamer.bytes().all(|b| matches!(b, b'A' | b'T' | b'G' | b'C')) {
                *self.noncoding_freqs.entry(hexamer.to_string()).or_insert(0.0) += 1.0 / total;
            }
        }
    }

    /// Log-likelihood ratio score for a sequence.
    pub fn score_sequence(&self, sequence: &str) -> f64 {
        let seq = sequence.to_uppercase();
        let bytes = seq.as_bytes();
        if bytes.len() < 6 {
            return 0.0;
        }
        let mut score = 0.0;
        let pseudo = 1e-6;
        for i in 0..=(bytes.len() - 6) {
            let hexamer = &seq[i..i + 6];
            let coding = self.coding_freqs.get(hexamer).copied().unwrap_or(pseudo);
            let noncoding = self.noncoding_freqs.get(hexamer).copied().unwrap_or(pseudo);
            score += (coding / noncoding).ln();
        }
        score
    }
}

impl fmt::Display for HexamerTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HexamerTable(coding={}, noncoding={})",
            self.coding_freqs.len(),
            self.noncoding_freqs.len(),
        )
    }
}

// ── Start Context Scorer ────────────────────────────────────────

/// Evaluate the Kozak/Shine-Dalgarno-like context around a start codon.
#[derive(Debug, Clone)]
pub struct StartContextScorer {
    /// Position weight matrix (relative to ATG, -6 to +4).
    pub pwm: Vec<[f64; 4]>,
    /// Offset of the ATG within the PWM window.
    pub atg_offset: usize,
}

impl StartContextScorer {
    /// Create a scorer with a default bacterial-like PWM.
    pub fn new_default() -> Self {
        // Simple model: favor A at -3 (Kozak) and G at +4
        let mut pwm = vec![[0.25_f64; 4]; 11]; // positions -6 to +4
        // Position -3 (index 3): strong A preference
        pwm[3] = [0.6, 0.1, 0.2, 0.1]; // A, T, G, C
        // Position +4 (index 10): G preference
        pwm[10] = [0.1, 0.1, 0.6, 0.2];
        Self { pwm, atg_offset: 6 }
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

    /// Score the context around a start codon at `atg_pos` in the sequence.
    pub fn score(&self, sequence: &[u8], atg_pos: usize) -> f64 {
        let window_start = if atg_pos >= self.atg_offset {
            atg_pos - self.atg_offset
        } else {
            return 0.0;
        };
        let window_end = window_start + self.pwm.len();
        if window_end > sequence.len() {
            return 0.0;
        }

        let mut log_score = 0.0;
        for (i, &weights) in self.pwm.iter().enumerate() {
            let base = sequence[window_start + i];
            if let Some(idx) = Self::base_index(base) {
                log_score += weights[idx].max(1e-10).ln();
            }
        }
        log_score
    }
}

impl fmt::Display for StartContextScorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StartContextScorer(pwm_len={}, atg_offset={})", self.pwm.len(), self.atg_offset)
    }
}

// ── Gene Predictor ──────────────────────────────────────────────

/// Configuration for the gene prediction engine.
#[derive(Debug, Clone)]
pub struct GenePredictorConfig {
    pub min_orf_length: usize,
    pub coding_weight: f64,
    pub context_weight: f64,
    pub score_threshold: f64,
    pub allow_gtg_start: bool,
    pub allow_ttg_start: bool,
}

impl GenePredictorConfig {
    pub fn new() -> Self {
        Self {
            min_orf_length: 90,
            coding_weight: 0.7,
            context_weight: 0.3,
            score_threshold: 0.0,
            allow_gtg_start: true,
            allow_ttg_start: true,
        }
    }

    pub fn with_min_orf_length(mut self, len: usize) -> Self {
        self.min_orf_length = len;
        self
    }

    pub fn with_score_threshold(mut self, thresh: f64) -> Self {
        self.score_threshold = thresh;
        self
    }

    pub fn with_coding_weight(mut self, w: f64) -> Self {
        self.coding_weight = w;
        self
    }

    pub fn with_context_weight(mut self, w: f64) -> Self {
        self.context_weight = w;
        self
    }
}

impl fmt::Display for GenePredictorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GenePredictorConfig(min_orf={}, threshold={:.2}, coding_w={:.2}, ctx_w={:.2})",
            self.min_orf_length, self.score_threshold, self.coding_weight, self.context_weight,
        )
    }
}

/// Find start codons in a sequence.
pub fn find_start_codons(sequence: &str, config: &GenePredictorConfig) -> Vec<(usize, String)> {
    let seq = sequence.to_uppercase();
    let bytes = seq.as_bytes();
    let mut starts = Vec::new();
    if bytes.len() < 3 {
        return starts;
    }
    for i in 0..=(bytes.len() - 3) {
        let codon = &seq[i..i + 3];
        let is_start = match codon {
            "ATG" => true,
            "GTG" => config.allow_gtg_start,
            "TTG" => config.allow_ttg_start,
            _ => false,
        };
        if is_start {
            starts.push((i, codon.to_string()));
        }
    }
    starts
}

/// Find stop codons in a sequence.
pub fn find_stop_codons(sequence: &str) -> Vec<(usize, String)> {
    let seq = sequence.to_uppercase();
    let bytes = seq.as_bytes();
    let mut stops = Vec::new();
    if bytes.len() < 3 {
        return stops;
    }
    for i in 0..=(bytes.len() - 3) {
        let codon = &seq[i..i + 3];
        if codon == "TAA" || codon == "TAG" || codon == "TGA" {
            stops.push((i, codon.to_string()));
        }
    }
    stops
}

/// Predict genes on the forward strand of a DNA sequence.
pub fn predict_genes_forward(
    sequence: &str,
    config: &GenePredictorConfig,
    hexamer: &HexamerTable,
    context_scorer: &StartContextScorer,
) -> Vec<PredictedGene> {
    let seq = sequence.to_uppercase();
    let bytes = seq.as_bytes();
    let starts = find_start_codons(&seq, config);
    let stops = find_stop_codons(&seq);
    let mut predictions = Vec::new();
    let mut gene_id = 0usize;

    for (start_pos, start_codon) in &starts {
        let frame = (*start_pos % 3) as u8;
        // Find the nearest in-frame stop codon
        for (stop_pos, stop_codon) in &stops {
            if *stop_pos <= *start_pos {
                continue;
            }
            if (*stop_pos - *start_pos) % 3 != 0 {
                continue;
            }
            let orf_len = stop_pos + 3 - start_pos;
            if orf_len < config.min_orf_length {
                break;
            }
            let orf_seq = &seq[*start_pos..*stop_pos + 3];
            let coding_score = hexamer.score_sequence(orf_seq);
            let ctx_score = context_scorer.score(bytes, *start_pos);
            gene_id += 1;

            let gene = PredictedGene::new(*start_pos, *stop_pos + 2, PredStrand::Forward, frame)
                .with_id(&format!("gene_{:04}", gene_id))
                .with_start_codon(start_codon)
                .with_stop_codon(stop_codon)
                .with_coding_score(coding_score)
                .with_start_context_score(ctx_score);

            let combined = gene.combined_score(config.coding_weight, config.context_weight);
            if combined >= config.score_threshold {
                predictions.push(gene);
            }
            break; // take first in-frame stop
        }
    }
    predictions
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_codon_start() {
        assert_eq!(classify_codon("ATG"), CodonType::Start);
        assert_eq!(classify_codon("atg"), CodonType::Start);
        assert_eq!(classify_codon("GTG"), CodonType::Start);
        assert_eq!(classify_codon("TTG"), CodonType::Start);
    }

    #[test]
    fn test_classify_codon_stop() {
        assert_eq!(classify_codon("TAA"), CodonType::Stop);
        assert_eq!(classify_codon("TAG"), CodonType::Stop);
        assert_eq!(classify_codon("TGA"), CodonType::Stop);
    }

    #[test]
    fn test_classify_codon_sense() {
        assert_eq!(classify_codon("GCA"), CodonType::Sense);
        assert_eq!(classify_codon("AAA"), CodonType::Sense);
    }

    #[test]
    fn test_predicted_gene_codon_count() {
        let g = PredictedGene::new(0, 89, PredStrand::Forward, 0);
        assert_eq!(g.length_bp, 90);
        assert_eq!(g.codon_count(), 30);
    }

    #[test]
    fn test_predicted_gene_combined_score() {
        let g = PredictedGene::new(0, 99, PredStrand::Forward, 0)
            .with_coding_score(10.0)
            .with_start_context_score(5.0);
        let score = g.combined_score(0.7, 0.3);
        assert!((score - 8.5).abs() < 1e-9);
    }

    #[test]
    fn test_predicted_gene_display() {
        let g = PredictedGene::new(100, 500, PredStrand::Forward, 1)
            .with_id("gene_0001");
        let s = format!("{}", g);
        assert!(s.contains("gene_0001"));
        assert!(s.contains("+"));
    }

    #[test]
    fn test_find_start_codons_atg_only() {
        let config = GenePredictorConfig::new()
            .with_min_orf_length(3);
        let seq = "AATGCCATGTT";
        let starts = find_start_codons(seq, &config);
        // ATG at 1 and ATG at 6
        let positions: Vec<usize> = starts.iter().map(|(p, _)| *p).collect();
        assert!(positions.contains(&1));
        assert!(positions.contains(&6));
    }

    #[test]
    fn test_find_stop_codons() {
        let stops = find_stop_codons("ATGTAACCCTAG");
        let positions: Vec<usize> = stops.iter().map(|(p, _)| *p).collect();
        assert!(positions.contains(&3)); // TAA at 3
        assert!(positions.contains(&9)); // TAG at 9
    }

    #[test]
    fn test_hexamer_train_and_score() {
        let mut table = HexamerTable::new();
        table.train_coding("ATGATGATGATGATGATGATG");
        table.train_noncoding("TTTTTTTTTTTTTTTTTTTT");
        // A coding-like sequence should score positive
        let score = table.score_sequence("ATGATGATG");
        assert!(score > 0.0);
    }

    #[test]
    fn test_hexamer_display() {
        let table = HexamerTable::new();
        let s = format!("{}", table);
        assert!(s.contains("HexamerTable"));
    }

    #[test]
    fn test_start_context_scorer() {
        let scorer = StartContextScorer::new_default();
        // Construct a sequence with ATG at position 6
        let seq = b"AAAAAAAATGAAAA";
        let score = scorer.score(seq, 7);
        // Should produce a finite score
        assert!(score.is_finite());
    }

    #[test]
    fn test_start_context_scorer_too_short() {
        let scorer = StartContextScorer::new_default();
        let seq = b"ATG";
        let score = scorer.score(seq, 0);
        assert_eq!(score, 0.0); // too short for context window
    }

    #[test]
    fn test_gene_predictor_config_builders() {
        let config = GenePredictorConfig::new()
            .with_min_orf_length(150)
            .with_score_threshold(1.5)
            .with_coding_weight(0.8)
            .with_context_weight(0.2);
        assert_eq!(config.min_orf_length, 150);
        assert!((config.score_threshold - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_predict_genes_forward_basic() {
        // ATG...30 codons...TAA
        let seq = "AAAAAAA".to_string()
            + "ATG"
            + &"GCA".repeat(29)
            + "TAA"
            + "AAAAAAA";
        let config = GenePredictorConfig::new()
            .with_min_orf_length(90)
            .with_score_threshold(f64::NEG_INFINITY);
        let hexamer = HexamerTable::new();
        let ctx = StartContextScorer::new_default();
        let genes = predict_genes_forward(&seq, &config, &hexamer, &ctx);
        assert!(!genes.is_empty());
        assert_eq!(genes[0].start_codon, "ATG");
        assert_eq!(genes[0].stop_codon, "TAA");
    }

    #[test]
    fn test_predict_genes_forward_too_short() {
        let seq = "ATGGCATAA"; // only 9 bp
        let config = GenePredictorConfig::new().with_min_orf_length(90);
        let hexamer = HexamerTable::new();
        let ctx = StartContextScorer::new_default();
        let genes = predict_genes_forward(&seq, &config, &hexamer, &ctx);
        assert!(genes.is_empty());
    }

    #[test]
    fn test_codon_type_display() {
        assert_eq!(format!("{}", CodonType::Start), "start");
        assert_eq!(format!("{}", CodonType::Stop), "stop");
        assert_eq!(format!("{}", CodonType::Sense), "sense");
    }

    #[test]
    fn test_pred_strand_display() {
        assert_eq!(format!("{}", PredStrand::Forward), "+");
        assert_eq!(format!("{}", PredStrand::Reverse), "-");
    }

    #[test]
    fn test_gene_predictor_config_display() {
        let c = GenePredictorConfig::new();
        let s = format!("{}", c);
        assert!(s.contains("min_orf=90"));
    }
}
