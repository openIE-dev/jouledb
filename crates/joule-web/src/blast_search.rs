//! BLAST-Style Heuristic Search — seed-and-extend, word hits, high-scoring
//! segment pairs (HSPs), E-value estimation, database scanning.
//!
//! Pure-Rust heuristic local alignment engine modeled after the BLAST
//! algorithm: builds a word index, identifies seeds, extends hits, and
//! scores high-scoring segment pairs with statistical significance.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BlastError {
    EmptyQuery,
    EmptyDatabase,
    InvalidWordSize(usize),
    NoHits,
}

impl fmt::Display for BlastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuery => write!(f, "empty query sequence"),
            Self::EmptyDatabase => write!(f, "empty database"),
            Self::InvalidWordSize(w) => write!(f, "invalid word size: {w}"),
            Self::NoHits => write!(f, "no hits found"),
        }
    }
}

impl std::error::Error for BlastError {}

// ── HSP (High-Scoring Segment Pair) ─────────────────────────────

/// A high-scoring segment pair from the search.
#[derive(Debug, Clone)]
pub struct Hsp {
    pub query_start: usize,
    pub query_end: usize,
    pub subject_start: usize,
    pub subject_end: usize,
    pub score: f64,
    pub bit_score: f64,
    pub e_value: f64,
    pub identity: f64,
    pub aligned_query: Vec<u8>,
    pub aligned_subject: Vec<u8>,
    pub subject_id: usize,
}

impl Hsp {
    /// Length of the aligned region.
    pub fn alignment_length(&self) -> usize {
        self.aligned_query.len()
    }
}

impl fmt::Display for Hsp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HSP(score={:.1}, bits={:.1}, E={:.2e}, id={:.1}%, q={}..{}, s={}..{})",
            self.score, self.bit_score, self.e_value, self.identity,
            self.query_start, self.query_end,
            self.subject_start, self.subject_end
        )
    }
}

// ── Seed Hit ────────────────────────────────────────────────────

/// A word seed hit between query and subject.
#[derive(Debug, Clone, Copy)]
struct SeedHit {
    query_pos: usize,
    subject_pos: usize,
    subject_id: usize,
}

// ── Blast Configuration ─────────────────────────────────────────

/// Configuration for BLAST-style search.
#[derive(Debug, Clone)]
pub struct BlastConfig {
    word_size: usize,
    match_score: f64,
    mismatch_penalty: f64,
    gap_open: f64,
    gap_extend: f64,
    e_value_threshold: f64,
    max_hits: usize,
    extend_dropoff: f64,
    db_length_override: Option<usize>,
}

impl BlastConfig {
    pub fn new() -> Self {
        Self {
            word_size: 11,
            match_score: 2.0,
            mismatch_penalty: -3.0,
            gap_open: -5.0,
            gap_extend: -2.0,
            e_value_threshold: 10.0,
            max_hits: 500,
            extend_dropoff: 7.0,
            db_length_override: None,
        }
    }

    pub fn with_word_size(mut self, w: usize) -> Self { self.word_size = w; self }
    pub fn with_match_score(mut self, s: f64) -> Self { self.match_score = s; self }
    pub fn with_mismatch_penalty(mut self, p: f64) -> Self { self.mismatch_penalty = p; self }
    pub fn with_gap_open(mut self, g: f64) -> Self { self.gap_open = g; self }
    pub fn with_gap_extend(mut self, g: f64) -> Self { self.gap_extend = g; self }
    pub fn with_e_value_threshold(mut self, e: f64) -> Self { self.e_value_threshold = e; self }
    pub fn with_max_hits(mut self, m: usize) -> Self { self.max_hits = m; self }
    pub fn with_extend_dropoff(mut self, d: f64) -> Self { self.extend_dropoff = d; self }

    pub fn with_db_length(mut self, len: usize) -> Self {
        self.db_length_override = Some(len);
        self
    }

    fn score_pair(&self, a: u8, b: u8) -> f64 {
        if a.to_ascii_uppercase() == b.to_ascii_uppercase() {
            self.match_score
        } else {
            self.mismatch_penalty
        }
    }
}

impl Default for BlastConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for BlastConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlastConfig(word={}, match={}, mis={}, E_thresh={:.1e})",
            self.word_size, self.match_score, self.mismatch_penalty,
            self.e_value_threshold
        )
    }
}

// ── Database Subject ────────────────────────────────────────────

/// A subject sequence in the search database.
#[derive(Debug, Clone)]
pub struct Subject {
    pub id: usize,
    pub name: String,
    pub sequence: Vec<u8>,
}

impl Subject {
    pub fn new(id: usize, name: &str, sequence: &[u8]) -> Self {
        Self {
            id,
            name: name.to_string(),
            sequence: sequence.to_vec(),
        }
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Subject({}: {}, {} bp)", self.id, self.name, self.sequence.len())
    }
}

// ── Search Result ───────────────────────────────────────────────

/// Complete BLAST search result.
#[derive(Debug, Clone)]
pub struct BlastResult {
    pub query_length: usize,
    pub db_length: usize,
    pub num_subjects: usize,
    pub hsps: Vec<Hsp>,
    pub num_seeds_scanned: usize,
    pub num_extensions: usize,
}

impl BlastResult {
    /// Number of significant hits.
    pub fn num_hits(&self) -> usize {
        self.hsps.len()
    }

    /// Best HSP by score.
    pub fn best_hit(&self) -> Option<&Hsp> {
        self.hsps.first()
    }
}

impl fmt::Display for BlastResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlastResult({} hits, query={}bp, db={}bp, seeds={}, extensions={})",
            self.hsps.len(), self.query_length, self.db_length,
            self.num_seeds_scanned, self.num_extensions
        )
    }
}

// ── Word Index ──────────────────────────────────────────────────

/// Index of words (k-mers) from the query sequence.
struct WordIndex {
    index: HashMap<Vec<u8>, Vec<usize>>,
    word_size: usize,
}

impl WordIndex {
    fn build(query: &[u8], word_size: usize) -> Self {
        let mut index: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        if query.len() >= word_size {
            for i in 0..=query.len() - word_size {
                let word: Vec<u8> = query[i..i + word_size]
                    .iter()
                    .map(|c| c.to_ascii_uppercase())
                    .collect();
                index.entry(word).or_default().push(i);
            }
        }
        Self { index, word_size }
    }

    fn find_hits(&self, subject: &[u8], subject_id: usize) -> Vec<SeedHit> {
        let mut hits = Vec::new();
        if subject.len() < self.word_size {
            return hits;
        }
        for i in 0..=subject.len() - self.word_size {
            let word: Vec<u8> = subject[i..i + self.word_size]
                .iter()
                .map(|c| c.to_ascii_uppercase())
                .collect();
            if let Some(positions) = self.index.get(&word) {
                for &qpos in positions {
                    hits.push(SeedHit {
                        query_pos: qpos,
                        subject_pos: i,
                        subject_id,
                    });
                }
            }
        }
        hits
    }
}

// ── BLAST Searcher ──────────────────────────────────────────────

/// BLAST-style heuristic sequence searcher.
#[derive(Debug, Clone)]
pub struct BlastSearcher {
    config: BlastConfig,
}

impl BlastSearcher {
    pub fn new(config: BlastConfig) -> Self {
        Self { config }
    }

    /// Search a query against a database of subjects.
    pub fn search(
        &self,
        query: &[u8],
        subjects: &[Subject],
    ) -> Result<BlastResult, BlastError> {
        if query.is_empty() {
            return Err(BlastError::EmptyQuery);
        }
        if subjects.is_empty() {
            return Err(BlastError::EmptyDatabase);
        }
        if self.config.word_size == 0 || self.config.word_size > query.len() {
            return Err(BlastError::InvalidWordSize(self.config.word_size));
        }

        let db_length: usize = self.config.db_length_override.unwrap_or_else(|| {
            subjects.iter().map(|s| s.sequence.len()).sum()
        });

        let word_idx = WordIndex::build(query, self.config.word_size);

        let mut all_hsps = Vec::new();
        let mut total_seeds = 0usize;
        let mut total_extensions = 0usize;

        for subj in subjects {
            let seed_hits = word_idx.find_hits(&subj.sequence, subj.id);
            total_seeds += seed_hits.len();

            for hit in &seed_hits {
                let (score, q_start, q_end, s_start, s_end) =
                    self.ungapped_extend(query, &subj.sequence, hit);
                total_extensions += 1;

                if score <= 0.0 {
                    continue;
                }

                let bit_score = self.raw_to_bit_score(score);
                let e_value = self.compute_e_value(bit_score, query.len(), db_length);

                if e_value > self.config.e_value_threshold {
                    continue;
                }

                let (aligned_q, aligned_s) =
                    self.build_aligned_pair(query, &subj.sequence, q_start, q_end, s_start, s_end);
                let identity = self.compute_identity(&aligned_q, &aligned_s);

                all_hsps.push(Hsp {
                    query_start: q_start,
                    query_end: q_end,
                    subject_start: s_start,
                    subject_end: s_end,
                    score,
                    bit_score,
                    e_value,
                    identity,
                    aligned_query: aligned_q,
                    aligned_subject: aligned_s,
                    subject_id: subj.id,
                });
            }
        }

        // Sort by score descending, deduplicate overlapping HSPs.
        all_hsps.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let hsps = self.remove_redundant_hsps(all_hsps);

        Ok(BlastResult {
            query_length: query.len(),
            db_length,
            num_subjects: subjects.len(),
            hsps,
            num_seeds_scanned: total_seeds,
            num_extensions: total_extensions,
        })
    }

    /// Ungapped extension from a seed hit.
    fn ungapped_extend(
        &self,
        query: &[u8],
        subject: &[u8],
        hit: &SeedHit,
    ) -> (f64, usize, usize, usize, usize) {
        let mut score = 0.0_f64;
        let mut max_score = 0.0_f64;

        // Score the seed itself.
        for k in 0..self.config.word_size {
            score += self.config.score_pair(
                query[hit.query_pos + k],
                subject[hit.subject_pos + k],
            );
        }
        max_score = score;

        // Extend right.
        let mut r_ext = 0usize;
        let mut best_r = 0usize;
        let mut r_score = score;
        let q_right_start = hit.query_pos + self.config.word_size;
        let s_right_start = hit.subject_pos + self.config.word_size;
        while q_right_start + r_ext < query.len() && s_right_start + r_ext < subject.len() {
            r_score += self.config.score_pair(
                query[q_right_start + r_ext],
                subject[s_right_start + r_ext],
            );
            r_ext += 1;
            if r_score > max_score {
                max_score = r_score;
                best_r = r_ext;
            }
            if max_score - r_score > self.config.extend_dropoff {
                break;
            }
        }

        // Extend left.
        let mut l_ext = 0usize;
        let mut best_l = 0usize;
        let mut l_score = max_score;
        let mut l_max = max_score;
        while hit.query_pos > l_ext && hit.subject_pos > l_ext {
            l_ext += 1;
            l_score += self.config.score_pair(
                query[hit.query_pos - l_ext],
                subject[hit.subject_pos - l_ext],
            );
            if l_score > l_max {
                l_max = l_score;
                best_l = l_ext;
            }
            if l_max - l_score > self.config.extend_dropoff {
                break;
            }
        }

        let final_score = l_max;
        let q_start = hit.query_pos - best_l;
        let q_end = q_right_start + best_r;
        let s_start = hit.subject_pos - best_l;
        let s_end = s_right_start + best_r;

        (final_score, q_start, q_end, s_start, s_end)
    }

    /// Convert raw score to bit score using Karlin–Altschul parameters.
    fn raw_to_bit_score(&self, raw: f64) -> f64 {
        let lambda = 1.28;
        let ln_k = 0.3;
        (lambda * raw - ln_k) / std::f64::consts::LN_2
    }

    /// Compute E-value from bit score.
    fn compute_e_value(&self, bit_score: f64, query_len: usize, db_len: usize) -> f64 {
        let m = query_len as f64;
        let n = db_len as f64;
        m * n * 2.0_f64.powf(-bit_score)
    }

    /// Build aligned pair from coordinate ranges.
    fn build_aligned_pair(
        &self,
        query: &[u8],
        subject: &[u8],
        q_start: usize,
        q_end: usize,
        s_start: usize,
        s_end: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let aq = query[q_start..q_end].to_vec();
        let as_ = subject[s_start..s_end].to_vec();
        (aq, as_)
    }

    /// Compute identity percentage from aligned pair.
    fn compute_identity(&self, aligned_q: &[u8], aligned_s: &[u8]) -> f64 {
        if aligned_q.is_empty() {
            return 0.0;
        }
        let matches = aligned_q
            .iter()
            .zip(aligned_s.iter())
            .filter(|(a, b)| a.to_ascii_uppercase() == b.to_ascii_uppercase())
            .count();
        matches as f64 / aligned_q.len() as f64 * 100.0
    }

    /// Remove HSPs that substantially overlap a higher-scoring HSP.
    fn remove_redundant_hsps(&self, hsps: Vec<Hsp>) -> Vec<Hsp> {
        let mut kept: Vec<Hsp> = Vec::new();
        for hsp in hsps {
            let dominated = kept.iter().any(|k| {
                k.subject_id == hsp.subject_id
                    && Self::overlap_frac(k.query_start, k.query_end, hsp.query_start, hsp.query_end)
                        > 0.5
            });
            if !dominated && kept.len() < self.config.max_hits {
                kept.push(hsp);
            }
        }
        kept
    }

    fn overlap_frac(s1: usize, e1: usize, s2: usize, e2: usize) -> f64 {
        let start = s1.max(s2);
        let end = e1.min(e2);
        if start >= end {
            return 0.0;
        }
        let overlap = (end - start) as f64;
        let len2 = (e2 - s2) as f64;
        if len2 == 0.0 { 0.0 } else { overlap / len2 }
    }
}

impl fmt::Display for BlastSearcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlastSearcher({})", self.config)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db(seqs: &[&[u8]]) -> Vec<Subject> {
        seqs.iter()
            .enumerate()
            .map(|(i, s)| Subject::new(i, &format!("seq_{i}"), s))
            .collect()
    }

    fn default_blast() -> BlastSearcher {
        BlastSearcher::new(BlastConfig::new().with_word_size(3))
    }

    #[test]
    fn exact_match_hit() {
        let bs = default_blast();
        let query = b"ACGTACGTACGT";
        let db = make_db(&[b"ACGTACGTACGT"]);
        let result = bs.search(query, &db).unwrap();
        assert!(!result.hsps.is_empty());
        assert!(result.hsps[0].identity > 99.0);
    }

    #[test]
    fn no_match_high_threshold() {
        let bs = BlastSearcher::new(
            BlastConfig::new()
                .with_word_size(3)
                .with_e_value_threshold(1e-100),
        );
        let result = bs.search(b"AAAAAA", &make_db(&[b"CCCCCC"])).unwrap();
        assert!(result.hsps.is_empty());
    }

    #[test]
    fn empty_query_err() {
        let bs = default_blast();
        assert!(bs.search(b"", &make_db(&[b"ACGT"])).is_err());
    }

    #[test]
    fn empty_db_err() {
        let bs = default_blast();
        assert!(bs.search(b"ACGT", &[]).is_err());
    }

    #[test]
    fn invalid_word_size() {
        let bs = BlastSearcher::new(BlastConfig::new().with_word_size(0));
        assert!(bs.search(b"ACGT", &make_db(&[b"ACGT"])).is_err());
    }

    #[test]
    fn multiple_subjects() {
        let bs = default_blast();
        let db = make_db(&[b"ACGTACGTACGT", b"TTTTTTTTTTTT", b"ACGTACGTACGT"]);
        let result = bs.search(b"ACGTACGTACGT", &db).unwrap();
        assert!(result.num_subjects == 3);
    }

    #[test]
    fn bit_score_positive() {
        let bs = default_blast();
        let result = bs.search(b"ACGTACGT", &make_db(&[b"ACGTACGT"])).unwrap();
        if let Some(h) = result.best_hit() {
            assert!(h.bit_score > 0.0);
        }
    }

    #[test]
    fn e_value_reasonable() {
        let bs = default_blast();
        let result = bs.search(b"ACGTACGT", &make_db(&[b"ACGTACGT"])).unwrap();
        if let Some(h) = result.best_hit() {
            assert!(h.e_value >= 0.0);
        }
    }

    #[test]
    fn hsp_display() {
        let bs = default_blast();
        let result = bs.search(b"ACGTACGT", &make_db(&[b"ACGTACGT"])).unwrap();
        if let Some(h) = result.best_hit() {
            let s = format!("{h}");
            assert!(s.contains("HSP("));
        }
    }

    #[test]
    fn result_display() {
        let bs = default_blast();
        let result = bs.search(b"ACGTACGT", &make_db(&[b"ACGTACGT"])).unwrap();
        assert!(format!("{result}").contains("BlastResult("));
    }

    #[test]
    fn subject_display() {
        let s = Subject::new(0, "test", b"ACGT");
        assert!(format!("{s}").contains("test"));
    }

    #[test]
    fn config_builder() {
        let cfg = BlastConfig::new()
            .with_word_size(7)
            .with_match_score(1.0)
            .with_mismatch_penalty(-2.0)
            .with_gap_open(-5.0)
            .with_gap_extend(-2.0)
            .with_e_value_threshold(0.001)
            .with_max_hits(100)
            .with_extend_dropoff(10.0)
            .with_db_length(1_000_000);
        assert!(format!("{cfg}").contains("word=7"));
    }

    #[test]
    fn overlap_frac_full() {
        assert!((BlastSearcher::overlap_frac(0, 10, 0, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn overlap_frac_none() {
        assert!((BlastSearcher::overlap_frac(0, 5, 10, 15)).abs() < 1e-9);
    }

    #[test]
    fn overlap_frac_partial() {
        let frac = BlastSearcher::overlap_frac(0, 10, 5, 15);
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn alignment_length_matches() {
        let bs = default_blast();
        let result = bs.search(b"ACGTACGT", &make_db(&[b"ACGTACGT"])).unwrap();
        if let Some(h) = result.best_hit() {
            assert!(h.alignment_length() > 0);
        }
    }

    #[test]
    fn searcher_display() {
        let bs = default_blast();
        assert!(format!("{bs}").contains("BlastSearcher"));
    }

    #[test]
    fn word_size_larger_than_query() {
        let bs = BlastSearcher::new(BlastConfig::new().with_word_size(20));
        assert!(bs.search(b"ACGT", &make_db(&[b"ACGT"])).is_err());
    }

    #[test]
    fn seeds_scanned_count() {
        let bs = default_blast();
        let result = bs.search(b"ACGTACGT", &make_db(&[b"ACGTACGT"])).unwrap();
        assert!(result.num_seeds_scanned > 0);
    }

    #[test]
    fn num_extensions_positive() {
        let bs = default_blast();
        let result = bs.search(b"ACGTACGT", &make_db(&[b"ACGTACGT"])).unwrap();
        assert!(result.num_extensions > 0);
    }
}
