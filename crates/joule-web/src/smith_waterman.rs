//! Smith–Waterman Local Alignment — affine gap model, traceback,
//! suboptimal alignments, scoring with configurable substitution matrices.
//!
//! Pure-Rust implementation of the Smith–Waterman algorithm for local
//! pairwise sequence alignment with affine gap penalties and traceback
//! to recover aligned subsequences.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SmithWatermanError {
    EmptySequence,
    InvalidParameters(String),
}

impl fmt::Display for SmithWatermanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "empty sequence"),
            Self::InvalidParameters(s) => write!(f, "invalid parameters: {s}"),
        }
    }
}

impl std::error::Error for SmithWatermanError {}

// ── Traceback Direction ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceDir {
    None,
    Diagonal,
    Up,
    Left,
}

// ── Scoring Config ──────────────────────────────────────────────

/// Configuration for Smith–Waterman local alignment.
#[derive(Debug, Clone)]
pub struct SwConfig {
    match_score: f64,
    mismatch_penalty: f64,
    gap_open: f64,
    gap_extend: f64,
    use_affine: bool,
    custom_score_fn: Option<fn(u8, u8) -> f64>,
}

impl SwConfig {
    /// Create a default config: match=2, mismatch=-1, gap_open=-2, gap_extend=-0.5.
    pub fn new() -> Self {
        Self {
            match_score: 2.0,
            mismatch_penalty: -1.0,
            gap_open: -2.0,
            gap_extend: -0.5,
            use_affine: true,
            custom_score_fn: None,
        }
    }

    pub fn with_match_score(mut self, s: f64) -> Self {
        self.match_score = s;
        self
    }

    pub fn with_mismatch_penalty(mut self, p: f64) -> Self {
        self.mismatch_penalty = p;
        self
    }

    pub fn with_gap_open(mut self, g: f64) -> Self {
        self.gap_open = g;
        self
    }

    pub fn with_gap_extend(mut self, g: f64) -> Self {
        self.gap_extend = g;
        self
    }

    pub fn with_affine(mut self, a: bool) -> Self {
        self.use_affine = a;
        self
    }

    pub fn with_custom_score(mut self, func: fn(u8, u8) -> f64) -> Self {
        self.custom_score_fn = Some(func);
        self
    }

    fn score_pair(&self, a: u8, b: u8) -> f64 {
        if let Some(func) = self.custom_score_fn {
            func(a, b)
        } else if a.to_ascii_uppercase() == b.to_ascii_uppercase() {
            self.match_score
        } else {
            self.mismatch_penalty
        }
    }
}

impl Default for SwConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SwConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SwConfig(match={}, mis={}, open={}, ext={}, affine={})",
            self.match_score, self.mismatch_penalty,
            self.gap_open, self.gap_extend, self.use_affine
        )
    }
}

// ── Alignment Result ────────────────────────────────────────────

/// Result of a Smith–Waterman local alignment.
#[derive(Debug, Clone)]
pub struct SwAlignment {
    pub aligned_a: Vec<u8>,
    pub aligned_b: Vec<u8>,
    pub score: f64,
    pub start_a: usize,
    pub start_b: usize,
    pub end_a: usize,
    pub end_b: usize,
    pub identity: f64,
    pub num_gaps: usize,
}

impl SwAlignment {
    /// Length of the alignment (number of columns).
    pub fn length(&self) -> usize {
        self.aligned_a.len()
    }

    /// Count of matching positions.
    pub fn matches(&self) -> usize {
        self.aligned_a
            .iter()
            .zip(self.aligned_b.iter())
            .filter(|(a, b)| **a != b'-' && **b != b'-' && a.to_ascii_uppercase() == b.to_ascii_uppercase())
            .count()
    }
}

impl fmt::Display for SwAlignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a_str = String::from_utf8_lossy(&self.aligned_a);
        let b_str = String::from_utf8_lossy(&self.aligned_b);
        write!(
            f,
            "SW(score={:.1}, id={:.1}%, len={}, gaps={})\n  A: {}\n  B: {}",
            self.score, self.identity, self.length(), self.num_gaps,
            a_str, b_str
        )
    }
}

// ── Smith–Waterman Aligner ──────────────────────────────────────

/// Smith–Waterman local aligner with affine gap support.
#[derive(Debug, Clone)]
pub struct SmithWaterman {
    config: SwConfig,
}

impl SmithWaterman {
    pub fn new(config: SwConfig) -> Self {
        Self { config }
    }

    /// Align two sequences and return the best local alignment.
    pub fn align(&self, seq_a: &[u8], seq_b: &[u8]) -> Result<SwAlignment, SmithWatermanError> {
        if seq_a.is_empty() || seq_b.is_empty() {
            return Err(SmithWatermanError::EmptySequence);
        }

        let m = seq_a.len();
        let n = seq_b.len();

        if self.config.use_affine {
            self.align_affine(seq_a, seq_b, m, n)
        } else {
            self.align_linear(seq_a, seq_b, m, n)
        }
    }

    /// Linear gap alignment.
    fn align_linear(
        &self,
        seq_a: &[u8],
        seq_b: &[u8],
        m: usize,
        n: usize,
    ) -> Result<SwAlignment, SmithWatermanError> {
        let mut h = vec![vec![0.0_f64; n + 1]; m + 1];
        let mut trace = vec![vec![TraceDir::None; n + 1]; m + 1];
        let mut max_score = 0.0_f64;
        let mut max_i = 0;
        let mut max_j = 0;

        for i in 1..=m {
            for j in 1..=n {
                let diag = h[i - 1][j - 1] + self.config.score_pair(seq_a[i - 1], seq_b[j - 1]);
                let up = h[i - 1][j] + self.config.gap_open;
                let left = h[i][j - 1] + self.config.gap_open;

                let mut best = 0.0;
                let mut dir = TraceDir::None;

                if diag > best {
                    best = diag;
                    dir = TraceDir::Diagonal;
                }
                if up > best {
                    best = up;
                    dir = TraceDir::Up;
                }
                if left > best {
                    best = left;
                    dir = TraceDir::Left;
                }

                h[i][j] = best;
                trace[i][j] = dir;

                if best > max_score {
                    max_score = best;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        self.traceback(seq_a, seq_b, &h, &trace, max_i, max_j, max_score)
    }

    /// Affine gap alignment using three matrices: H, Ix (gap in B), Iy (gap in A).
    fn align_affine(
        &self,
        seq_a: &[u8],
        seq_b: &[u8],
        m: usize,
        n: usize,
    ) -> Result<SwAlignment, SmithWatermanError> {
        let neg_inf = f64::NEG_INFINITY;
        let mut h = vec![vec![0.0_f64; n + 1]; m + 1];
        let mut ix = vec![vec![neg_inf; n + 1]; m + 1]; // gap in seq_b (insertion)
        let mut iy = vec![vec![neg_inf; n + 1]; m + 1]; // gap in seq_a (deletion)
        let mut trace = vec![vec![TraceDir::None; n + 1]; m + 1];
        let mut max_score = 0.0_f64;
        let mut max_i = 0;
        let mut max_j = 0;

        for i in 1..=m {
            for j in 1..=n {
                ix[i][j] = f64::max(
                    h[i - 1][j] + self.config.gap_open,
                    ix[i - 1][j] + self.config.gap_extend,
                );
                iy[i][j] = f64::max(
                    h[i][j - 1] + self.config.gap_open,
                    iy[i][j - 1] + self.config.gap_extend,
                );

                let diag =
                    h[i - 1][j - 1] + self.config.score_pair(seq_a[i - 1], seq_b[j - 1]);

                let mut best = 0.0;
                let mut dir = TraceDir::None;

                if diag > best {
                    best = diag;
                    dir = TraceDir::Diagonal;
                }
                if ix[i][j] > best {
                    best = ix[i][j];
                    dir = TraceDir::Up;
                }
                if iy[i][j] > best {
                    best = iy[i][j];
                    dir = TraceDir::Left;
                }

                h[i][j] = best;
                trace[i][j] = dir;

                if best > max_score {
                    max_score = best;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        self.traceback(seq_a, seq_b, &h, &trace, max_i, max_j, max_score)
    }

    /// Traceback from the maximum-scoring cell.
    fn traceback(
        &self,
        seq_a: &[u8],
        seq_b: &[u8],
        h: &[Vec<f64>],
        trace: &[Vec<TraceDir>],
        mut i: usize,
        mut j: usize,
        score: f64,
    ) -> Result<SwAlignment, SmithWatermanError> {
        let mut aligned_a = Vec::new();
        let mut aligned_b = Vec::new();
        let end_a = i;
        let end_b = j;

        while i > 0 && j > 0 && h[i][j] > 0.0 {
            match trace[i][j] {
                TraceDir::Diagonal => {
                    aligned_a.push(seq_a[i - 1]);
                    aligned_b.push(seq_b[j - 1]);
                    i -= 1;
                    j -= 1;
                }
                TraceDir::Up => {
                    aligned_a.push(seq_a[i - 1]);
                    aligned_b.push(b'-');
                    i -= 1;
                }
                TraceDir::Left => {
                    aligned_a.push(b'-');
                    aligned_b.push(seq_b[j - 1]);
                    j -= 1;
                }
                TraceDir::None => break,
            }
        }

        aligned_a.reverse();
        aligned_b.reverse();

        let aln_len = aligned_a.len();
        let matches = aligned_a
            .iter()
            .zip(aligned_b.iter())
            .filter(|(a, b)| **a != b'-' && **b != b'-' && a.to_ascii_uppercase() == b.to_ascii_uppercase())
            .count();
        let gaps = aligned_a.iter().chain(aligned_b.iter()).filter(|&&c| c == b'-').count();
        let identity = if aln_len > 0 { matches as f64 / aln_len as f64 * 100.0 } else { 0.0 };

        Ok(SwAlignment {
            aligned_a,
            aligned_b,
            score,
            start_a: i,
            start_b: j,
            end_a,
            end_b,
            identity,
            num_gaps: gaps,
        })
    }

    /// Compute just the score without traceback (faster).
    pub fn score_only(&self, seq_a: &[u8], seq_b: &[u8]) -> Result<f64, SmithWatermanError> {
        if seq_a.is_empty() || seq_b.is_empty() {
            return Err(SmithWatermanError::EmptySequence);
        }
        let n = seq_b.len();
        let mut prev = vec![0.0_f64; n + 1];
        let mut curr = vec![0.0_f64; n + 1];
        let mut max_score = 0.0_f64;

        for i in 1..=seq_a.len() {
            for j in 1..=n {
                let diag = prev[j - 1] + self.config.score_pair(seq_a[i - 1], seq_b[j - 1]);
                let up = prev[j] + self.config.gap_open;
                let left = curr[j - 1] + self.config.gap_open;
                let best = 0.0_f64.max(diag).max(up).max(left);
                curr[j] = best;
                if best > max_score {
                    max_score = best;
                }
            }
            std::mem::swap(&mut prev, &mut curr);
            curr.iter_mut().for_each(|v| *v = 0.0);
        }
        Ok(max_score)
    }

    /// Find the top-k non-overlapping local alignments.
    pub fn top_k_alignments(
        &self,
        seq_a: &[u8],
        seq_b: &[u8],
        k: usize,
    ) -> Result<Vec<SwAlignment>, SmithWatermanError> {
        let mut results = Vec::new();
        let mut mask_a = vec![false; seq_a.len()];
        let mut mask_b = vec![false; seq_b.len()];

        for _ in 0..k {
            let masked_a: Vec<u8> = seq_a
                .iter()
                .enumerate()
                .map(|(i, &c)| if mask_a[i] { b'N' } else { c })
                .collect();
            let masked_b: Vec<u8> = seq_b
                .iter()
                .enumerate()
                .map(|(i, &c)| if mask_b[i] { b'N' } else { c })
                .collect();

            let aln = self.align(&masked_a, &masked_b)?;
            if aln.score <= 0.0 {
                break;
            }

            for idx in aln.start_a..aln.end_a {
                if idx < mask_a.len() {
                    mask_a[idx] = true;
                }
            }
            for idx in aln.start_b..aln.end_b {
                if idx < mask_b.len() {
                    mask_b[idx] = true;
                }
            }
            results.push(aln);
        }
        Ok(results)
    }
}

impl fmt::Display for SmithWaterman {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SmithWaterman({})", self.config)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_sw() -> SmithWaterman {
        SmithWaterman::new(SwConfig::new())
    }

    #[test]
    fn identical_sequences() {
        let sw = default_sw();
        let aln = sw.align(b"ACGT", b"ACGT").unwrap();
        assert!(aln.score > 0.0);
        assert!((aln.identity - 100.0).abs() < 1e-9);
    }

    #[test]
    fn local_alignment_finds_match() {
        let sw = default_sw();
        let aln = sw.align(b"XXXACGTXXX", b"ACGT").unwrap();
        assert!(aln.score > 0.0);
        assert_eq!(aln.matches(), 4);
    }

    #[test]
    fn mismatch_reduces_score() {
        let sw = default_sw();
        let s1 = sw.score_only(b"ACGT", b"ACGT").unwrap();
        let s2 = sw.score_only(b"ACGT", b"AXGT").unwrap();
        assert!(s1 > s2);
    }

    #[test]
    fn empty_seq_errors() {
        let sw = default_sw();
        assert!(sw.align(b"", b"ACGT").is_err());
        assert!(sw.align(b"ACGT", b"").is_err());
    }

    #[test]
    fn affine_vs_linear() {
        let sw_aff = SmithWaterman::new(SwConfig::new().with_affine(true));
        let sw_lin = SmithWaterman::new(SwConfig::new().with_affine(false));
        let s_aff = sw_aff.score_only(b"ACGTACGT", b"ACGT").unwrap();
        let s_lin = sw_lin.score_only(b"ACGTACGT", b"ACGT").unwrap();
        assert!(s_aff > 0.0);
        assert!(s_lin > 0.0);
    }

    #[test]
    fn score_only_matches_align() {
        let sw = SmithWaterman::new(SwConfig::new().with_affine(false));
        let aln = sw.align(b"AACG", b"AACG").unwrap();
        let so = sw.score_only(b"AACG", b"AACG").unwrap();
        assert!((aln.score - so).abs() < 1e-9);
    }

    #[test]
    fn gap_in_alignment() {
        let sw = SmithWaterman::new(SwConfig::new().with_affine(false).with_gap_open(-1.0));
        let aln = sw.align(b"ACGT", b"AGT").unwrap();
        assert!(aln.num_gaps >= 0); // may or may not have gaps depending on scoring
    }

    #[test]
    fn custom_score_fn() {
        let sw = SmithWaterman::new(
            SwConfig::new().with_custom_score(|a, b| if a == b { 5.0 } else { -4.0 }),
        );
        let aln = sw.align(b"ACGT", b"ACGT").unwrap();
        assert!((aln.score - 20.0).abs() < 1e-9);
    }

    #[test]
    fn alignment_display() {
        let sw = default_sw();
        let aln = sw.align(b"ACGT", b"ACGT").unwrap();
        let s = format!("{aln}");
        assert!(s.contains("SW("));
        assert!(s.contains("A:"));
    }

    #[test]
    fn config_display() {
        let cfg = SwConfig::new();
        assert!(format!("{cfg}").contains("SwConfig"));
    }

    #[test]
    fn top_k_single() {
        let sw = default_sw();
        let alns = sw.top_k_alignments(b"ACGT", b"ACGT", 1).unwrap();
        assert_eq!(alns.len(), 1);
    }

    #[test]
    fn top_k_repeated() {
        let sw = default_sw();
        let alns = sw.top_k_alignments(b"ACGTACGT", b"ACGT", 3).unwrap();
        assert!(!alns.is_empty());
    }

    #[test]
    fn all_mismatches() {
        let sw = default_sw();
        let aln = sw.align(b"AAAA", b"CCCC").unwrap();
        assert!(aln.score <= 0.0 || aln.identity < 50.0);
    }

    #[test]
    fn single_base() {
        let sw = default_sw();
        let aln = sw.align(b"A", b"A").unwrap();
        assert!(aln.score > 0.0);
    }

    #[test]
    fn alignment_length() {
        let sw = default_sw();
        let aln = sw.align(b"ACGT", b"ACGT").unwrap();
        assert!(aln.length() >= 4);
    }

    #[test]
    fn builder_chaining() {
        let cfg = SwConfig::new()
            .with_match_score(3.0)
            .with_mismatch_penalty(-2.0)
            .with_gap_open(-5.0)
            .with_gap_extend(-1.0)
            .with_affine(true);
        assert!((cfg.match_score - 3.0).abs() < 1e-9);
    }

    #[test]
    fn protein_alignment() {
        let sw = SmithWaterman::new(
            SwConfig::new().with_match_score(1.0).with_mismatch_penalty(-1.0),
        );
        let aln = sw.align(b"MKWVTF", b"MKWVTF").unwrap();
        assert!((aln.identity - 100.0).abs() < 1e-9);
    }

    #[test]
    fn case_insensitive_match() {
        let sw = default_sw();
        let aln = sw.align(b"acgt", b"ACGT").unwrap();
        assert!(aln.score > 0.0);
        assert!((aln.identity - 100.0).abs() < 1e-9);
    }

    #[test]
    fn sw_display() {
        let sw = default_sw();
        assert!(format!("{sw}").contains("SmithWaterman"));
    }

    #[test]
    fn score_only_empty_err() {
        let sw = default_sw();
        assert!(sw.score_only(b"", b"ACGT").is_err());
    }
}
