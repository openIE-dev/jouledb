//! Needleman–Wunsch Global Alignment — linear and affine gap models,
//! optimal path traceback, alignment statistics, banded acceleration.
//!
//! Pure-Rust implementation of the Needleman–Wunsch algorithm for global
//! pairwise sequence alignment with configurable scoring and optional
//! band constraint for long sequences.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum NwError {
    EmptySequence,
    InvalidBandwidth(String),
    InvalidParameters(String),
}

impl fmt::Display for NwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "empty sequence"),
            Self::InvalidBandwidth(s) => write!(f, "invalid bandwidth: {s}"),
            Self::InvalidParameters(s) => write!(f, "invalid parameters: {s}"),
        }
    }
}

impl std::error::Error for NwError {}

// ── Traceback Direction ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceDir {
    None,
    Diagonal,
    Up,
    Left,
}

// ── Configuration ───────────────────────────────────────────────

/// Configuration for Needleman–Wunsch global alignment.
#[derive(Debug, Clone)]
pub struct NwConfig {
    match_score: f64,
    mismatch_penalty: f64,
    gap_open: f64,
    gap_extend: f64,
    use_affine: bool,
    bandwidth: Option<usize>,
    end_gap_free: bool,
    custom_score_fn: Option<fn(u8, u8) -> f64>,
}

impl NwConfig {
    pub fn new() -> Self {
        Self {
            match_score: 1.0,
            mismatch_penalty: -1.0,
            gap_open: -2.0,
            gap_extend: -0.5,
            use_affine: false,
            bandwidth: None,
            end_gap_free: false,
            custom_score_fn: None,
        }
    }

    pub fn with_match_score(mut self, s: f64) -> Self { self.match_score = s; self }
    pub fn with_mismatch_penalty(mut self, p: f64) -> Self { self.mismatch_penalty = p; self }
    pub fn with_gap_open(mut self, g: f64) -> Self { self.gap_open = g; self }
    pub fn with_gap_extend(mut self, g: f64) -> Self { self.gap_extend = g; self }
    pub fn with_affine(mut self, a: bool) -> Self { self.use_affine = a; self }
    pub fn with_bandwidth(mut self, b: usize) -> Self { self.bandwidth = Some(b); self }
    pub fn with_end_gap_free(mut self, e: bool) -> Self { self.end_gap_free = e; self }

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

impl Default for NwConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for NwConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NwConfig(match={}, mis={}, open={}, ext={}, affine={}, band={:?})",
            self.match_score, self.mismatch_penalty,
            self.gap_open, self.gap_extend, self.use_affine, self.bandwidth
        )
    }
}

// ── Alignment Result ────────────────────────────────────────────

/// Result of a Needleman–Wunsch global alignment.
#[derive(Debug, Clone)]
pub struct NwAlignment {
    pub aligned_a: Vec<u8>,
    pub aligned_b: Vec<u8>,
    pub score: f64,
    pub identity: f64,
    pub num_gaps: usize,
    pub num_mismatches: usize,
}

impl NwAlignment {
    /// Number of alignment columns.
    pub fn length(&self) -> usize {
        self.aligned_a.len()
    }

    /// Count matching positions.
    pub fn matches(&self) -> usize {
        self.aligned_a
            .iter()
            .zip(self.aligned_b.iter())
            .filter(|(a, b)| {
                **a != b'-' && **b != b'-'
                    && a.to_ascii_uppercase() == b.to_ascii_uppercase()
            })
            .count()
    }

    /// Build a midline string: '|' for match, ':' for similar, ' ' for mismatch/gap.
    pub fn midline(&self) -> Vec<u8> {
        self.aligned_a
            .iter()
            .zip(self.aligned_b.iter())
            .map(|(a, b)| {
                if *a == b'-' || *b == b'-' {
                    b' '
                } else if a.to_ascii_uppercase() == b.to_ascii_uppercase() {
                    b'|'
                } else {
                    b' '
                }
            })
            .collect()
    }
}

impl fmt::Display for NwAlignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a_str = String::from_utf8_lossy(&self.aligned_a);
        let binding = self.midline();
        let m_str = String::from_utf8_lossy(&binding);
        let b_str = String::from_utf8_lossy(&self.aligned_b);
        write!(
            f,
            "NW(score={:.1}, id={:.1}%, gaps={}, mis={})\n  {}\n  {}\n  {}",
            self.score, self.identity, self.num_gaps, self.num_mismatches,
            a_str, m_str, b_str
        )
    }
}

// ── Needleman–Wunsch Aligner ────────────────────────────────────

/// Global sequence aligner using the Needleman–Wunsch algorithm.
#[derive(Debug, Clone)]
pub struct NeedlemanWunsch {
    config: NwConfig,
}

impl NeedlemanWunsch {
    pub fn new(config: NwConfig) -> Self {
        Self { config }
    }

    /// Perform global alignment of two sequences.
    pub fn align(&self, seq_a: &[u8], seq_b: &[u8]) -> Result<NwAlignment, NwError> {
        if seq_a.is_empty() || seq_b.is_empty() {
            return Err(NwError::EmptySequence);
        }

        if self.config.use_affine {
            self.align_affine(seq_a, seq_b)
        } else {
            self.align_linear(seq_a, seq_b)
        }
    }

    /// Linear gap penalty alignment.
    fn align_linear(&self, seq_a: &[u8], seq_b: &[u8]) -> Result<NwAlignment, NwError> {
        let m = seq_a.len();
        let n = seq_b.len();
        let gap = self.config.gap_open;

        let mut h = vec![vec![0.0_f64; n + 1]; m + 1];
        let mut trace = vec![vec![TraceDir::None; n + 1]; m + 1];

        // Initialize borders.
        for i in 1..=m {
            h[i][0] = if self.config.end_gap_free { 0.0 } else { gap * i as f64 };
            trace[i][0] = TraceDir::Up;
        }
        for j in 1..=n {
            h[0][j] = if self.config.end_gap_free { 0.0 } else { gap * j as f64 };
            trace[0][j] = TraceDir::Left;
        }

        let band = self.config.bandwidth;

        for i in 1..=m {
            let j_lo = if let Some(bw) = band {
                if i > bw { i - bw } else { 1 }
            } else {
                1
            };
            let j_hi = if let Some(bw) = band {
                (i + bw).min(n)
            } else {
                n
            };

            for j in j_lo..=j_hi {
                let diag = h[i - 1][j - 1] + self.config.score_pair(seq_a[i - 1], seq_b[j - 1]);
                let up = h[i - 1][j] + gap;
                let left = h[i][j - 1] + gap;

                let (best, dir) = if diag >= up && diag >= left {
                    (diag, TraceDir::Diagonal)
                } else if up >= left {
                    (up, TraceDir::Up)
                } else {
                    (left, TraceDir::Left)
                };

                h[i][j] = best;
                trace[i][j] = dir;
            }
        }

        self.traceback_global(seq_a, seq_b, &trace, m, n, h[m][n])
    }

    /// Affine gap penalty alignment using three matrices.
    fn align_affine(&self, seq_a: &[u8], seq_b: &[u8]) -> Result<NwAlignment, NwError> {
        let m = seq_a.len();
        let n = seq_b.len();
        let neg_inf = f64::NEG_INFINITY;

        let mut h = vec![vec![0.0_f64; n + 1]; m + 1];
        let mut ix = vec![vec![neg_inf; n + 1]; m + 1];
        let mut iy = vec![vec![neg_inf; n + 1]; m + 1];
        let mut trace = vec![vec![TraceDir::None; n + 1]; m + 1];

        for i in 1..=m {
            h[i][0] = self.config.gap_open + self.config.gap_extend * (i as f64 - 1.0);
            trace[i][0] = TraceDir::Up;
        }
        for j in 1..=n {
            h[0][j] = self.config.gap_open + self.config.gap_extend * (j as f64 - 1.0);
            trace[0][j] = TraceDir::Left;
        }

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

                let (best, dir) = if diag >= ix[i][j] && diag >= iy[i][j] {
                    (diag, TraceDir::Diagonal)
                } else if ix[i][j] >= iy[i][j] {
                    (ix[i][j], TraceDir::Up)
                } else {
                    (iy[i][j], TraceDir::Left)
                };

                h[i][j] = best;
                trace[i][j] = dir;
            }
        }

        self.traceback_global(seq_a, seq_b, &trace, m, n, h[m][n])
    }

    /// Traceback for global alignment (start at [m][n]).
    fn traceback_global(
        &self,
        seq_a: &[u8],
        seq_b: &[u8],
        trace: &[Vec<TraceDir>],
        mut i: usize,
        mut j: usize,
        score: f64,
    ) -> Result<NwAlignment, NwError> {
        let mut al_a = Vec::new();
        let mut al_b = Vec::new();

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && trace[i][j] == TraceDir::Diagonal {
                al_a.push(seq_a[i - 1]);
                al_b.push(seq_b[j - 1]);
                i -= 1;
                j -= 1;
            } else if i > 0 && (j == 0 || trace[i][j] == TraceDir::Up) {
                al_a.push(seq_a[i - 1]);
                al_b.push(b'-');
                i -= 1;
            } else {
                al_a.push(b'-');
                al_b.push(seq_b[j - 1]);
                j -= 1;
            }
        }

        al_a.reverse();
        al_b.reverse();

        let aln_len = al_a.len();
        let matches = al_a
            .iter()
            .zip(al_b.iter())
            .filter(|(a, b)| {
                **a != b'-' && **b != b'-'
                    && a.to_ascii_uppercase() == b.to_ascii_uppercase()
            })
            .count();
        let gaps = al_a.iter().chain(al_b.iter()).filter(|&&c| c == b'-').count();
        let mismatches = al_a
            .iter()
            .zip(al_b.iter())
            .filter(|(a, b)| {
                **a != b'-' && **b != b'-'
                    && a.to_ascii_uppercase() != b.to_ascii_uppercase()
            })
            .count();
        let identity = if aln_len > 0 { matches as f64 / aln_len as f64 * 100.0 } else { 0.0 };

        Ok(NwAlignment {
            aligned_a: al_a,
            aligned_b: al_b,
            score,
            identity,
            num_gaps: gaps,
            num_mismatches: mismatches,
        })
    }

    /// Compute only the optimal score (O(n) space via two-row DP).
    pub fn score_only(&self, seq_a: &[u8], seq_b: &[u8]) -> Result<f64, NwError> {
        if seq_a.is_empty() || seq_b.is_empty() {
            return Err(NwError::EmptySequence);
        }
        let n = seq_b.len();
        let gap = self.config.gap_open;
        let mut prev = vec![0.0_f64; n + 1];
        let mut curr = vec![0.0_f64; n + 1];

        for j in 1..=n {
            prev[j] = gap * j as f64;
        }

        for i in 1..=seq_a.len() {
            curr[0] = gap * i as f64;
            for j in 1..=n {
                let diag = prev[j - 1] + self.config.score_pair(seq_a[i - 1], seq_b[j - 1]);
                let up = prev[j] + gap;
                let left = curr[j - 1] + gap;
                curr[j] = diag.max(up).max(left);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        Ok(prev[n])
    }

    /// Edit distance (Levenshtein): match=0, mismatch=-1, gap=-1.
    pub fn edit_distance(seq_a: &[u8], seq_b: &[u8]) -> usize {
        let m = seq_a.len();
        let n = seq_b.len();
        let mut prev = (0..=n).collect::<Vec<_>>();
        let mut curr = vec![0usize; n + 1];

        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = if seq_a[i - 1].to_ascii_uppercase()
                    == seq_b[j - 1].to_ascii_uppercase()
                {
                    0
                } else {
                    1
                };
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[n]
    }
}

impl fmt::Display for NeedlemanWunsch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NeedlemanWunsch({})", self.config)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nw() -> NeedlemanWunsch {
        NeedlemanWunsch::new(NwConfig::new())
    }

    #[test]
    fn identical_global() {
        let aln = nw().align(b"ACGT", b"ACGT").unwrap();
        assert!((aln.identity - 100.0).abs() < 1e-9);
        assert_eq!(aln.num_gaps, 0);
        assert_eq!(aln.num_mismatches, 0);
    }

    #[test]
    fn one_mismatch() {
        let aln = nw().align(b"ACGT", b"AXGT").unwrap();
        assert_eq!(aln.num_mismatches, 1);
    }

    #[test]
    fn insertion_gap() {
        let aln = nw().align(b"ACT", b"ACGT").unwrap();
        assert!(aln.num_gaps > 0 || aln.num_mismatches > 0);
    }

    #[test]
    fn empty_err() {
        assert!(nw().align(b"", b"ACGT").is_err());
    }

    #[test]
    fn score_only_matches_align() {
        let aligner = nw();
        let aln = aligner.align(b"ACGT", b"ACGT").unwrap();
        let so = aligner.score_only(b"ACGT", b"ACGT").unwrap();
        assert!((aln.score - so).abs() < 1e-9);
    }

    #[test]
    fn affine_alignment() {
        let aligner = NeedlemanWunsch::new(NwConfig::new().with_affine(true));
        let aln = aligner.align(b"ACGT", b"ACGT").unwrap();
        assert!((aln.identity - 100.0).abs() < 1e-9);
    }

    #[test]
    fn affine_gap_extension() {
        let aligner = NeedlemanWunsch::new(
            NwConfig::new()
                .with_affine(true)
                .with_gap_open(-5.0)
                .with_gap_extend(-1.0),
        );
        let aln = aligner.align(b"ACGT", b"ACXXXXGT").unwrap();
        assert!(aln.num_gaps > 0 || aln.num_mismatches > 0);
    }

    #[test]
    fn banded_alignment() {
        let aligner = NeedlemanWunsch::new(NwConfig::new().with_bandwidth(3));
        let aln = aligner.align(b"ACGT", b"ACGT").unwrap();
        assert!((aln.identity - 100.0).abs() < 1e-9);
    }

    #[test]
    fn edit_distance_identical() {
        assert_eq!(NeedlemanWunsch::edit_distance(b"ACGT", b"ACGT"), 0);
    }

    #[test]
    fn edit_distance_one() {
        assert_eq!(NeedlemanWunsch::edit_distance(b"ACGT", b"ACGA"), 1);
    }

    #[test]
    fn edit_distance_insertion() {
        assert_eq!(NeedlemanWunsch::edit_distance(b"ACT", b"ACGT"), 1);
    }

    #[test]
    fn midline_generation() {
        let aln = nw().align(b"ACGT", b"ACGT").unwrap();
        let ml = aln.midline();
        assert!(ml.iter().all(|c| *c == b'|'));
    }

    #[test]
    fn matches_count() {
        let aln = nw().align(b"ACGT", b"ACGT").unwrap();
        assert_eq!(aln.matches(), 4);
    }

    #[test]
    fn alignment_length_gte_input() {
        let aln = nw().align(b"ACT", b"ACGT").unwrap();
        assert!(aln.length() >= 3);
    }

    #[test]
    fn display_alignment() {
        let aln = nw().align(b"ACGT", b"ACGT").unwrap();
        let s = format!("{aln}");
        assert!(s.contains("NW("));
    }

    #[test]
    fn display_aligner() {
        assert!(format!("{}", nw()).contains("NeedlemanWunsch"));
    }

    #[test]
    fn custom_score_fn() {
        let aligner = NeedlemanWunsch::new(
            NwConfig::new().with_custom_score(|a, b| if a == b { 3.0 } else { -2.0 }),
        );
        let aln = aligner.align(b"AA", b"AA").unwrap();
        assert!((aln.score - 6.0).abs() < 1e-9);
    }

    #[test]
    fn config_builder_all() {
        let cfg = NwConfig::new()
            .with_match_score(2.0)
            .with_mismatch_penalty(-3.0)
            .with_gap_open(-5.0)
            .with_gap_extend(-1.0)
            .with_affine(true)
            .with_bandwidth(10)
            .with_end_gap_free(true);
        assert!(format!("{cfg}").contains("band=Some(10)"));
    }

    #[test]
    fn case_insensitive() {
        let aln = nw().align(b"acgt", b"ACGT").unwrap();
        assert!((aln.identity - 100.0).abs() < 1e-9);
    }

    #[test]
    fn score_only_empty_err() {
        assert!(nw().score_only(b"", b"ACGT").is_err());
    }
}
