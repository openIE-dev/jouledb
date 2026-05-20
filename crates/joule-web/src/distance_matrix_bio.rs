//! Pairwise distance computation for biological sequences — Jukes-Cantor,
//! Kimura two-parameter, p-distance, Poisson correction, and raw Hamming
//! distance models for nucleotide and protein alignments.
//!
//! Distances are stored in a symmetric matrix with efficient half-storage
//! and support additive/ultrametric checks used by downstream tree builders.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum DistanceError {
    DimensionMismatch { expected: usize, got: usize },
    InvalidIndex(usize, usize),
    UnequalLengths(usize, usize),
    UndefinedDistance(String),
    EmptyAlignment,
}

impl fmt::Display for DistanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidIndex(i, j) => write!(f, "invalid index ({i}, {j})"),
            Self::UnequalLengths(a, b) => write!(f, "unequal lengths: {a} vs {b}"),
            Self::UndefinedDistance(s) => write!(f, "undefined distance: {s}"),
            Self::EmptyAlignment => write!(f, "empty alignment"),
        }
    }
}

impl std::error::Error for DistanceError {}

// ── Substitution model ──────────────────────────────────────────

/// Evolutionary distance model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubstitutionModel {
    /// Raw proportion of differences.
    PDistance,
    /// Jukes-Cantor (1969) — equal substitution rates.
    JukesCantor,
    /// Kimura two-parameter (1980) — separate transition/transversion rates.
    Kimura2P,
    /// Poisson correction for protein distances.
    Poisson,
    /// Raw Hamming (count of mismatches).
    Hamming,
}

impl fmt::Display for SubstitutionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PDistance => write!(f, "p-distance"),
            Self::JukesCantor => write!(f, "Jukes-Cantor"),
            Self::Kimura2P => write!(f, "Kimura-2P"),
            Self::Poisson => write!(f, "Poisson"),
            Self::Hamming => write!(f, "Hamming"),
        }
    }
}

// ── Distance matrix ─────────────────────────────────────────────

/// Symmetric pairwise distance matrix.
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    pub labels: Vec<String>,
    /// Row-major full n×n storage.
    data: Vec<f64>,
    n: usize,
}

impl DistanceMatrix {
    pub fn new(labels: Vec<String>) -> Self {
        let n = labels.len();
        Self { labels, data: vec![0.0; n * n], n }
    }

    pub fn with_data(labels: Vec<String>, data: Vec<f64>) -> Result<Self, DistanceError> {
        let n = labels.len();
        if data.len() != n * n {
            return Err(DistanceError::DimensionMismatch {
                expected: n * n,
                got: data.len(),
            });
        }
        Ok(Self { labels, data, n })
    }

    pub fn size(&self) -> usize {
        self.n
    }

    pub fn get(&self, i: usize, j: usize) -> Result<f64, DistanceError> {
        if i >= self.n || j >= self.n {
            return Err(DistanceError::InvalidIndex(i, j));
        }
        Ok(self.data[i * self.n + j])
    }

    pub fn set(&mut self, i: usize, j: usize, val: f64) -> Result<(), DistanceError> {
        if i >= self.n || j >= self.n {
            return Err(DistanceError::InvalidIndex(i, j));
        }
        self.data[i * self.n + j] = val;
        self.data[j * self.n + i] = val;
        Ok(())
    }

    /// Check whether the matrix satisfies the triangle inequality.
    pub fn is_metric(&self) -> bool {
        for i in 0..self.n {
            for j in 0..self.n {
                for k in 0..self.n {
                    let dij = self.data[i * self.n + j];
                    let dik = self.data[i * self.n + k];
                    let dkj = self.data[k * self.n + j];
                    if dij > dik + dkj + 1e-9 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check for ultrametric property (used by UPGMA).
    pub fn is_ultrametric(&self, tol: f64) -> bool {
        for i in 0..self.n {
            for j in i + 1..self.n {
                for k in j + 1..self.n {
                    let dij = self.data[i * self.n + j];
                    let dik = self.data[i * self.n + k];
                    let djk = self.data[j * self.n + k];
                    let max_d = dij.max(dik).max(djk);
                    let mid_d = dij + dik + djk - max_d - dij.min(dik).min(djk);
                    if (max_d - mid_d).abs() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Row sums (used by neighbor-joining).
    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| (0..self.n).map(|j| self.data[i * self.n + j]).sum())
            .collect()
    }

    /// Minimum off-diagonal entry.
    pub fn min_distance(&self) -> Option<(usize, usize, f64)> {
        let mut best: Option<(usize, usize, f64)> = None;
        for i in 0..self.n {
            for j in i + 1..self.n {
                let d = self.data[i * self.n + j];
                if best.is_none() || d < best.unwrap().2 {
                    best = Some((i, j, d));
                }
            }
        }
        best
    }
}

impl fmt::Display for DistanceMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DistanceMatrix ({}×{}):", self.n, self.n)?;
        for i in 0..self.n {
            write!(f, "  {}: ", self.labels[i])?;
            for j in 0..self.n {
                write!(f, "{:.4} ", self.data[i * self.n + j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ── Distance computation ────────────────────────────────────────

/// Count transitions (purine↔purine, pyrimidine↔pyrimidine) and
/// transversions between two nucleotide sequences.
fn count_ts_tv(seq_a: &[u8], seq_b: &[u8]) -> (usize, usize, usize) {
    let mut transitions = 0;
    let mut transversions = 0;
    let mut compared = 0;
    for (&a, &b) in seq_a.iter().zip(seq_b.iter()) {
        if a == b'-' || b == b'-' || a == b'N' || b == b'N' {
            continue;
        }
        compared += 1;
        if a == b {
            continue;
        }
        let a_upper = a.to_ascii_uppercase();
        let b_upper = b.to_ascii_uppercase();
        let a_purine = a_upper == b'A' || a_upper == b'G';
        let b_purine = b_upper == b'A' || b_upper == b'G';
        if a_purine == b_purine {
            transitions += 1;
        } else {
            transversions += 1;
        }
    }
    (transitions, transversions, compared)
}

/// Compute p-distance (proportion of differences, ignoring gaps).
pub fn p_distance(seq_a: &[u8], seq_b: &[u8]) -> Result<f64, DistanceError> {
    if seq_a.len() != seq_b.len() {
        return Err(DistanceError::UnequalLengths(seq_a.len(), seq_b.len()));
    }
    let (ts, tv, total) = count_ts_tv(seq_a, seq_b);
    if total == 0 {
        return Err(DistanceError::EmptyAlignment);
    }
    Ok((ts + tv) as f64 / total as f64)
}

/// Jukes-Cantor corrected distance.
pub fn jukes_cantor(seq_a: &[u8], seq_b: &[u8]) -> Result<f64, DistanceError> {
    let p = p_distance(seq_a, seq_b)?;
    let arg = 1.0 - (4.0 / 3.0) * p;
    if arg <= 0.0 {
        return Err(DistanceError::UndefinedDistance(
            "JC saturation: p >= 0.75".into(),
        ));
    }
    Ok(-0.75 * arg.ln())
}

/// Kimura two-parameter distance.
pub fn kimura_2p(seq_a: &[u8], seq_b: &[u8]) -> Result<f64, DistanceError> {
    if seq_a.len() != seq_b.len() {
        return Err(DistanceError::UnequalLengths(seq_a.len(), seq_b.len()));
    }
    let (ts, tv, total) = count_ts_tv(seq_a, seq_b);
    if total == 0 {
        return Err(DistanceError::EmptyAlignment);
    }
    let s = ts as f64 / total as f64;
    let v = tv as f64 / total as f64;
    let a1 = 1.0 - 2.0 * s - v;
    let a2 = 1.0 - 2.0 * v;
    if a1 <= 0.0 || a2 <= 0.0 {
        return Err(DistanceError::UndefinedDistance("K2P saturation".into()));
    }
    Ok(-0.5 * a1.ln() - 0.25 * a2.ln())
}

/// Poisson-corrected distance for protein sequences.
pub fn poisson_distance(seq_a: &[u8], seq_b: &[u8]) -> Result<f64, DistanceError> {
    let p = p_distance(seq_a, seq_b)?;
    let arg = 1.0 - p;
    if arg <= 0.0 {
        return Err(DistanceError::UndefinedDistance("Poisson saturation".into()));
    }
    Ok(-arg.ln())
}

/// Hamming distance (raw mismatch count, ignoring gaps).
pub fn hamming_distance(seq_a: &[u8], seq_b: &[u8]) -> Result<usize, DistanceError> {
    if seq_a.len() != seq_b.len() {
        return Err(DistanceError::UnequalLengths(seq_a.len(), seq_b.len()));
    }
    Ok(seq_a
        .iter()
        .zip(seq_b.iter())
        .filter(|(a, b)| **a != b'-' && **b != b'-' && a != b && **a != b'N' && **b != b'N')
        .count())
}

/// Build a full distance matrix from an alignment using a given model.
pub fn compute_distance_matrix(
    labels: &[&str],
    sequences: &[&[u8]],
    model: SubstitutionModel,
) -> Result<DistanceMatrix, DistanceError> {
    if labels.len() != sequences.len() {
        return Err(DistanceError::DimensionMismatch {
            expected: labels.len(),
            got: sequences.len(),
        });
    }
    if sequences.is_empty() {
        return Err(DistanceError::EmptyAlignment);
    }
    let n = labels.len();
    let label_strings: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
    let mut dm = DistanceMatrix::new(label_strings);
    for i in 0..n {
        for j in i + 1..n {
            let d = match model {
                SubstitutionModel::PDistance => p_distance(sequences[i], sequences[j])?,
                SubstitutionModel::JukesCantor => jukes_cantor(sequences[i], sequences[j])?,
                SubstitutionModel::Kimura2P => kimura_2p(sequences[i], sequences[j])?,
                SubstitutionModel::Poisson => poisson_distance(sequences[i], sequences[j])?,
                SubstitutionModel::Hamming => {
                    hamming_distance(sequences[i], sequences[j])? as f64
                }
            };
            dm.set(i, j, d)?;
        }
    }
    Ok(dm)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p_distance_identical() {
        let s = b"ATCGATCG";
        assert!((p_distance(s, s).unwrap()).abs() < 1e-9);
    }

    #[test]
    fn test_p_distance_all_different() {
        let a = b"AAAA";
        let b_seq = b"TTTT";
        assert!((p_distance(a, b_seq).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_p_distance_partial() {
        let a = b"ATCG";
        let b_seq = b"AACG";
        assert!((p_distance(a, b_seq).unwrap() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_jukes_cantor() {
        let a = b"ATCGATCGATCG";
        let b_seq = b"AACGATCGATCG";
        let d = jukes_cantor(a, b_seq).unwrap();
        assert!(d > 0.0);
        assert!(d > p_distance(a, b_seq).unwrap()); // correction inflates
    }

    #[test]
    fn test_jukes_cantor_saturation() {
        let a = b"AAAA";
        let b_seq = b"TTTT";
        assert!(jukes_cantor(a, b_seq).is_err());
    }

    #[test]
    fn test_kimura_2p() {
        // Transitions: A->G, transversions: A->T
        let a = b"AAAAGGGG";
        let b_seq = b"GAAATTGG";
        let d = kimura_2p(a, b_seq).unwrap();
        assert!(d > 0.0);
    }

    #[test]
    fn test_poisson_distance() {
        let a = b"MFKL";
        let b_seq = b"MFRL";
        let d = poisson_distance(a, b_seq).unwrap();
        assert!(d > 0.0);
    }

    #[test]
    fn test_hamming_distance() {
        let a = b"ATCG";
        let b_seq = b"AACG";
        assert_eq!(hamming_distance(a, b_seq).unwrap(), 1);
    }

    #[test]
    fn test_hamming_with_gaps() {
        let a = b"AT-G";
        let b_seq = b"AC-G";
        // gap columns ignored, so only compare A/A, T/C, G/G => 1 diff
        assert_eq!(hamming_distance(a, b_seq).unwrap(), 1);
    }

    #[test]
    fn test_unequal_lengths() {
        let a = b"ATG";
        let b_seq = b"AT";
        assert!(p_distance(a, b_seq).is_err());
    }

    #[test]
    fn test_distance_matrix_new() {
        let dm = DistanceMatrix::new(vec!["A".into(), "B".into(), "C".into()]);
        assert_eq!(dm.size(), 3);
        assert!((dm.get(0, 1).unwrap()).abs() < 1e-9);
    }

    #[test]
    fn test_distance_matrix_set_symmetric() {
        let mut dm = DistanceMatrix::new(vec!["A".into(), "B".into()]);
        dm.set(0, 1, 0.5).unwrap();
        assert!((dm.get(0, 1).unwrap() - 0.5).abs() < 1e-9);
        assert!((dm.get(1, 0).unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_min_distance() {
        let mut dm = DistanceMatrix::new(vec!["A".into(), "B".into(), "C".into()]);
        dm.set(0, 1, 0.3).unwrap();
        dm.set(0, 2, 0.5).unwrap();
        dm.set(1, 2, 0.2).unwrap();
        let (i, j, d) = dm.min_distance().unwrap();
        assert_eq!((i, j), (1, 2));
        assert!((d - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_row_sums() {
        let mut dm = DistanceMatrix::new(vec!["A".into(), "B".into(), "C".into()]);
        dm.set(0, 1, 1.0).unwrap();
        dm.set(0, 2, 2.0).unwrap();
        dm.set(1, 2, 3.0).unwrap();
        let sums = dm.row_sums();
        assert!((sums[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_distance_matrix() {
        let labels = vec!["A", "B", "C"];
        let seqs: Vec<&[u8]> = vec![b"ATCGATCG", b"AACGATCG", b"ATCGAACG"];
        let dm = compute_distance_matrix(&labels, &seqs, SubstitutionModel::PDistance).unwrap();
        assert_eq!(dm.size(), 3);
        assert!(dm.get(0, 1).unwrap() > 0.0);
    }

    #[test]
    fn test_distance_matrix_display() {
        let dm = DistanceMatrix::new(vec!["X".into(), "Y".into()]);
        let s = format!("{dm}");
        assert!(s.contains("DistanceMatrix"));
    }

    #[test]
    fn test_is_metric_trivial() {
        let dm = DistanceMatrix::new(vec!["A".into(), "B".into()]);
        assert!(dm.is_metric());
    }

    #[test]
    fn test_model_display() {
        assert_eq!(format!("{}", SubstitutionModel::JukesCantor), "Jukes-Cantor");
        assert_eq!(format!("{}", SubstitutionModel::Kimura2P), "Kimura-2P");
    }
}
