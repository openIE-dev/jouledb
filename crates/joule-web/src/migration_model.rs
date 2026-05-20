//! Migration models for population genetics.
//!
//! Implements the island model, stepping-stone model, gene flow
//! estimation from Fst, isolation-with-migration (IM) framework,
//! and migration matrix analysis. All computations are std-only
//! with f64 math.

use std::fmt;

// ── Migration Rate ──────────────────────────────────────────────────

/// Pairwise migration rate between two populations.
#[derive(Debug, Clone, PartialEq)]
pub struct MigrationRate {
    pub from_pop: usize,
    pub to_pop: usize,
    pub rate: f64,
}

impl MigrationRate {
    pub fn new(from: usize, to: usize, rate: f64) -> Self {
        Self { from_pop: from, to_pop: to, rate: rate.max(0.0) }
    }

    /// Effective number of migrants per generation: Nm = rate * Ne.
    pub fn nm(&self, ne: f64) -> f64 {
        self.rate * ne
    }
}

impl fmt::Display for MigrationRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m({}->{})={:.6}", self.from_pop, self.to_pop, self.rate)
    }
}

// ── Migration Matrix ────────────────────────────────────────────────

/// Full migration matrix for k populations.
///
/// Entry `matrix[i][j]` = fraction of population `i` that are migrants
/// from population `j` each generation (backward in time).
#[derive(Debug, Clone, PartialEq)]
pub struct MigrationMatrix {
    pub matrix: Vec<Vec<f64>>,
    pub n_pops: usize,
}

impl MigrationMatrix {
    /// Create an empty (no migration) matrix.
    pub fn new(n_pops: usize) -> Self {
        let matrix = vec![vec![0.0; n_pops]; n_pops];
        Self { matrix, n_pops }
    }

    /// Set migration rate from pop `from` to pop `to`.
    pub fn with_rate(mut self, from: usize, to: usize, rate: f64) -> Self {
        if from < self.n_pops && to < self.n_pops {
            self.matrix[to][from] = rate.max(0.0);
        }
        self
    }

    /// Set symmetric migration between two populations.
    pub fn with_symmetric_rate(mut self, pop1: usize, pop2: usize, rate: f64) -> Self {
        let r = rate.max(0.0);
        if pop1 < self.n_pops && pop2 < self.n_pops {
            self.matrix[pop1][pop2] = r;
            self.matrix[pop2][pop1] = r;
        }
        self
    }

    /// Total immigration rate into population `i`.
    pub fn total_immigration(&self, pop: usize) -> f64 {
        if pop >= self.n_pops {
            return 0.0;
        }
        self.matrix[pop].iter().enumerate()
            .filter(|&(j, _)| j != pop)
            .map(|(_, &m)| m)
            .sum()
    }

    /// Total emigration rate from population `i`.
    pub fn total_emigration(&self, pop: usize) -> f64 {
        if pop >= self.n_pops {
            return 0.0;
        }
        self.matrix.iter().enumerate()
            .filter(|&(j, _)| j != pop)
            .map(|(_, row)| row[pop])
            .sum()
    }

    /// Check if matrix is conservative (rows sum to <=1).
    pub fn is_conservative(&self) -> bool {
        for i in 0..self.n_pops {
            let total = self.total_immigration(i);
            if total > 1.0 + 1e-10 {
                return false;
            }
        }
        true
    }

    /// All pairwise migration rates as a flat list.
    pub fn pairwise_rates(&self) -> Vec<MigrationRate> {
        let mut rates = Vec::new();
        for i in 0..self.n_pops {
            for j in 0..self.n_pops {
                if i != j && self.matrix[i][j] > 0.0 {
                    rates.push(MigrationRate::new(j, i, self.matrix[i][j]));
                }
            }
        }
        rates
    }

    /// Eigenvalue-based connectivity: spectral gap of the migration matrix.
    pub fn spectral_gap(&self) -> f64 {
        if self.n_pops < 2 {
            return 0.0;
        }
        // Power iteration for second-largest eigenvalue
        let transition = self.transition_matrix();
        let lambda2 = second_eigenvalue(&transition);
        1.0 - lambda2.abs()
    }

    /// Convert to a proper transition matrix (rows sum to 1).
    fn transition_matrix(&self) -> Vec<Vec<f64>> {
        let mut trans = vec![vec![0.0; self.n_pops]; self.n_pops];
        for i in 0..self.n_pops {
            let total_mig = self.total_immigration(i);
            for j in 0..self.n_pops {
                if i == j {
                    trans[i][j] = 1.0 - total_mig;
                } else {
                    trans[i][j] = self.matrix[i][j];
                }
            }
        }
        trans
    }
}

impl fmt::Display for MigrationMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MigrationMatrix({}x{}, gap={:.4})", self.n_pops, self.n_pops, self.spectral_gap())
    }
}

// ── Island Model ────────────────────────────────────────────────────

/// The Wright-Fisher island model with symmetric migration.
#[derive(Debug, Clone, PartialEq)]
pub struct IslandModel {
    pub n_pops: usize,
    pub ne_per_pop: f64,
    pub migration_rate: f64,
}

impl IslandModel {
    pub fn new(n_pops: usize, ne: f64, migration_rate: f64) -> Self {
        Self {
            n_pops,
            ne_per_pop: ne.max(1.0),
            migration_rate: migration_rate.max(0.0),
        }
    }

    pub fn with_n_pops(mut self, n: usize) -> Self {
        self.n_pops = n.max(2);
        self
    }

    pub fn with_ne(mut self, ne: f64) -> Self {
        self.ne_per_pop = ne.max(1.0);
        self
    }

    pub fn with_migration_rate(mut self, m: f64) -> Self {
        self.migration_rate = m.max(0.0);
        self
    }

    /// Expected Fst under the island model: Fst ≈ 1 / (1 + 4*Ne*m).
    pub fn expected_fst(&self) -> f64 {
        let nm = self.ne_per_pop * self.migration_rate;
        1.0 / (1.0 + 4.0 * nm)
    }

    /// Number of migrants per generation: Nm.
    pub fn nm(&self) -> f64 {
        self.ne_per_pop * self.migration_rate
    }

    /// Total effective population size: Ne_total ≈ n * Ne.
    pub fn total_ne(&self) -> f64 {
        self.n_pops as f64 * self.ne_per_pop
    }

    /// Generate the migration matrix for this island model.
    pub fn to_migration_matrix(&self) -> MigrationMatrix {
        let m_per_pair = self.migration_rate / (self.n_pops - 1).max(1) as f64;
        let mut mat = MigrationMatrix::new(self.n_pops);
        for i in 0..self.n_pops {
            for j in 0..self.n_pops {
                if i != j {
                    mat.matrix[i][j] = m_per_pair;
                }
            }
        }
        mat
    }

    /// Expected coalescence time for two lineages in different populations.
    pub fn expected_between_coalescence(&self) -> f64 {
        let ne = self.ne_per_pop;
        let m = self.migration_rate;
        let k = self.n_pops as f64;
        // E[T_between] ≈ 2*Ne + k / (2*m*(k-1))
        2.0 * ne + k / (2.0 * m * (k - 1.0).max(1.0))
    }

    /// Expected coalescence time for two lineages in the same population.
    pub fn expected_within_coalescence(&self) -> f64 {
        2.0 * self.ne_per_pop
    }
}

impl fmt::Display for IslandModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IslandModel(k={}, Ne={:.0}, m={:.6}, Fst={:.4})",
            self.n_pops, self.ne_per_pop, self.migration_rate, self.expected_fst()
        )
    }
}

// ── Stepping Stone Model ────────────────────────────────────────────

/// Linear stepping-stone model: populations arranged in a line,
/// migration only between adjacent populations.
#[derive(Debug, Clone, PartialEq)]
pub struct SteppingStoneModel {
    pub n_pops: usize,
    pub ne_per_pop: f64,
    pub adjacent_rate: f64,
    pub circular: bool,
}

impl SteppingStoneModel {
    pub fn new(n_pops: usize, ne: f64, adjacent_rate: f64) -> Self {
        Self {
            n_pops,
            ne_per_pop: ne.max(1.0),
            adjacent_rate: adjacent_rate.max(0.0),
            circular: false,
        }
    }

    pub fn with_circular(mut self, circ: bool) -> Self {
        self.circular = circ;
        self
    }

    pub fn with_ne(mut self, ne: f64) -> Self {
        self.ne_per_pop = ne.max(1.0);
        self
    }

    pub fn with_adjacent_rate(mut self, m: f64) -> Self {
        self.adjacent_rate = m.max(0.0);
        self
    }

    /// Generate the migration matrix.
    pub fn to_migration_matrix(&self) -> MigrationMatrix {
        let mut mat = MigrationMatrix::new(self.n_pops);
        for i in 0..self.n_pops {
            if i > 0 {
                mat.matrix[i][i - 1] = self.adjacent_rate;
            }
            if i + 1 < self.n_pops {
                mat.matrix[i][i + 1] = self.adjacent_rate;
            }
        }
        if self.circular && self.n_pops > 2 {
            mat.matrix[0][self.n_pops - 1] = self.adjacent_rate;
            mat.matrix[self.n_pops - 1][0] = self.adjacent_rate;
        }
        mat
    }

    /// Expected Fst between populations separated by `d` steps.
    pub fn expected_fst_by_distance(&self, distance: usize) -> f64 {
        let nm = self.ne_per_pop * self.adjacent_rate;
        if nm <= 0.0 {
            return 1.0;
        }
        // Isolation by distance: Fst ≈ d / (4*Ne*m + d) for 1D stepping stone
        let d = distance as f64;
        d / (4.0 * nm + d)
    }

    /// Pairwise Fst matrix for all population pairs.
    pub fn fst_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_pops;
        let mut mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = if self.circular {
                    let d1 = j - i;
                    let d2 = n - d1;
                    d1.min(d2)
                } else {
                    j - i
                };
                let fst = self.expected_fst_by_distance(dist);
                mat[i][j] = fst;
                mat[j][i] = fst;
            }
        }
        mat
    }
}

impl fmt::Display for SteppingStoneModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let topology = if self.circular { "circular" } else { "linear" };
        write!(
            f,
            "SteppingStone({}, k={}, Ne={:.0}, m={:.6})",
            topology, self.n_pops, self.ne_per_pop, self.adjacent_rate
        )
    }
}

// ── Gene Flow Estimation ────────────────────────────────────────────

/// Estimate Nm (gene flow) from observed Fst.
pub fn nm_from_fst(fst: f64) -> f64 {
    if fst <= 0.0 || fst >= 1.0 {
        return 0.0;
    }
    (1.0 - fst) / (4.0 * fst)
}

/// Estimate Nm for k populations (Takahata & Nei 1984).
pub fn nm_from_fst_k_pops(fst: f64, k: usize) -> f64 {
    if fst <= 0.0 || fst >= 1.0 || k < 2 {
        return 0.0;
    }
    let kf = k as f64;
    (kf * (1.0 - fst)) / (4.0 * (kf - 1.0) * fst)
}

/// Estimate migration rate m from Nm and Ne.
pub fn migration_rate_from_nm(nm: f64, ne: f64) -> f64 {
    if ne <= 0.0 { 0.0 } else { nm / ne }
}

/// Private alleles method (Barton & Slatkin 1986):
/// ln(Nm) = a - b * ln(mean_private_freq).
pub fn nm_from_private_alleles(mean_private_freq: f64) -> f64 {
    if mean_private_freq <= 0.0 || mean_private_freq >= 1.0 {
        return 0.0;
    }
    let ln_nm = 0.505 - 2.44 * mean_private_freq.ln();
    ln_nm.exp()
}

// ── Isolation with Migration ────────────────────────────────────────

/// Parameters for a two-population isolation-with-migration model.
#[derive(Debug, Clone, PartialEq)]
pub struct ImParams {
    pub ne_pop1: f64,
    pub ne_pop2: f64,
    pub ne_ancestral: f64,
    pub migration_1to2: f64,
    pub migration_2to1: f64,
    pub divergence_time: f64,
}

impl ImParams {
    pub fn new() -> Self {
        Self {
            ne_pop1: 10_000.0,
            ne_pop2: 10_000.0,
            ne_ancestral: 10_000.0,
            migration_1to2: 0.0,
            migration_2to1: 0.0,
            divergence_time: 0.0,
        }
    }

    pub fn with_ne1(mut self, ne: f64) -> Self {
        self.ne_pop1 = ne.max(1.0);
        self
    }

    pub fn with_ne2(mut self, ne: f64) -> Self {
        self.ne_pop2 = ne.max(1.0);
        self
    }

    pub fn with_ne_ancestral(mut self, ne: f64) -> Self {
        self.ne_ancestral = ne.max(1.0);
        self
    }

    pub fn with_migration(mut self, m_1to2: f64, m_2to1: f64) -> Self {
        self.migration_1to2 = m_1to2.max(0.0);
        self.migration_2to1 = m_2to1.max(0.0);
        self
    }

    pub fn with_divergence_time(mut self, t: f64) -> Self {
        self.divergence_time = t.max(0.0);
        self
    }

    /// Scaled divergence time in coalescent units: T = t / (2 * Ne_anc).
    pub fn scaled_divergence(&self) -> f64 {
        if self.ne_ancestral <= 0.0 { 0.0 } else { self.divergence_time / (2.0 * self.ne_ancestral) }
    }

    /// Scaled migration rates: M = 2 * Ne * m.
    pub fn scaled_migration(&self) -> (f64, f64) {
        (
            2.0 * self.ne_pop1 * self.migration_1to2,
            2.0 * self.ne_pop2 * self.migration_2to1,
        )
    }

    /// Expected Fst under IM model (no migration case).
    pub fn expected_fst_no_migration(&self) -> f64 {
        let t = self.scaled_divergence();
        // Fst ≈ 1 - exp(-T) for divergence without migration
        1.0 - (-t).exp()
    }
}

impl Default for ImParams {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ImParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IM(Ne1={:.0}, Ne2={:.0}, Ne_a={:.0}, m12={:.6}, m21={:.6}, T={:.0}gen)",
            self.ne_pop1, self.ne_pop2, self.ne_ancestral,
            self.migration_1to2, self.migration_2to1, self.divergence_time
        )
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Approximate second eigenvalue of a transition matrix via power iteration.
fn second_eigenvalue(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n < 2 {
        return 0.0;
    }

    // Start with a vector orthogonal to the stationary distribution (1/n, ..., 1/n)
    let mut v: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

    for _ in 0..100 {
        // Matrix-vector multiply
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i][j] * v[j];
            }
        }

        // Remove component along stationary vector
        let mean: f64 = w.iter().sum::<f64>() / n as f64;
        for val in &mut w {
            *val -= mean;
        }

        // Normalize
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return 0.0;
        }
        for val in &mut w {
            *val /= norm;
        }

        v = w;
    }

    // Rayleigh quotient
    let mut mv = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mv[i] += matrix[i][j] * v[j];
        }
    }
    let lambda: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
    lambda
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    #[test]
    fn test_migration_rate_nm() {
        let mr = MigrationRate::new(0, 1, 0.01);
        assert!((mr.nm(1000.0) - 10.0).abs() < EPS);
    }

    #[test]
    fn test_island_model_fst() {
        let model = IslandModel::new(10, 1000.0, 0.01);
        let fst = model.expected_fst();
        // Fst = 1/(1+4*1000*0.01) = 1/41 ≈ 0.0244
        assert!((fst - 1.0 / 41.0).abs() < EPS);
    }

    #[test]
    fn test_island_model_nm() {
        let model = IslandModel::new(5, 500.0, 0.02);
        assert!((model.nm() - 10.0).abs() < EPS);
    }

    #[test]
    fn test_island_migration_matrix() {
        let model = IslandModel::new(3, 1000.0, 0.03);
        let mat = model.to_migration_matrix();
        assert_eq!(mat.n_pops, 3);
        // Each off-diagonal should be 0.03/2 = 0.015
        assert!((mat.matrix[0][1] - 0.015).abs() < EPS);
        assert!((mat.matrix[0][2] - 0.015).abs() < EPS);
    }

    #[test]
    fn test_stepping_stone_linear() {
        let model = SteppingStoneModel::new(5, 1000.0, 0.01);
        let mat = model.to_migration_matrix();
        // Population 0 only has neighbor 1
        assert!(mat.matrix[0][1] > 0.0);
        assert!((mat.matrix[0][2] - 0.0).abs() < EPS);
    }

    #[test]
    fn test_stepping_stone_circular() {
        let model = SteppingStoneModel::new(5, 1000.0, 0.01).with_circular(true);
        let mat = model.to_migration_matrix();
        // Pop 0 neighbors: 4 and 1
        assert!(mat.matrix[0][4] > 0.0);
        assert!(mat.matrix[0][1] > 0.0);
    }

    #[test]
    fn test_stepping_stone_ibd() {
        let model = SteppingStoneModel::new(10, 1000.0, 0.01);
        let fst1 = model.expected_fst_by_distance(1);
        let fst5 = model.expected_fst_by_distance(5);
        assert!(fst5 > fst1); // Fst increases with distance
    }

    #[test]
    fn test_fst_matrix() {
        let model = SteppingStoneModel::new(4, 1000.0, 0.01);
        let mat = model.fst_matrix();
        assert_eq!(mat.len(), 4);
        assert!((mat[0][0] - 0.0).abs() < EPS); // diagonal = 0
        assert!((mat[0][1] - mat[1][0]).abs() < EPS); // symmetric
    }

    #[test]
    fn test_nm_from_fst() {
        let nm = nm_from_fst(0.2);
        assert!((nm - 1.0).abs() < EPS);
    }

    #[test]
    fn test_nm_from_fst_k_pops() {
        let nm2 = nm_from_fst(0.2);
        let nm5 = nm_from_fst_k_pops(0.2, 5);
        // With more populations, Nm estimate changes
        assert!(nm5 != nm2);
    }

    #[test]
    fn test_nm_from_private_alleles() {
        let nm = nm_from_private_alleles(0.1);
        assert!(nm > 0.0);
        assert!(nm.is_finite());
    }

    #[test]
    fn test_im_params() {
        let im = ImParams::new()
            .with_ne1(5000.0)
            .with_ne2(8000.0)
            .with_ne_ancestral(12000.0)
            .with_migration(0.001, 0.002)
            .with_divergence_time(50000.0);
        assert!((im.ne_pop1 - 5000.0).abs() < EPS);
        assert!(im.scaled_divergence() > 0.0);
    }

    #[test]
    fn test_im_no_migration_fst() {
        let im = ImParams::new()
            .with_ne_ancestral(10000.0)
            .with_divergence_time(100000.0);
        let fst = im.expected_fst_no_migration();
        assert!(fst > 0.0);
        assert!(fst <= 1.0);
    }

    #[test]
    fn test_migration_matrix_conservative() {
        let model = IslandModel::new(4, 1000.0, 0.01);
        let mat = model.to_migration_matrix();
        assert!(mat.is_conservative());
    }

    #[test]
    fn test_migration_matrix_display() {
        let mat = MigrationMatrix::new(3);
        let s = format!("{}", mat);
        assert!(s.contains("3x3"));
    }

    #[test]
    fn test_island_model_display() {
        let model = IslandModel::new(5, 1000.0, 0.01);
        let s = format!("{}", model);
        assert!(s.contains("k=5"));
    }

    #[test]
    fn test_stepping_stone_display() {
        let model = SteppingStoneModel::new(5, 1000.0, 0.01);
        let s = format!("{}", model);
        assert!(s.contains("linear"));
    }

    #[test]
    fn test_im_display() {
        let im = ImParams::new();
        let s = format!("{}", im);
        assert!(s.contains("IM"));
    }

    #[test]
    fn test_migration_rate_display() {
        let mr = MigrationRate::new(0, 1, 0.005);
        let s = format!("{}", mr);
        assert!(s.contains("0->1"));
    }

    #[test]
    fn test_total_emigration() {
        let model = IslandModel::new(3, 1000.0, 0.03);
        let mat = model.to_migration_matrix();
        let emig = mat.total_emigration(0);
        assert!(emig > 0.0);
    }
}
