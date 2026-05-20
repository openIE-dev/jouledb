//! Spatial Autocorrelation — Moran's I (global/local), Geary's C, spatial
//! weights matrix (contiguity/distance), LISA cluster map, significance
//! testing (permutation), SpatialWeightsConfig builder.
//!
//! All computation is std-only, f64 precision, permutation-based inference.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SpatialAutoError {
    TooFewObservations(usize),
    DimensionMismatch { expected: usize, got: usize },
    InvalidThreshold(f64),
    NoNeighbors(usize),
    InvalidPermutations(usize),
}

impl fmt::Display for SpatialAutoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewObservations(n) => write!(f, "need >= 3 observations, got {n}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidThreshold(t) => write!(f, "invalid distance threshold: {t}"),
            Self::NoNeighbors(i) => write!(f, "observation {i} has no neighbors"),
            Self::InvalidPermutations(p) => write!(f, "permutations must be > 0, got {p}"),
        }
    }
}

impl std::error::Error for SpatialAutoError {}

// ── Location ────────────────────────────────────────────────────

/// A 2-D point with an associated attribute value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Location {
    pub x: f64,
    pub y: f64,
    pub value: f64,
}

impl Location {
    pub fn new(x: f64, y: f64, value: f64) -> Self {
        Self { x, y, value }
    }

    pub fn distance_to(&self, other: &Location) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}) val={:.4}", self.x, self.y, self.value)
    }
}

// ── Weights type ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightsType {
    /// Queen / rook contiguity approximated via distance threshold.
    Contiguity,
    /// Inverse-distance weighting.
    InverseDistance,
    /// Binary: 1 if within threshold, 0 otherwise.
    Binary,
    /// K-nearest neighbors.
    KNearest(usize),
}

impl fmt::Display for WeightsType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contiguity => write!(f, "contiguity"),
            Self::InverseDistance => write!(f, "inverse-distance"),
            Self::Binary => write!(f, "binary"),
            Self::KNearest(k) => write!(f, "{k}-nearest-neighbors"),
        }
    }
}

// ── SpatialWeightsConfig ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SpatialWeightsConfig {
    pub weights_type: WeightsType,
    pub threshold: f64,
    pub row_standardize: bool,
}

impl SpatialWeightsConfig {
    pub fn new() -> Self {
        Self {
            weights_type: WeightsType::Binary,
            threshold: 1.0,
            row_standardize: true,
        }
    }

    pub fn with_weights_type(mut self, wt: WeightsType) -> Self {
        self.weights_type = wt;
        self
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    pub fn with_row_standardize(mut self, rs: bool) -> Self {
        self.row_standardize = rs;
        self
    }
}

impl Default for SpatialWeightsConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SpatialWeightsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpatialWeightsConfig(type={}, threshold={:.4}, row_std={})",
            self.weights_type, self.threshold, self.row_standardize
        )
    }
}

// ── Spatial weights matrix ──────────────────────────────────────

/// Row-major dense spatial weights matrix.
#[derive(Debug, Clone)]
pub struct SpatialWeights {
    pub n: usize,
    pub data: Vec<f64>,
}

impl SpatialWeights {
    pub fn build(
        locations: &[Location],
        config: &SpatialWeightsConfig,
    ) -> Result<Self, SpatialAutoError> {
        let n = locations.len();
        if n < 3 {
            return Err(SpatialAutoError::TooFewObservations(n));
        }
        if config.threshold <= 0.0 && !matches!(config.weights_type, WeightsType::KNearest(_)) {
            return Err(SpatialAutoError::InvalidThreshold(config.threshold));
        }
        let mut data = vec![0.0; n * n];

        match config.weights_type {
            WeightsType::Binary | WeightsType::Contiguity => {
                for i in 0..n {
                    for j in 0..n {
                        if i != j && locations[i].distance_to(&locations[j]) <= config.threshold {
                            data[i * n + j] = 1.0;
                        }
                    }
                }
            }
            WeightsType::InverseDistance => {
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            let d = locations[i].distance_to(&locations[j]);
                            if d <= config.threshold && d > 1e-15 {
                                data[i * n + j] = 1.0 / d;
                            }
                        }
                    }
                }
            }
            WeightsType::KNearest(k) => {
                for i in 0..n {
                    let mut dists: Vec<(usize, f64)> = (0..n)
                        .filter(|j| *j != i)
                        .map(|j| (j, locations[i].distance_to(&locations[j])))
                        .collect();
                    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    for &(j, _) in dists.iter().take(k) {
                        data[i * n + j] = 1.0;
                    }
                }
            }
        }

        if config.row_standardize {
            for i in 0..n {
                let row_sum: f64 = (0..n).map(|j| data[i * n + j]).sum();
                if row_sum > 1e-15 {
                    for j in 0..n {
                        data[i * n + j] /= row_sum;
                    }
                }
            }
        }

        Ok(Self { n, data })
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    pub fn total_weight(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn neighbor_count(&self, i: usize) -> usize {
        (0..self.n).filter(|j| self.get(i, *j) > 1e-15).count()
    }
}

impl fmt::Display for SpatialWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialWeights(n={}, total_w={:.4})", self.n, self.total_weight())
    }
}

// ── LISA cluster type ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LisaCluster {
    HighHigh,
    LowLow,
    HighLow,
    LowHigh,
    NotSignificant,
}

impl fmt::Display for LisaCluster {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighHigh => write!(f, "High-High"),
            Self::LowLow => write!(f, "Low-Low"),
            Self::HighLow => write!(f, "High-Low"),
            Self::LowHigh => write!(f, "Low-High"),
            Self::NotSignificant => write!(f, "Not Significant"),
        }
    }
}

// ── Autocorrelation result ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AutocorrResult {
    pub statistic: f64,
    pub expected: f64,
    pub variance: f64,
    pub z_score: f64,
    pub p_value: f64,
}

impl fmt::Display for AutocorrResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stat={:.6}, E={:.6}, var={:.6}, z={:.4}, p={:.6}",
            self.statistic, self.expected, self.variance, self.z_score, self.p_value
        )
    }
}

// ── LISA result ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LisaResult {
    pub local_i: Vec<f64>,
    pub clusters: Vec<LisaCluster>,
    pub p_values: Vec<f64>,
}

impl fmt::Display for LisaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sig = self.clusters.iter().filter(|c| **c != LisaCluster::NotSignificant).count();
        write!(f, "LISA({} obs, {} significant)", self.local_i.len(), sig)
    }
}

// ── Simple PRNG for permutation tests ───────────────────────────

struct SimplePrng {
    state: u64,
}

impl SimplePrng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn shuffle(&mut self, slice: &mut [f64]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// ── Helper functions ────────────────────────────────────────────

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn variance_pop(vals: &[f64]) -> f64 {
    let m = mean(vals);
    vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / vals.len() as f64
}

fn z_to_p(z: f64) -> f64 {
    // Two-tailed p-value approximation via error function.
    let az = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * az);
    let d = 0.3989422804014327;
    let p_tail = d * (-az * az / 2.0).exp()
        * (t * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))));
    2.0 * p_tail
}

// ── Global Moran's I ────────────────────────────────────────────

pub fn morans_i(
    locations: &[Location],
    weights: &SpatialWeights,
) -> Result<AutocorrResult, SpatialAutoError> {
    let n = locations.len();
    if n != weights.n {
        return Err(SpatialAutoError::DimensionMismatch { expected: n, got: weights.n });
    }
    let vals: Vec<f64> = locations.iter().map(|l| l.value).collect();
    let m = mean(&vals);
    let denom: f64 = vals.iter().map(|v| (v - m) * (v - m)).sum();
    if denom.abs() < 1e-15 {
        return Ok(AutocorrResult { statistic: 0.0, expected: 0.0, variance: 0.0, z_score: 0.0, p_value: 1.0 });
    }

    let s0 = weights.total_weight();
    let mut numer = 0.0;
    for i in 0..n {
        for j in 0..n {
            numer += weights.get(i, j) * (vals[i] - m) * (vals[j] - m);
        }
    }

    let i_stat = (n as f64 / s0) * (numer / denom);
    let expected = -1.0 / (n as f64 - 1.0);
    // Normality assumption variance
    let n_f = n as f64;
    let var_i = (n_f * n_f - 3.0 * n_f + 3.0) / ((n_f - 1.0) * (n_f + 1.0) * s0 * s0)
        - expected * expected;
    let var_safe = if var_i > 1e-15 { var_i } else { 1e-15 };
    let z = (i_stat - expected) / var_safe.sqrt();
    let p = z_to_p(z);

    Ok(AutocorrResult { statistic: i_stat, expected, variance: var_safe, z_score: z, p_value: p })
}

// ── Geary's C ───────────────────────────────────────────────────

pub fn gearys_c(
    locations: &[Location],
    weights: &SpatialWeights,
) -> Result<AutocorrResult, SpatialAutoError> {
    let n = locations.len();
    if n != weights.n {
        return Err(SpatialAutoError::DimensionMismatch { expected: n, got: weights.n });
    }
    let vals: Vec<f64> = locations.iter().map(|l| l.value).collect();
    let m = mean(&vals);
    let denom: f64 = vals.iter().map(|v| (v - m) * (v - m)).sum();
    if denom.abs() < 1e-15 {
        return Ok(AutocorrResult { statistic: 1.0, expected: 1.0, variance: 0.0, z_score: 0.0, p_value: 1.0 });
    }

    let s0 = weights.total_weight();
    let mut numer = 0.0;
    for i in 0..n {
        for j in 0..n {
            numer += weights.get(i, j) * (vals[i] - vals[j]) * (vals[i] - vals[j]);
        }
    }

    let c_stat = ((n as f64 - 1.0) / (2.0 * s0)) * (numer / denom);
    let expected = 1.0;
    let n_f = n as f64;
    let var_c = (2.0 * s0 * s0 + 2.0 * n_f - 3.0) / ((n_f - 1.0) * (n_f + 1.0) * s0 * s0);
    let var_safe = if var_c > 1e-15 { var_c } else { 1e-15 };
    let z = (c_stat - expected) / var_safe.sqrt();
    let p = z_to_p(z);

    Ok(AutocorrResult { statistic: c_stat, expected, variance: var_safe, z_score: z, p_value: p })
}

// ── Permutation test ────────────────────────────────────────────

pub fn permutation_test(
    locations: &[Location],
    weights: &SpatialWeights,
    permutations: usize,
    seed: u64,
) -> Result<f64, SpatialAutoError> {
    if permutations == 0 {
        return Err(SpatialAutoError::InvalidPermutations(permutations));
    }
    let observed = morans_i(locations, weights)?.statistic;
    let mut vals: Vec<f64> = locations.iter().map(|l| l.value).collect();
    let mut rng = SimplePrng::new(seed);
    let mut count_extreme = 0usize;

    let locs_coords: Vec<(f64, f64)> = locations.iter().map(|l| (l.x, l.y)).collect();
    for _ in 0..permutations {
        rng.shuffle(&mut vals);
        let perm_locs: Vec<Location> = locs_coords
            .iter()
            .zip(vals.iter())
            .map(|(&(x, y), &v)| Location::new(x, y, v))
            .collect();
        if let Ok(r) = morans_i(&perm_locs, weights) {
            if r.statistic.abs() >= observed.abs() {
                count_extreme += 1;
            }
        }
    }
    Ok((count_extreme as f64 + 1.0) / (permutations as f64 + 1.0))
}

// ── Local Moran's I (LISA) ──────────────────────────────────────

pub fn lisa(
    locations: &[Location],
    weights: &SpatialWeights,
    significance: f64,
    permutations: usize,
    seed: u64,
) -> Result<LisaResult, SpatialAutoError> {
    let n = locations.len();
    if n != weights.n {
        return Err(SpatialAutoError::DimensionMismatch { expected: n, got: weights.n });
    }
    if n < 3 {
        return Err(SpatialAutoError::TooFewObservations(n));
    }

    let vals: Vec<f64> = locations.iter().map(|l| l.value).collect();
    let m = mean(&vals);
    let var = variance_pop(&vals);
    if var < 1e-15 {
        return Ok(LisaResult {
            local_i: vec![0.0; n],
            clusters: vec![LisaCluster::NotSignificant; n],
            p_values: vec![1.0; n],
        });
    }

    let z_vals: Vec<f64> = vals.iter().map(|v| (v - m) / var.sqrt()).collect();

    let mut local_i = vec![0.0; n];
    for i in 0..n {
        let mut lag = 0.0;
        for j in 0..n {
            lag += weights.get(i, j) * z_vals[j];
        }
        local_i[i] = z_vals[i] * lag;
    }

    // Permutation p-values per location
    let mut p_values = vec![1.0; n];
    let mut rng = SimplePrng::new(seed);
    let mut perm_vals = vals.clone();
    for i in 0..n {
        let mut extreme = 0usize;
        for _ in 0..permutations {
            rng.shuffle(&mut perm_vals);
            let perm_z: Vec<f64> = perm_vals.iter().map(|v| (v - m) / var.sqrt()).collect();
            let mut lag_p = 0.0;
            for j in 0..n {
                lag_p += weights.get(i, j) * perm_z[j];
            }
            let li_p = perm_z[i] * lag_p;
            if li_p.abs() >= local_i[i].abs() {
                extreme += 1;
            }
        }
        p_values[i] = (extreme as f64 + 1.0) / (permutations as f64 + 1.0);
    }

    let mut clusters = vec![LisaCluster::NotSignificant; n];
    for i in 0..n {
        if p_values[i] > significance {
            continue;
        }
        let zi = z_vals[i];
        let mut lag = 0.0;
        for j in 0..n {
            lag += weights.get(i, j) * z_vals[j];
        }
        clusters[i] = match (zi > 0.0, lag > 0.0) {
            (true, true) => LisaCluster::HighHigh,
            (false, false) => LisaCluster::LowLow,
            (true, false) => LisaCluster::HighLow,
            (false, true) => LisaCluster::LowHigh,
        };
    }

    Ok(LisaResult { local_i, clusters, p_values })
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grid() -> Vec<Location> {
        vec![
            Location::new(0.0, 0.0, 10.0),
            Location::new(1.0, 0.0, 12.0),
            Location::new(2.0, 0.0, 14.0),
            Location::new(0.0, 1.0, 11.0),
            Location::new(1.0, 1.0, 13.0),
            Location::new(2.0, 1.0, 15.0),
            Location::new(0.0, 2.0, 9.0),
            Location::new(1.0, 2.0, 8.0),
            Location::new(2.0, 2.0, 7.0),
        ]
    }

    #[test]
    fn test_location_display() {
        let l = Location::new(1.5, 2.3, 4.0);
        assert!(format!("{l}").contains("1.5"));
    }

    #[test]
    fn test_location_distance() {
        let a = Location::new(0.0, 0.0, 0.0);
        let b = Location::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let c = SpatialWeightsConfig::new()
            .with_weights_type(WeightsType::InverseDistance)
            .with_threshold(2.0)
            .with_row_standardize(false);
        assert_eq!(c.weights_type, WeightsType::InverseDistance);
        assert!((c.threshold - 2.0).abs() < 1e-10);
        assert!(!c.row_standardize);
    }

    #[test]
    fn test_config_display() {
        let c = SpatialWeightsConfig::default();
        let s = format!("{c}");
        assert!(s.contains("binary"));
    }

    #[test]
    fn test_weights_binary() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5).with_row_standardize(false);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        assert_eq!(w.n, 9);
        assert!(w.get(0, 1) > 0.0);
        assert!((w.get(0, 0)).abs() < 1e-15);
    }

    #[test]
    fn test_weights_knearest() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new()
            .with_weights_type(WeightsType::KNearest(2))
            .with_row_standardize(false);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        for i in 0..9 {
            let nc = w.neighbor_count(i);
            assert_eq!(nc, 2);
        }
    }

    #[test]
    fn test_weights_row_standardized() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        for i in 0..w.n {
            let row_sum: f64 = (0..w.n).map(|j| w.get(i, j)).sum();
            if w.neighbor_count(i) > 0 {
                assert!((row_sum - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_weights_too_few() {
        let locs = vec![Location::new(0.0, 0.0, 1.0)];
        let cfg = SpatialWeightsConfig::new();
        assert!(SpatialWeights::build(&locs, &cfg).is_err());
    }

    #[test]
    fn test_weights_invalid_threshold() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(-1.0);
        assert!(SpatialWeights::build(&locs, &cfg).is_err());
    }

    #[test]
    fn test_morans_i_positive() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        let r = morans_i(&locs, &w).unwrap();
        assert!(r.statistic.is_finite());
        assert!(r.z_score.is_finite());
    }

    #[test]
    fn test_morans_i_dimension_mismatch() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        let short_locs = &locs[..5];
        assert!(morans_i(short_locs, &w).is_err());
    }

    #[test]
    fn test_gearys_c() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        let r = gearys_c(&locs, &w).unwrap();
        assert!(r.statistic.is_finite());
        assert!(r.expected == 1.0);
    }

    #[test]
    fn test_permutation_test() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        let p = permutation_test(&locs, &w, 99, 42).unwrap();
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn test_permutation_invalid() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        assert!(permutation_test(&locs, &w, 0, 42).is_err());
    }

    #[test]
    fn test_lisa_clusters() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        let r = lisa(&locs, &w, 0.05, 99, 42).unwrap();
        assert_eq!(r.local_i.len(), 9);
        assert_eq!(r.clusters.len(), 9);
        assert_eq!(r.p_values.len(), 9);
    }

    #[test]
    fn test_lisa_display() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        let r = lisa(&locs, &w, 0.05, 99, 42).unwrap();
        let s = format!("{r}");
        assert!(s.contains("LISA"));
    }

    #[test]
    fn test_lisa_cluster_display() {
        assert_eq!(format!("{}", LisaCluster::HighHigh), "High-High");
        assert_eq!(format!("{}", LisaCluster::LowLow), "Low-Low");
        assert_eq!(format!("{}", LisaCluster::NotSignificant), "Not Significant");
    }

    #[test]
    fn test_constant_values() {
        let locs: Vec<Location> = (0..5)
            .map(|i| Location::new(i as f64, 0.0, 5.0))
            .collect();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        let r = morans_i(&locs, &w).unwrap();
        assert!((r.statistic).abs() < 1e-10);
    }

    #[test]
    fn test_weights_inverse_distance() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new()
            .with_weights_type(WeightsType::InverseDistance)
            .with_threshold(1.5)
            .with_row_standardize(false);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        assert!(w.get(0, 1) > w.get(0, 4));
    }

    #[test]
    fn test_weights_display() {
        let locs = sample_grid();
        let cfg = SpatialWeightsConfig::new().with_threshold(1.5);
        let w = SpatialWeights::build(&locs, &cfg).unwrap();
        let s = format!("{w}");
        assert!(s.contains("SpatialWeights"));
    }

    #[test]
    fn test_autocorr_result_display() {
        let r = AutocorrResult {
            statistic: 0.5,
            expected: -0.125,
            variance: 0.01,
            z_score: 6.25,
            p_value: 0.001,
        };
        let s = format!("{r}");
        assert!(s.contains("stat="));
    }
}
