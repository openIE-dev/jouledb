//! Dimensionality Reduction — PCA via eigendecomposition, explained variance,
//! simplified t-SNE, feature scaling (standardize, min-max), and random
//! projection (Johnson-Lindenstrauss).
//!
//! Pure Rust — no external ML or linear algebra dependencies.

use std::fmt;

// ── Feature Scaling ─────────────────────────────────────────────

/// Standardize features to zero mean and unit variance (z-score normalization).
#[derive(Debug, Clone)]
pub struct StandardScaler {
    means: Vec<f64>,
    stds: Vec<f64>,
    n_features: usize,
}

impl StandardScaler {
    /// Fit the scaler on training data.
    pub fn fit(data: &[Vec<f64>]) -> Self {
        assert!(!data.is_empty(), "empty dataset");
        let n = data.len() as f64;
        let n_features = data[0].len();

        let mut means = vec![0.0; n_features];
        for row in data {
            for (j, val) in row.iter().enumerate() {
                means[j] += val;
            }
        }
        for m in &mut means {
            *m /= n;
        }

        let mut stds = vec![0.0; n_features];
        for row in data {
            for (j, val) in row.iter().enumerate() {
                stds[j] += (val - means[j]).powi(2);
            }
        }
        for s in &mut stds {
            *s = (*s / n).sqrt().max(1e-10);
        }

        Self { means, stds, n_features }
    }

    /// Transform data using the fitted parameters.
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| (v - self.means[j]) / self.stds[j])
                    .collect()
            })
            .collect()
    }

    /// Fit and transform in one step.
    pub fn fit_transform(data: &[Vec<f64>]) -> (Self, Vec<Vec<f64>>) {
        let scaler = Self::fit(data);
        let transformed = scaler.transform(data);
        (scaler, transformed)
    }

    /// Inverse transform scaled data back to original space.
    pub fn inverse_transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| v * self.stds[j] + self.means[j])
                    .collect()
            })
            .collect()
    }

    /// Return feature means.
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Return feature standard deviations.
    pub fn stds(&self) -> &[f64] {
        &self.stds
    }
}

impl fmt::Display for StandardScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StandardScaler(features={})", self.n_features)
    }
}

/// Min-max scaling to [0, 1] range.
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    mins: Vec<f64>,
    maxs: Vec<f64>,
    n_features: usize,
}

impl MinMaxScaler {
    /// Fit the scaler on training data.
    pub fn fit(data: &[Vec<f64>]) -> Self {
        assert!(!data.is_empty(), "empty dataset");
        let n_features = data[0].len();
        let mut mins = vec![f64::INFINITY; n_features];
        let mut maxs = vec![f64::NEG_INFINITY; n_features];

        for row in data {
            for (j, val) in row.iter().enumerate() {
                if *val < mins[j] {
                    mins[j] = *val;
                }
                if *val > maxs[j] {
                    maxs[j] = *val;
                }
            }
        }
        Self { mins, maxs, n_features }
    }

    /// Transform data to [0, 1] range.
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| {
                        let range = self.maxs[j] - self.mins[j];
                        if range.abs() < 1e-15 { 0.0 } else { (v - self.mins[j]) / range }
                    })
                    .collect()
            })
            .collect()
    }

    /// Fit and transform in one step.
    pub fn fit_transform(data: &[Vec<f64>]) -> (Self, Vec<Vec<f64>>) {
        let scaler = Self::fit(data);
        let transformed = scaler.transform(data);
        (scaler, transformed)
    }

    /// Inverse transform.
    pub fn inverse_transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| v * (self.maxs[j] - self.mins[j]) + self.mins[j])
                    .collect()
            })
            .collect()
    }
}

impl fmt::Display for MinMaxScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MinMaxScaler(features={})", self.n_features)
    }
}

// ── PCA ─────────────────────────────────────────────────────────

/// Principal Component Analysis via eigendecomposition of the covariance matrix.
#[derive(Debug, Clone)]
pub struct PCA {
    /// Principal component directions (rows are components).
    pub components: Vec<Vec<f64>>,
    /// Eigenvalues (variance explained by each component).
    pub eigenvalues: Vec<f64>,
    /// Explained variance ratio for each component.
    pub explained_variance_ratio: Vec<f64>,
    /// Mean of training data (for centering).
    mean: Vec<f64>,
    /// Number of components retained.
    n_components: usize,
    /// Original number of features.
    n_features: usize,
}

impl PCA {
    /// Fit PCA on the data, retaining `n_components` principal components.
    pub fn fit(data: &[Vec<f64>], n_components: usize) -> Self {
        assert!(!data.is_empty());
        let n = data.len();
        let d = data[0].len();
        let n_components = n_components.min(d);

        // Compute mean
        let mut mean = vec![0.0; d];
        for row in data {
            for (j, val) in row.iter().enumerate() {
                mean[j] += val;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }

        // Center data
        let centered: Vec<Vec<f64>> = data
            .iter()
            .map(|row| row.iter().enumerate().map(|(j, v)| v - mean[j]).collect())
            .collect();

        // Compute covariance matrix: C = X^T * X / (n - 1)
        let mut cov = vec![vec![0.0; d]; d];
        for row in &centered {
            for i in 0..d {
                for j in i..d {
                    cov[i][j] += row[i] * row[j];
                }
            }
        }
        let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
        for i in 0..d {
            for j in i..d {
                cov[i][j] /= denom;
                cov[j][i] = cov[i][j];
            }
        }

        // Power iteration to find top eigenvectors
        let (eigenvalues, eigenvectors) = power_iteration_eigen(&cov, n_components, 200);

        // Explained variance ratio
        let total_var: f64 = eigenvalues.iter().sum::<f64>().max(1e-15);
        let explained_variance_ratio: Vec<f64> =
            eigenvalues.iter().map(|ev| ev / total_var).collect();

        Self {
            components: eigenvectors,
            eigenvalues,
            explained_variance_ratio,
            mean,
            n_components,
            n_features: d,
        }
    }

    /// Transform data into the PCA space.
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                let centered: Vec<f64> =
                    row.iter().enumerate().map(|(j, v)| v - self.mean[j]).collect();
                self.components
                    .iter()
                    .map(|comp| {
                        comp.iter().zip(centered.iter()).map(|(c, x)| c * x).sum()
                    })
                    .collect()
            })
            .collect()
    }

    /// Inverse transform from PCA space back to original space.
    pub fn inverse_transform(&self, reduced: &[Vec<f64>]) -> Vec<Vec<f64>> {
        reduced
            .iter()
            .map(|row| {
                let mut result = self.mean.clone();
                for (i, coeff) in row.iter().enumerate() {
                    if i < self.components.len() {
                        for (j, comp_val) in self.components[i].iter().enumerate() {
                            result[j] += coeff * comp_val;
                        }
                    }
                }
                result
            })
            .collect()
    }

    /// Total explained variance ratio (sum).
    pub fn total_explained_variance(&self) -> f64 {
        self.explained_variance_ratio.iter().sum()
    }

    /// Number of components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

impl fmt::Display for PCA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PCA(components={}, explained_var={:.4})",
            self.n_components,
            self.total_explained_variance()
        )
    }
}

/// Power iteration to find top k eigenvectors of a symmetric matrix.
fn power_iteration_eigen(
    matrix: &[Vec<f64>],
    k: usize,
    max_iter: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let d = matrix.len();
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Working copy for deflation
    let mut mat: Vec<Vec<f64>> = matrix.to_vec();

    for component in 0..k {
        // Initialize with deterministic vector
        let mut v: Vec<f64> = (0..d).map(|i| ((i + component + 1) as f64).sin()).collect();
        let n = vec_norm(&v);
        if n > f64::EPSILON {
            vec_scale_inplace(&mut v, 1.0 / n);
        }

        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            let mut av = mat_vec_mul(&mat, &v);

            // Orthogonalize against previous eigenvectors
            for prev in &eigenvectors {
                let proj = dot_prod(&av, prev);
                for (j, pv) in prev.iter().enumerate() {
                    av[j] -= proj * pv;
                }
            }

            eigenvalue = vec_norm(&av);
            if eigenvalue < f64::EPSILON {
                break;
            }
            vec_scale_inplace(&mut av, 1.0 / eigenvalue);

            let diff: f64 = v
                .iter()
                .zip(av.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            v = av;
            if diff < 1e-10 {
                break;
            }
        }

        // Deflate: M = M - eigenvalue * v * v^T
        for i in 0..d {
            for j in 0..d {
                mat[i][j] -= eigenvalue * v[i] * v[j];
            }
        }

        eigenvalues.push(eigenvalue);
        eigenvectors.push(v);
    }

    (eigenvalues, eigenvectors)
}

// ── t-SNE (Simplified) ──────────────────────────────────────────

/// Simplified t-SNE for dimensionality reduction to 2D.
#[derive(Debug, Clone)]
pub struct TSNE {
    /// Perplexity parameter (typical range: 5-50).
    pub perplexity: f64,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of iterations.
    pub n_iter: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for TSNE {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 500,
            seed: 42,
        }
    }
}

impl TSNE {
    /// Create with custom perplexity.
    pub fn with_perplexity(perplexity: f64) -> Self {
        Self { perplexity, ..Default::default() }
    }

    /// Run t-SNE, reducing data to 2D.
    pub fn fit_transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = data.len();
        if n <= 1 {
            return data.iter().map(|_| vec![0.0, 0.0]).collect();
        }

        // Compute pairwise squared distances
        let mut dists = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = sq_dist(&data[i], &data[j]);
                dists[i][j] = d;
                dists[j][i] = d;
            }
        }

        // Compute pairwise affinities p_ij using binary search for perplexity
        let p = self.compute_pairwise_affinities(&dists, n);

        // Initialize low-dimensional embedding
        let mut y = self.init_embedding(n);

        let mut gains = vec![vec![1.0; 2]; n];
        let mut y_delta = vec![vec![0.0; 2]; n];

        // Gradient descent
        for iter in 0..self.n_iter {
            let momentum = if iter < 250 { 0.5 } else { 0.8 };

            // Compute Q distribution (Student-t with df=1)
            let mut q_numer = vec![vec![0.0; n]; n];
            let mut q_sum = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dist_sq = (y[i][0] - y[j][0]).powi(2) + (y[i][1] - y[j][1]).powi(2);
                    let val = 1.0 / (1.0 + dist_sq);
                    q_numer[i][j] = val;
                    q_numer[j][i] = val;
                    q_sum += 2.0 * val;
                }
            }
            q_sum = q_sum.max(1e-15);

            // Compute gradient
            let mut grad = vec![vec![0.0; 2]; n];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let q_ij = q_numer[i][j] / q_sum;
                    let factor = 4.0 * (p[i][j] - q_ij) * q_numer[i][j];
                    grad[i][0] += factor * (y[i][0] - y[j][0]);
                    grad[i][1] += factor * (y[i][1] - y[j][1]);
                }
            }

            // Update with adaptive gains
            for i in 0..n {
                for d in 0..2 {
                    let sign_match = (grad[i][d] > 0.0) == (y_delta[i][d] > 0.0);
                    let scaled: f64 = gains[i][d] * 0.8;
                    gains[i][d] = if sign_match {
                        scaled.max(0.01)
                    } else {
                        gains[i][d] + 0.2
                    };
                    y_delta[i][d] = momentum * y_delta[i][d]
                        - self.learning_rate * gains[i][d] * grad[i][d];
                    y[i][d] += y_delta[i][d];
                }
            }

            // Re-center
            let mut mean_y = [0.0; 2];
            for yi in &y {
                mean_y[0] += yi[0];
                mean_y[1] += yi[1];
            }
            mean_y[0] /= n as f64;
            mean_y[1] /= n as f64;
            for yi in &mut y {
                yi[0] -= mean_y[0];
                yi[1] -= mean_y[1];
            }
        }

        y
    }

    fn compute_pairwise_affinities(&self, dists: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
        let target_entropy = self.perplexity.ln();
        let mut p = vec![vec![0.0; n]; n];

        for i in 0..n {
            // Binary search for sigma_i
            let mut lo = 1e-10_f64;
            let mut hi = 1e4_f64;
            let mut sigma = 1.0;

            for _ in 0..50 {
                sigma = (lo + hi) / 2.0;
                let two_sigma_sq = 2.0 * sigma * sigma;

                let mut probs = vec![0.0; n];
                for j in 0..n {
                    if j != i {
                        probs[j] = (-dists[i][j] / two_sigma_sq).exp();
                    }
                }
                let sum: f64 = probs.iter().sum::<f64>().max(1e-15);
                for pv in &mut probs {
                    *pv /= sum;
                }

                let h: f64 = -probs
                    .iter()
                    .filter(|&&pv| pv > 1e-15)
                    .map(|pv| pv * pv.ln())
                    .sum::<f64>();

                if (h - target_entropy).abs() < 1e-5 {
                    break;
                }
                if h > target_entropy {
                    hi = sigma;
                } else {
                    lo = sigma;
                }
            }

            let two_sigma_sq = 2.0 * sigma * sigma;
            for j in 0..n {
                if j != i {
                    p[i][j] = (-dists[i][j] / two_sigma_sq).exp();
                }
            }
            let sum: f64 = p[i].iter().sum::<f64>().max(1e-15);
            for pv in &mut p[i] {
                *pv /= sum;
            }
        }

        // Symmetrize: p_ij = (p_i|j + p_j|i) / (2n)
        let mut p_sym = vec![vec![0.0; n]; n];
        let denom = 2.0 * n as f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = (p[i][j] + p[j][i]) / denom;
                let val = val.max(1e-12);
                p_sym[i][j] = val;
                p_sym[j][i] = val;
            }
        }
        p_sym
    }

    fn init_embedding(&self, n: usize) -> Vec<Vec<f64>> {
        let mut rng_state = if self.seed == 0 { 1u64 } else { self.seed };
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let mut point = Vec::with_capacity(2);
            for _ in 0..2 {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let val = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                point.push(val * 0.01);
            }
            result.push(point);
        }
        result
    }
}

impl fmt::Display for TSNE {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TSNE(perplexity={}, lr={}, n_iter={})",
            self.perplexity, self.learning_rate, self.n_iter
        )
    }
}

// ── Random Projection ───────────────────────────────────────────

/// Random projection for dimensionality reduction (Johnson-Lindenstrauss lemma).
#[derive(Debug, Clone)]
pub struct RandomProjection {
    /// Projection matrix (target_dim x original_dim).
    pub projection_matrix: Vec<Vec<f64>>,
    /// Target dimensionality.
    pub target_dim: usize,
    /// Original dimensionality.
    pub original_dim: usize,
}

impl RandomProjection {
    /// Create a random projection to `target_dim` dimensions.
    pub fn new(original_dim: usize, target_dim: usize, seed: u64) -> Self {
        let scale = (1.0 / target_dim as f64).sqrt();
        let mut rng_state = if seed == 0 { 1u64 } else { seed };

        let mut matrix = Vec::with_capacity(target_dim);
        for _ in 0..target_dim {
            let mut row = Vec::with_capacity(original_dim);
            for _ in 0..original_dim {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                // Sparse random projection: {-1, 0, +1} with probabilities {1/6, 2/3, 1/6}
                let r = rng_state % 6;
                let val = if r == 0 {
                    -scale * 3.0_f64.sqrt()
                } else if r == 5 {
                    scale * 3.0_f64.sqrt()
                } else {
                    0.0
                };
                row.push(val);
            }
            matrix.push(row);
        }

        Self {
            projection_matrix: matrix,
            target_dim,
            original_dim,
        }
    }

    /// Compute the minimum target dimension for JL guarantee.
    ///
    /// For `n` points and `eps` distortion, target_dim >= 4 * ln(n) / (eps^2 / 2 - eps^3 / 3).
    pub fn jl_min_dim(n_samples: usize, eps: f64) -> usize {
        if n_samples <= 1 || eps <= 0.0 || eps >= 1.0 {
            return 1;
        }
        let numerator = 4.0 * (n_samples as f64).ln();
        let denominator = eps * eps / 2.0 - eps * eps * eps / 3.0;
        (numerator / denominator).ceil() as usize
    }

    /// Transform data using the random projection.
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                self.projection_matrix
                    .iter()
                    .map(|proj_row| {
                        proj_row.iter().zip(row.iter()).map(|(p, x)| p * x).sum()
                    })
                    .collect()
            })
            .collect()
    }
}

impl fmt::Display for RandomProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RandomProjection({} -> {})",
            self.original_dim, self.target_dim
        )
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_norm(v: &[f64]) -> f64 {
    dot_prod(v, v).sqrt()
}

fn vec_scale_inplace(v: &mut [f64], s: f64) {
    for val in v.iter_mut() {
        *val *= s;
    }
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<Vec<f64>> {
        vec![
            vec![2.5, 2.4],
            vec![0.5, 0.7],
            vec![2.2, 2.9],
            vec![1.9, 2.2],
            vec![3.1, 3.0],
            vec![2.3, 2.7],
            vec![2.0, 1.6],
            vec![1.0, 1.1],
            vec![1.5, 1.6],
            vec![1.1, 0.9],
        ]
    }

    fn clustered_data() -> Vec<Vec<f64>> {
        vec![
            vec![0.1, 0.1, 0.0], vec![0.2, 0.0, 0.1], vec![0.0, 0.2, 0.1],
            vec![10.0, 10.0, 10.0], vec![10.1, 10.2, 10.0], vec![10.0, 9.9, 10.1],
        ]
    }

    #[test]
    fn test_standard_scaler_fit() {
        let data = sample_data();
        let scaler = StandardScaler::fit(&data);
        assert_eq!(scaler.means().len(), 2);
        assert_eq!(scaler.stds().len(), 2);
    }

    #[test]
    fn test_standard_scaler_transform() {
        let data = sample_data();
        let (scaler, transformed) = StandardScaler::fit_transform(&data);
        // Mean of transformed data should be approximately 0
        let mut means = vec![0.0; 2];
        for row in &transformed {
            for (j, v) in row.iter().enumerate() {
                means[j] += v;
            }
        }
        for m in &mut means {
            *m /= transformed.len() as f64;
        }
        assert!(means[0].abs() < 1e-10);
        assert!(means[1].abs() < 1e-10);
        let _ = scaler; // suppress unused
    }

    #[test]
    fn test_standard_scaler_inverse() {
        let data = sample_data();
        let (scaler, transformed) = StandardScaler::fit_transform(&data);
        let restored = scaler.inverse_transform(&transformed);
        for (orig, rest) in data.iter().zip(restored.iter()) {
            for (a, b) in orig.iter().zip(rest.iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_minmax_scaler_range() {
        let data = sample_data();
        let (_scaler, transformed) = MinMaxScaler::fit_transform(&data);
        for row in &transformed {
            for &v in row {
                assert!(v >= -1e-10 && v <= 1.0 + 1e-10);
            }
        }
    }

    #[test]
    fn test_minmax_scaler_inverse() {
        let data = sample_data();
        let (scaler, transformed) = MinMaxScaler::fit_transform(&data);
        let restored = scaler.inverse_transform(&transformed);
        for (orig, rest) in data.iter().zip(restored.iter()) {
            for (a, b) in orig.iter().zip(rest.iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_pca_fit() {
        let data = sample_data();
        let pca = PCA::fit(&data, 2);
        assert_eq!(pca.n_components(), 2);
        assert_eq!(pca.components.len(), 2);
        assert_eq!(pca.eigenvalues.len(), 2);
    }

    #[test]
    fn test_pca_transform() {
        let data = sample_data();
        let pca = PCA::fit(&data, 2);
        let transformed = pca.transform(&data);
        assert_eq!(transformed.len(), data.len());
        assert_eq!(transformed[0].len(), 2);
    }

    #[test]
    fn test_pca_reduce_dim() {
        let data = sample_data();
        let pca = PCA::fit(&data, 1);
        let transformed = pca.transform(&data);
        assert_eq!(transformed[0].len(), 1);
    }

    #[test]
    fn test_pca_explained_variance() {
        let data = sample_data();
        let pca = PCA::fit(&data, 2);
        let total = pca.total_explained_variance();
        assert!(total > 0.9); // 2 components should explain most variance for 2D data
    }

    #[test]
    fn test_pca_inverse_transform() {
        let data = sample_data();
        let pca = PCA::fit(&data, 2);
        let transformed = pca.transform(&data);
        let restored = pca.inverse_transform(&transformed);
        // With all components retained, reconstruction should be close
        for (orig, rest) in data.iter().zip(restored.iter()) {
            for (a, b) in orig.iter().zip(rest.iter()) {
                assert!((a - b).abs() < 1.0);
            }
        }
    }

    #[test]
    fn test_pca_display() {
        let data = sample_data();
        let pca = PCA::fit(&data, 2);
        let s = format!("{}", pca);
        assert!(s.contains("PCA"));
    }

    #[test]
    fn test_tsne_basic() {
        let data = clustered_data();
        let tsne = TSNE { n_iter: 100, seed: 42, ..Default::default() };
        let result = tsne.fit_transform(&data);
        assert_eq!(result.len(), 6);
        assert_eq!(result[0].len(), 2);
        // All values should be finite
        for row in &result {
            for v in row {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_tsne_separation() {
        let data = clustered_data();
        let tsne = TSNE { n_iter: 200, seed: 42, ..Default::default() };
        let result = tsne.fit_transform(&data);
        // Cluster 0 (first 3) should be closer together than to cluster 1 (last 3)
        let c0_center = [
            (result[0][0] + result[1][0] + result[2][0]) / 3.0,
            (result[0][1] + result[1][1] + result[2][1]) / 3.0,
        ];
        let c1_center = [
            (result[3][0] + result[4][0] + result[5][0]) / 3.0,
            (result[3][1] + result[4][1] + result[5][1]) / 3.0,
        ];
        let inter_dist = (c0_center[0] - c1_center[0]).powi(2) + (c0_center[1] - c1_center[1]).powi(2);
        assert!(inter_dist > 0.0); // clusters should be separated
    }

    #[test]
    fn test_tsne_display() {
        let tsne = TSNE::default();
        let s = format!("{}", tsne);
        assert!(s.contains("TSNE"));
    }

    #[test]
    fn test_random_projection_basic() {
        let rp = RandomProjection::new(10, 3, 42);
        assert_eq!(rp.original_dim, 10);
        assert_eq!(rp.target_dim, 3);
        assert_eq!(rp.projection_matrix.len(), 3);
        assert_eq!(rp.projection_matrix[0].len(), 10);
    }

    #[test]
    fn test_random_projection_transform() {
        let data = vec![vec![1.0; 10], vec![2.0; 10], vec![0.0; 10]];
        let rp = RandomProjection::new(10, 3, 42);
        let result = rp.transform(&data);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_jl_min_dim() {
        let dim = RandomProjection::jl_min_dim(1000, 0.1);
        assert!(dim > 0);
        assert!(dim < 10000);
    }

    #[test]
    fn test_random_projection_display() {
        let rp = RandomProjection::new(100, 10, 42);
        let s = format!("{}", rp);
        assert!(s.contains("100"));
        assert!(s.contains("10"));
    }

    #[test]
    fn test_standard_scaler_display() {
        let data = sample_data();
        let scaler = StandardScaler::fit(&data);
        let s = format!("{}", scaler);
        assert!(s.contains("StandardScaler"));
    }

    #[test]
    fn test_minmax_scaler_display() {
        let data = sample_data();
        let scaler = MinMaxScaler::fit(&data);
        let s = format!("{}", scaler);
        assert!(s.contains("MinMaxScaler"));
    }
}
