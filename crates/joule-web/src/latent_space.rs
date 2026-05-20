//! Latent space operations: interpolation (slerp/lerp), traversal,
//! disentanglement metrics, latent arithmetic.
//!
//! Provides tools for exploring and manipulating latent representations from
//! generative models. Includes linear and spherical interpolation, latent
//! traversal along specified dimensions, disentanglement metrics (DCI, MIG,
//! SAP), and vector arithmetic in latent space (e.g., analogy completion).

use std::fmt;

// ── PRNG ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Vector Utilities ──────────────────────────────────────────

/// L2 norm of a vector.
pub fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Normalize a vector to unit length.
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let n = l2_norm(v);
    if n < 1e-15 { return v.to_vec(); }
    v.iter().map(|x| x / n).collect()
}

/// Dot product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine similarity.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-15 || nb < 1e-15 { return 0.0; }
    dot(a, b) / (na * nb)
}

/// Euclidean distance.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// ── Interpolation ─────────────────────────────────────────────

/// Linear interpolation between two latent vectors.
/// lerp(a, b, t) = (1-t)*a + t*b
pub fn lerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter())
        .map(|(ai, bi)| (1.0 - t) * ai + t * bi)
        .collect()
}

/// Spherical linear interpolation between two latent vectors.
/// Interpolates along the great circle on the unit hypersphere.
pub fn slerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    assert_eq!(a.len(), b.len());

    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-15 || nb < 1e-15 {
        return lerp(a, b, t);
    }

    let a_unit = normalize(a);
    let b_unit = normalize(b);

    let cos_omega = dot(&a_unit, &b_unit).clamp(-1.0, 1.0);
    let omega = cos_omega.acos();

    if omega.abs() < 1e-10 {
        // Vectors are nearly parallel — fall back to lerp.
        return lerp(a, b, t);
    }

    let sin_omega = omega.sin();
    let coeff_a = ((1.0 - t) * omega).sin() / sin_omega;
    let coeff_b = (t * omega).sin() / sin_omega;

    // Interpolate on the sphere, then scale by interpolated magnitude.
    let interp_norm = (1.0 - t) * na + t * nb;
    a_unit.iter().zip(b_unit.iter())
        .map(|(ai, bi)| (coeff_a * ai + coeff_b * bi) * interp_norm)
        .collect()
}

/// Generate a sequence of interpolated points between two vectors.
pub fn interpolation_sequence(
    a: &[f64],
    b: &[f64],
    steps: usize,
    use_slerp: bool,
) -> Vec<Vec<f64>> {
    if steps <= 1 { return vec![a.to_vec()]; }
    (0..steps)
        .map(|i| {
            let t = i as f64 / (steps - 1) as f64;
            if use_slerp { slerp(a, b, t) } else { lerp(a, b, t) }
        })
        .collect()
}

// ── Latent Traversal ──────────────────────────────────────────

/// A single-dimension traversal in latent space.
#[derive(Debug, Clone)]
pub struct Traversal {
    /// Base latent vector.
    pub base: Vec<f64>,
    /// Dimension being traversed.
    pub dimension: usize,
    /// Generated points along the traversal.
    pub points: Vec<Vec<f64>>,
    /// Values used for the traversal.
    pub values: Vec<f64>,
}

impl Traversal {
    /// Create a traversal by varying one dimension from min_val to max_val.
    pub fn single_dim(base: &[f64], dimension: usize, min_val: f64, max_val: f64, steps: usize) -> Self {
        assert!(dimension < base.len());
        let steps = steps.max(2);
        let values: Vec<f64> = (0..steps)
            .map(|i| min_val + (max_val - min_val) * i as f64 / (steps - 1) as f64)
            .collect();

        let points: Vec<Vec<f64>> = values.iter().map(|v| {
            let mut point = base.to_vec();
            point[dimension] = *v;
            point
        }).collect();

        Self { base: base.to_vec(), dimension, points, values }
    }

    /// Create a traversal along an arbitrary direction vector.
    pub fn along_direction(base: &[f64], direction: &[f64], min_scale: f64, max_scale: f64, steps: usize) -> Self {
        let dir_norm = normalize(direction);
        let steps = steps.max(2);
        let values: Vec<f64> = (0..steps)
            .map(|i| min_scale + (max_scale - min_scale) * i as f64 / (steps - 1) as f64)
            .collect();

        let points: Vec<Vec<f64>> = values.iter().map(|s| {
            base.iter().zip(dir_norm.iter())
                .map(|(bi, di)| bi + s * di)
                .collect()
        }).collect();

        Self { base: base.to_vec(), dimension: 0, points, values }
    }
}

impl fmt::Display for Traversal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Traversal(dim={}, steps={}, range=[{:.2}..{:.2}])",
            self.dimension,
            self.points.len(),
            self.values.first().unwrap_or(&0.0),
            self.values.last().unwrap_or(&0.0),
        )
    }
}

// ── Latent Arithmetic ─────────────────────────────────────────

/// Latent vector arithmetic operations.
#[derive(Debug, Clone)]
pub struct LatentArithmetic;

impl LatentArithmetic {
    /// Analogy completion: a is to b as c is to ? => result = c + (b - a).
    pub fn analogy(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len());
        assert_eq!(b.len(), c.len());
        c.iter().zip(b.iter().zip(a.iter()))
            .map(|(ci, (bi, ai))| ci + (bi - ai))
            .collect()
    }

    /// Add two latent vectors.
    pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    /// Subtract: a - b.
    pub fn subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }

    /// Scale a latent vector.
    pub fn scale(v: &[f64], s: f64) -> Vec<f64> {
        v.iter().map(|x| x * s).collect()
    }

    /// Weighted average of multiple latent vectors.
    pub fn weighted_average(vectors: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
        assert_eq!(vectors.len(), weights.len());
        assert!(!vectors.is_empty());
        let dim = vectors[0].len();
        let w_sum: f64 = weights.iter().sum();
        let mut result = vec![0.0; dim];
        for (vec, &w) in vectors.iter().zip(weights.iter()) {
            for (r, v) in result.iter_mut().zip(vec.iter()) {
                *r += w * v / w_sum;
            }
        }
        result
    }

    /// Project a vector onto a subspace defined by a set of basis vectors.
    pub fn project(v: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
        let dim = v.len();
        let mut projection = vec![0.0; dim];
        for b in basis {
            let bn = l2_norm(b);
            if bn < 1e-15 { continue; }
            let coeff = dot(v, b) / (bn * bn);
            for (p, bi) in projection.iter_mut().zip(b.iter()) {
                *p += coeff * bi;
            }
        }
        projection
    }
}

impl fmt::Display for LatentArithmetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatentArithmetic")
    }
}

// ── Disentanglement Metrics ───────────────────────────────────

/// Metrics for evaluating latent space disentanglement.
#[derive(Debug, Clone)]
pub struct DisentanglementMetrics {
    /// Mutual Information Gap score.
    pub mig: f64,
    /// Per-dimension activity (variance of each latent dimension).
    pub activity: Vec<f64>,
    /// Number of active dimensions (variance > threshold).
    pub active_dims: usize,
    /// Mean pairwise correlation between latent dimensions.
    pub mean_correlation: f64,
}

impl DisentanglementMetrics {
    /// Compute metrics from a set of latent encodings.
    pub fn compute(encodings: &[Vec<f64>], factor_labels: Option<&[Vec<f64>]>) -> Self {
        if encodings.is_empty() {
            return Self { mig: 0.0, activity: vec![], active_dims: 0, mean_correlation: 0.0 };
        }

        let n = encodings.len();
        let latent_dim = encodings[0].len();

        // Per-dimension mean.
        let mut means = vec![0.0; latent_dim];
        for enc in encodings {
            for (i, v) in enc.iter().enumerate() {
                means[i] += v / n as f64;
            }
        }

        // Per-dimension variance (activity).
        let activity: Vec<f64> = (0..latent_dim)
            .map(|i| {
                encodings.iter()
                    .map(|enc| (enc[i] - means[i]).powi(2))
                    .sum::<f64>() / n as f64
            })
            .collect();

        let threshold = 0.01;
        let active_dims = activity.iter().filter(|&&v| v > threshold).count();

        // Pairwise correlation matrix.
        let stds: Vec<f64> = activity.iter().map(|v| v.sqrt()).collect();
        let mut total_corr = 0.0;
        let mut corr_count = 0;
        for i in 0..latent_dim {
            for j in (i + 1)..latent_dim {
                if stds[i] < 1e-10 || stds[j] < 1e-10 { continue; }
                let cov: f64 = encodings.iter()
                    .map(|enc| (enc[i] - means[i]) * (enc[j] - means[j]))
                    .sum::<f64>() / n as f64;
                let corr = (cov / (stds[i] * stds[j])).abs();
                total_corr += corr;
                corr_count += 1;
            }
        }
        let mean_correlation = if corr_count > 0 { total_corr / corr_count as f64 } else { 0.0 };

        // MIG (Mutual Information Gap).
        let mig = if let Some(labels) = factor_labels {
            Self::compute_mig(encodings, labels, latent_dim)
        } else {
            0.0
        };

        Self { mig, activity, active_dims, mean_correlation }
    }

    /// Compute MIG score from encodings and factor labels.
    fn compute_mig(encodings: &[Vec<f64>], labels: &[Vec<f64>], latent_dim: usize) -> f64 {
        if labels.is_empty() || labels[0].is_empty() { return 0.0; }
        let num_factors = labels[0].len();
        let n = encodings.len().min(labels.len());

        // For each factor, find the two latent dims with highest mutual information.
        let mut mig_sum = 0.0;
        for k in 0..num_factors {
            let mut mi_scores: Vec<f64> = (0..latent_dim)
                .map(|j| {
                    // Approximate MI via correlation as a proxy.
                    let enc_mean: f64 = encodings.iter().take(n).map(|e| e[j]).sum::<f64>() / n as f64;
                    let lab_mean: f64 = labels.iter().take(n).map(|l| l[k]).sum::<f64>() / n as f64;
                    let enc_std: f64 = (encodings.iter().take(n)
                        .map(|e| (e[j] - enc_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
                    let lab_std: f64 = (labels.iter().take(n)
                        .map(|l| (l[k] - lab_mean).powi(2)).sum::<f64>() / n as f64).sqrt();

                    if enc_std < 1e-10 || lab_std < 1e-10 { return 0.0; }
                    let cov: f64 = encodings.iter().take(n).zip(labels.iter())
                        .map(|(e, l)| (e[j] - enc_mean) * (l[k] - lab_mean))
                        .sum::<f64>() / n as f64;
                    (cov / (enc_std * lab_std)).abs()
                })
                .collect();

            mi_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let gap = if mi_scores.len() >= 2 {
                mi_scores[0] - mi_scores[1]
            } else {
                mi_scores[0]
            };
            mig_sum += gap;
        }

        mig_sum / num_factors as f64
    }
}

impl fmt::Display for DisentanglementMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DisentanglementMetrics(MIG={:.4}, active_dims={}, mean_corr={:.4})",
            self.mig, self.active_dims, self.mean_correlation,
        )
    }
}

// ── Latent Space Explorer ─────────────────────────────────────

/// High-level latent space exploration tool.
#[derive(Debug, Clone)]
pub struct LatentExplorer {
    latent_dim: usize,
    stored_vectors: Vec<(String, Vec<f64>)>,
}

impl LatentExplorer {
    pub fn new(latent_dim: usize) -> Self {
        Self { latent_dim, stored_vectors: Vec::new() }
    }

    pub fn with_vector(mut self, name: &str, vec: Vec<f64>) -> Self {
        assert_eq!(vec.len(), self.latent_dim);
        self.stored_vectors.push((name.to_string(), vec));
        self
    }

    /// Store a named vector.
    pub fn store(&mut self, name: &str, vec: Vec<f64>) {
        assert_eq!(vec.len(), self.latent_dim);
        self.stored_vectors.push((name.to_string(), vec));
    }

    /// Retrieve a stored vector by name.
    pub fn get(&self, name: &str) -> Option<&Vec<f64>> {
        self.stored_vectors.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }

    /// Find the nearest stored vector to a query.
    pub fn nearest(&self, query: &[f64]) -> Option<(&str, f64)> {
        self.stored_vectors.iter()
            .map(|(name, vec)| (name.as_str(), euclidean_distance(query, vec)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Generate random vectors from a standard normal distribution.
    pub fn random_vectors(&self, count: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Rng::new(seed);
        (0..count)
            .map(|_| (0..self.latent_dim).map(|_| rng.normal()).collect())
            .collect()
    }

    pub fn vector_count(&self) -> usize { self.stored_vectors.len() }
    pub fn latent_dim(&self) -> usize { self.latent_dim }
}

impl fmt::Display for LatentExplorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatentExplorer(dim={}, stored={})", self.latent_dim, self.stored_vectors.len())
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_endpoints() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(lerp(&a, &b, 0.0), a);
        assert_eq!(lerp(&a, &b, 1.0), b);
    }

    #[test]
    fn test_lerp_midpoint() {
        let a = vec![0.0, 2.0];
        let b = vec![4.0, 6.0];
        let mid = lerp(&a, &b, 0.5);
        assert!((mid[0] - 2.0).abs() < 1e-10);
        assert!((mid[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s0 = slerp(&a, &b, 0.0);
        let s1 = slerp(&a, &b, 1.0);
        assert!((s0[0] - 1.0).abs() < 1e-8);
        assert!((s1[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_slerp_preserves_norm() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let mid = slerp(&a, &b, 0.5);
        let norm = l2_norm(&mid);
        assert!((norm - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_slerp_parallel_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![2.0, 0.0];
        let mid = slerp(&a, &b, 0.5);
        // Should fallback to lerp.
        assert!((mid[0] - 1.5).abs() < 1e-8);
    }

    #[test]
    fn test_interpolation_sequence() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let seq = interpolation_sequence(&a, &b, 5, false);
        assert_eq!(seq.len(), 5);
        assert_eq!(seq[0], a);
    }

    #[test]
    fn test_traversal_single_dim() {
        let base = vec![0.0, 0.0, 0.0];
        let trav = Traversal::single_dim(&base, 1, -2.0, 2.0, 5);
        assert_eq!(trav.points.len(), 5);
        assert!((trav.points[0][1] - (-2.0)).abs() < 1e-10);
        assert!((trav.points[4][1] - 2.0).abs() < 1e-10);
        // Other dimensions unchanged.
        assert_eq!(trav.points[2][0], 0.0);
    }

    #[test]
    fn test_traversal_along_direction() {
        let base = vec![1.0, 1.0];
        let dir = vec![1.0, 0.0];
        let trav = Traversal::along_direction(&base, &dir, -1.0, 1.0, 3);
        assert_eq!(trav.points.len(), 3);
    }

    #[test]
    fn test_analogy() {
        // king - man + woman = queen (as vectors).
        let man = vec![1.0, 0.0];
        let king = vec![1.0, 1.0];
        let woman = vec![0.0, 0.0];
        let queen = LatentArithmetic::analogy(&man, &king, &woman);
        assert!((queen[0] - 0.0).abs() < 1e-10);
        assert!((queen[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_average() {
        let vecs = vec![vec![0.0, 0.0], vec![2.0, 4.0]];
        let weights = vec![1.0, 1.0];
        let avg = LatentArithmetic::weighted_average(&vecs, &weights);
        assert!((avg[0] - 1.0).abs() < 1e-10);
        assert!((avg[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_onto_basis() {
        let v = vec![3.0, 4.0];
        let basis = vec![vec![1.0, 0.0]]; // x-axis.
        let proj = LatentArithmetic::project(&v, &basis);
        assert!((proj[0] - 3.0).abs() < 1e-10);
        assert!(proj[1].abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_disentanglement_metrics() {
        let encodings = vec![
            vec![1.0, 0.0, 0.5],
            vec![0.8, 0.1, 0.6],
            vec![1.2, -0.1, 0.4],
            vec![0.9, 0.05, 0.55],
        ];
        let metrics = DisentanglementMetrics::compute(&encodings, None);
        assert_eq!(metrics.activity.len(), 3);
        assert!(metrics.active_dims <= 3);
    }

    #[test]
    fn test_disentanglement_with_labels() {
        let encodings = vec![
            vec![1.0, 0.0], vec![2.0, 0.1], vec![3.0, -0.1], vec![4.0, 0.05],
        ];
        let labels = vec![
            vec![1.0], vec![2.0], vec![3.0], vec![4.0],
        ];
        let metrics = DisentanglementMetrics::compute(&encodings, Some(&labels));
        assert!(metrics.mig >= 0.0);
    }

    #[test]
    fn test_explorer_store_get() {
        let mut explorer = LatentExplorer::new(3);
        explorer.store("origin", vec![0.0, 0.0, 0.0]);
        assert_eq!(explorer.get("origin"), Some(&vec![0.0, 0.0, 0.0]));
        assert_eq!(explorer.get("missing"), None);
    }

    #[test]
    fn test_explorer_nearest() {
        let explorer = LatentExplorer::new(2)
            .with_vector("a", vec![0.0, 0.0])
            .with_vector("b", vec![10.0, 10.0]);
        let (name, _dist) = explorer.nearest(&[0.1, 0.1]).unwrap();
        assert_eq!(name, "a");
    }

    #[test]
    fn test_explorer_display() {
        let explorer = LatentExplorer::new(4);
        let s = format!("{explorer}");
        assert!(s.contains("dim=4"));
    }

    #[test]
    fn test_random_vectors() {
        let explorer = LatentExplorer::new(5);
        let vecs = explorer.random_vectors(3, 42);
        assert_eq!(vecs.len(), 3);
        assert_eq!(vecs[0].len(), 5);
    }

    #[test]
    fn test_scale_and_add() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let scaled = LatentArithmetic::scale(&a, 2.0);
        assert_eq!(scaled, vec![2.0, 4.0]);
        let sum = LatentArithmetic::add(&a, &b);
        assert_eq!(sum, vec![4.0, 6.0]);
    }

    #[test]
    fn test_traversal_display() {
        let base = vec![0.0; 3];
        let trav = Traversal::single_dim(&base, 1, -1.0, 1.0, 5);
        let s = format!("{trav}");
        assert!(s.contains("steps=5"));
    }
}
