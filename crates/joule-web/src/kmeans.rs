//! K-Means Clustering — Lloyd's algorithm with K-means++ initialization,
//! convergence detection, silhouette score, inertia tracking, and elbow
//! method helper.
//!
//! Pure Rust — no external ML dependencies.

use std::fmt;

// ── KMeans Configuration ────────────────────────────────────────

/// Initialization strategy for centroids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitMethod {
    /// Random selection of k data points (deterministic with seed).
    Random,
    /// K-means++ initialization for better spread.
    KMeansPlusPlus,
}

/// Configuration for K-Means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters.
    pub k: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on centroid movement.
    pub tol: f64,
    /// Initialization method.
    pub init: InitMethod,
    /// Seed for deterministic initialization.
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 3,
            max_iter: 300,
            tol: 1e-4,
            init: InitMethod::KMeansPlusPlus,
            seed: 42,
        }
    }
}

// ── KMeans Result ───────────────────────────────────────────────

/// Result of K-Means clustering.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Final centroid positions: centroids[cluster][dimension].
    pub centroids: Vec<Vec<f64>>,
    /// Cluster assignments for each data point.
    pub assignments: Vec<usize>,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Inertia (sum of squared distances to nearest centroid).
    pub inertia: f64,
    /// Whether the algorithm converged within max_iter.
    pub converged: bool,
    /// Inertia history across iterations.
    pub inertia_history: Vec<f64>,
}

impl fmt::Display for KMeansResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KMeansResult(k={}, iterations={}, inertia={:.4}, converged={})",
            self.centroids.len(),
            self.iterations,
            self.inertia,
            self.converged
        )
    }
}

// ── Simple PRNG ─────────────────────────────────────────────────

/// Lightweight xorshift64 PRNG for deterministic initialization.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

// ── Distance Helpers ────────────────────────────────────────────

fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean(a, b).sqrt()
}

// ── KMeans Core ─────────────────────────────────────────────────

/// Run K-Means clustering on the given data.
pub fn kmeans(data: &[Vec<f64>], config: &KMeansConfig) -> KMeansResult {
    assert!(!data.is_empty(), "empty dataset");
    assert!(config.k > 0, "k must be positive");
    assert!(config.k <= data.len(), "k cannot exceed number of data points");

    let dims = data[0].len();
    let mut rng = Rng::new(config.seed);

    // Initialize centroids
    let mut centroids = match config.init {
        InitMethod::Random => init_random(data, config.k, &mut rng),
        InitMethod::KMeansPlusPlus => init_kmeanspp(data, config.k, &mut rng),
    };

    let mut assignments = vec![0usize; data.len()];
    let mut inertia_history = Vec::with_capacity(config.max_iter);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        // Assignment step
        for (i, point) in data.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f64::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = squared_euclidean(point, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }

        // Compute inertia
        let inertia = compute_inertia(data, &assignments, &centroids);
        inertia_history.push(inertia);

        // Update step
        let mut new_centroids = vec![vec![0.0; dims]; config.k];
        let mut counts = vec![0usize; config.k];
        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, val) in point.iter().enumerate() {
                new_centroids[c][j] += val;
            }
        }
        for c in 0..config.k {
            if counts[c] > 0 {
                for j in 0..dims {
                    new_centroids[c][j] /= counts[c] as f64;
                }
            } else {
                // Reinitialize empty clusters to a random data point
                let idx = rng.next_usize(data.len());
                new_centroids[c] = data[idx].clone();
            }
        }

        // Check convergence
        let max_shift: f64 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| euclidean(old, new))
            .fold(0.0_f64, f64::max);

        centroids = new_centroids;
        iterations = iter + 1;

        if max_shift < config.tol {
            converged = true;
            break;
        }
    }

    let inertia = compute_inertia(data, &assignments, &centroids);

    KMeansResult {
        centroids,
        assignments,
        iterations,
        inertia,
        converged,
        inertia_history,
    }
}

fn init_random(data: &[Vec<f64>], k: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut chosen = Vec::with_capacity(k);
    let mut used = vec![false; data.len()];
    for _ in 0..k {
        let mut idx = rng.next_usize(data.len());
        // Avoid duplicates
        let mut attempts = 0;
        while used[idx] && attempts < data.len() {
            idx = (idx + 1) % data.len();
            attempts += 1;
        }
        used[idx] = true;
        chosen.push(data[idx].clone());
    }
    chosen
}

fn init_kmeanspp(data: &[Vec<f64>], k: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut centroids = Vec::with_capacity(k);

    // First centroid: random
    let first = rng.next_usize(data.len());
    centroids.push(data[first].clone());

    for _ in 1..k {
        // Compute D(x)^2 for each point
        let mut dists: Vec<f64> = data
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|c| squared_euclidean(point, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        // Normalize to create probability distribution
        let total: f64 = dists.iter().sum();
        if total > 0.0 {
            for d in &mut dists {
                *d /= total;
            }
        }

        // Sample proportional to D(x)^2
        let r = rng.next_f64();
        let mut cumulative = 0.0;
        let mut chosen = data.len() - 1;
        for (i, d) in dists.iter().enumerate() {
            cumulative += d;
            if cumulative >= r {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }
    centroids
}

fn compute_inertia(data: &[Vec<f64>], assignments: &[usize], centroids: &[Vec<f64>]) -> f64 {
    data.iter()
        .zip(assignments.iter())
        .map(|(point, &c)| squared_euclidean(point, &centroids[c]))
        .sum()
}

// ── Cluster Assignment ──────────────────────────────────────────

/// Assign new data points to the nearest centroid.
pub fn assign_clusters(data: &[Vec<f64>], centroids: &[Vec<f64>]) -> Vec<usize> {
    data.iter()
        .map(|point| {
            centroids
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    squared_euclidean(point, a.1)
                        .partial_cmp(&squared_euclidean(point, b.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

// ── Silhouette Score ────────────────────────────────────────────

/// Compute the silhouette score for clustered data.
///
/// Returns the mean silhouette coefficient across all samples.
/// Range: [-1, 1], where 1 = well-clustered, 0 = overlapping, -1 = mis-clustered.
pub fn silhouette_score(data: &[Vec<f64>], assignments: &[usize]) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }
    let k = assignments.iter().copied().max().unwrap_or(0) + 1;
    if k <= 1 {
        return 0.0;
    }

    let n = data.len();
    let mut total_silhouette = 0.0;

    for i in 0..n {
        let ci = assignments[i];

        // a(i) = mean distance to points in same cluster
        let mut same_count = 0usize;
        let mut same_dist = 0.0;
        for j in 0..n {
            if j != i && assignments[j] == ci {
                same_dist += euclidean(&data[i], &data[j]);
                same_count += 1;
            }
        }
        let a = if same_count > 0 { same_dist / same_count as f64 } else { 0.0 };

        // b(i) = min mean distance to points in other clusters
        let mut b = f64::INFINITY;
        for cluster in 0..k {
            if cluster == ci {
                continue;
            }
            let mut cluster_dist = 0.0;
            let mut cluster_count = 0usize;
            for j in 0..n {
                if assignments[j] == cluster {
                    cluster_dist += euclidean(&data[i], &data[j]);
                    cluster_count += 1;
                }
            }
            if cluster_count > 0 {
                let mean_dist = cluster_dist / cluster_count as f64;
                if mean_dist < b {
                    b = mean_dist;
                }
            }
        }

        let s = if a.max(b) > 0.0 { (b - a) / a.max(b) } else { 0.0 };
        total_silhouette += s;
    }

    total_silhouette / n as f64
}

// ── Elbow Method Helper ─────────────────────────────────────────

/// Run K-Means for multiple k values and return (k, inertia) pairs.
///
/// Useful for the elbow method to select optimal k.
pub fn elbow_method(
    data: &[Vec<f64>],
    k_range: std::ops::RangeInclusive<usize>,
    max_iter: usize,
    seed: u64,
) -> Vec<(usize, f64)> {
    k_range
        .filter(|k| *k > 0 && *k <= data.len())
        .map(|k| {
            let config = KMeansConfig {
                k,
                max_iter,
                seed,
                ..Default::default()
            };
            let result = kmeans(data, &config);
            (k, result.inertia)
        })
        .collect()
}

/// Find the "elbow" point in an inertia curve using the maximum curvature method.
/// Returns the index of the best k in the provided (k, inertia) pairs.
pub fn find_elbow(curve: &[(usize, f64)]) -> Option<usize> {
    if curve.len() < 3 {
        return curve.first().map(|c| c.0);
    }

    // Use the max-distance-to-line heuristic
    let first = (curve[0].0 as f64, curve[0].1);
    let last = (curve[curve.len() - 1].0 as f64, curve[curve.len() - 1].1);

    let line_len = ((last.0 - first.0).powi(2) + (last.1 - first.1).powi(2)).sqrt();
    if line_len < f64::EPSILON {
        return Some(curve[0].0);
    }

    let mut max_dist = 0.0_f64;
    let mut best_idx = 0;

    for (idx, (k, inertia)) in curve.iter().enumerate() {
        let px = *k as f64;
        let py = *inertia;
        // Distance from point to line
        let dist = ((last.1 - first.1) * px - (last.0 - first.0) * py
            + last.0 * first.1 - last.1 * first.0)
            .abs()
            / line_len;
        if dist > max_dist {
            max_dist = dist;
            best_idx = idx;
        }
    }

    Some(curve[best_idx].0)
}

// ── Cluster Sizes ───────────────────────────────────────────────

/// Count the number of points in each cluster.
pub fn cluster_sizes(assignments: &[usize], k: usize) -> Vec<usize> {
    let mut counts = vec![0usize; k];
    for &a in assignments {
        if a < k {
            counts[a] += 1;
        }
    }
    counts
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_clusters() -> Vec<Vec<f64>> {
        vec![
            // Cluster 0: around (0, 0)
            vec![0.1, 0.1], vec![-0.1, 0.0], vec![0.0, -0.1], vec![0.2, 0.2],
            // Cluster 1: around (5, 5)
            vec![5.0, 5.0], vec![5.1, 4.9], vec![4.9, 5.1], vec![5.2, 5.2],
            // Cluster 2: around (10, 0)
            vec![10.0, 0.0], vec![10.1, 0.1], vec![9.9, -0.1], vec![10.2, 0.2],
        ]
    }

    #[test]
    fn test_kmeans_basic() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 3, seed: 42, ..Default::default() };
        let result = kmeans(&data, &config);
        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.assignments.len(), 12);
        assert!(result.inertia < 5.0);
    }

    #[test]
    fn test_kmeans_convergence() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 3, seed: 42, ..Default::default() };
        let result = kmeans(&data, &config);
        assert!(result.converged);
        assert!(result.iterations < config.max_iter);
    }

    #[test]
    fn test_kmeans_random_init() {
        let data = simple_clusters();
        let config = KMeansConfig {
            k: 3,
            init: InitMethod::Random,
            seed: 42,
            ..Default::default()
        };
        let result = kmeans(&data, &config);
        assert_eq!(result.centroids.len(), 3);
    }

    #[test]
    fn test_kmeans_k1() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 1, seed: 42, ..Default::default() };
        let result = kmeans(&data, &config);
        assert_eq!(result.centroids.len(), 1);
        // All points assigned to cluster 0
        assert!(result.assignments.iter().all(|a| *a == 0));
    }

    #[test]
    fn test_correct_cluster_separation() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 3, seed: 42, ..Default::default() };
        let result = kmeans(&data, &config);

        // Points 0..4 should be in one cluster, 4..8 in another, 8..12 in third
        let c0 = result.assignments[0];
        let c1 = result.assignments[4];
        let c2 = result.assignments[8];
        assert_ne!(c0, c1);
        assert_ne!(c1, c2);
        assert_ne!(c0, c2);

        // All points in each group should share the same cluster
        for i in 0..4 {
            assert_eq!(result.assignments[i], c0);
        }
        for i in 4..8 {
            assert_eq!(result.assignments[i], c1);
        }
        for i in 8..12 {
            assert_eq!(result.assignments[i], c2);
        }
    }

    #[test]
    fn test_inertia_history() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 3, seed: 42, ..Default::default() };
        let result = kmeans(&data, &config);
        assert!(!result.inertia_history.is_empty());
        // Inertia should generally decrease
        if result.inertia_history.len() >= 2 {
            assert!(result.inertia_history.last().unwrap() <= result.inertia_history.first().unwrap());
        }
    }

    #[test]
    fn test_silhouette_score_good() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 3, seed: 42, ..Default::default() };
        let result = kmeans(&data, &config);
        let score = silhouette_score(&data, &result.assignments);
        // Well-separated clusters should have high silhouette score
        assert!(score > 0.5);
    }

    #[test]
    fn test_silhouette_score_single_cluster() {
        let data = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let assignments = vec![0, 0];
        let score = silhouette_score(&data, &assignments);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_assign_clusters() {
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let points = vec![vec![1.0, 1.0], vec![9.0, 9.0], vec![0.5, 0.5]];
        let assignments = assign_clusters(&points, &centroids);
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 1);
        assert_eq!(assignments[2], 0);
    }

    #[test]
    fn test_elbow_method() {
        let data = simple_clusters();
        let curve = elbow_method(&data, 1..=5, 100, 42);
        assert_eq!(curve.len(), 5);
        // Inertia should decrease as k increases
        for window in curve.windows(2) {
            assert!(window[1].1 <= window[0].1 + 1e-10);
        }
    }

    #[test]
    fn test_find_elbow() {
        let curve = vec![
            (1, 100.0), (2, 40.0), (3, 10.0), (4, 8.0), (5, 7.0),
        ];
        let elbow = find_elbow(&curve);
        assert!(elbow.is_some());
        // The elbow should be around k=3
        let k = elbow.unwrap();
        assert!(k >= 2 && k <= 4);
    }

    #[test]
    fn test_find_elbow_short() {
        let curve = vec![(1, 100.0)];
        let elbow = find_elbow(&curve);
        assert_eq!(elbow, Some(1));
    }

    #[test]
    fn test_cluster_sizes() {
        let assignments = vec![0, 1, 0, 2, 1, 0];
        let sizes = cluster_sizes(&assignments, 3);
        assert_eq!(sizes, vec![3, 2, 1]);
    }

    #[test]
    fn test_kmeans_display() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 3, seed: 42, ..Default::default() };
        let result = kmeans(&data, &config);
        let s = format!("{}", result);
        assert!(s.contains("KMeansResult"));
        assert!(s.contains("k=3"));
    }

    #[test]
    fn test_kmeans_deterministic() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 3, seed: 123, ..Default::default() };
        let r1 = kmeans(&data, &config);
        let r2 = kmeans(&data, &config);
        assert_eq!(r1.assignments, r2.assignments);
        assert!((r1.inertia - r2.inertia).abs() < 1e-10);
    }

    #[test]
    fn test_kmeans_max_iter_limit() {
        let data = simple_clusters();
        let config = KMeansConfig { k: 3, max_iter: 1, seed: 42, ..Default::default() };
        let result = kmeans(&data, &config);
        assert!(result.iterations <= 1);
    }
}
