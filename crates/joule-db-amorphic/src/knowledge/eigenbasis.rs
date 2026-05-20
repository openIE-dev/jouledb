//! The Eigenbasis: 34 patterns × 16 dimensions = the 1KB brain.
//!
//! Ported from ai_graph_classification/formula.py.
//!
//! Any concept scored on 34 structural patterns gets a position in
//! 16-dimensional structural space. Two concepts are related if their
//! positions are close. No training. No embeddings. No neural network.
//!
//! The 34 patterns are the grammar of reality:
//! Transformation, Hierarchy, Emergence, Feedback, Cycle, Flow,
//! Containment, Signal, Selection, Replication, ...
//!
//! R(A, B) = cos(E*(A-μ), E*(B-μ))
//!
//! Where E is the 16×34 eigenbasis derived from SVD of the 215×34 score matrix.
//! Stable core: dimensions 1-8 (bootstrap validated, mean cosine 0.938-0.974).

/// The 34 structural patterns — the grammar of how things relate.
pub const PATTERN_NAMES: [&str; 34] = [
    "support_surface",
    "pulsation",
    "agent_action_instrument",
    "containment",
    "flow",
    "transformation",
    "hierarchy",
    "cycle",
    "emergence",
    "signal",
    "balance",
    "replication",
    "interface",
    "compression",
    "branching",
    "symmetry",
    "feedback",
    "gradient",
    "resonance",
    "selection",
    "binding",
    "recursion",
    "duality",
    "accumulation",
    "decay",
    "oscillation",
    "network",
    "barrier",
    "catalyst",
    "memory",
    "polarity",
    "resilience",
    "threshold",
    "representation",
];

/// Number of patterns.
pub const NUM_PATTERNS: usize = 34;

/// Number of eigendimensions (derived from 80% variance threshold).
pub const NUM_EIGENDIMS: usize = 16;

/// Stable core dimensions (bootstrap-validated).
pub const STABLE_CORE: usize = 8;

/// A concept's scores on 34 structural patterns.
#[derive(Clone, Debug)]
pub struct PatternScores {
    /// Scores in [0.0, 1.0] for each of the 34 patterns.
    /// 0.0 = no affinity. 1.0 = defining characteristic.
    pub scores: [f64; NUM_PATTERNS],
}

impl PatternScores {
    /// Create with all zeros (unknown concept).
    pub fn zeros() -> Self {
        Self {
            scores: [0.0; NUM_PATTERNS],
        }
    }

    /// Create from a sparse set of (pattern_name, score) pairs.
    pub fn from_sparse(entries: &[(&str, f64)]) -> Self {
        let mut scores = [0.0; NUM_PATTERNS];
        for (name, score) in entries {
            if let Some(idx) = pattern_index(name) {
                scores[idx] = *score;
            }
        }
        Self { scores }
    }

    /// Set a pattern score by name.
    pub fn set(&mut self, pattern: &str, score: f64) {
        if let Some(idx) = pattern_index(pattern) {
            self.scores[idx] = score;
        }
    }

    /// Get a pattern score by name.
    pub fn get(&self, pattern: &str) -> f64 {
        pattern_index(pattern)
            .map(|idx| self.scores[idx])
            .unwrap_or(0.0)
    }

    /// Number of non-zero scores.
    pub fn num_scored(&self) -> usize {
        self.scores.iter().filter(|&&s| s > 0.001).count()
    }
}

/// Get the index of a pattern by name.
pub fn pattern_index(name: &str) -> Option<usize> {
    let lower = name.to_lowercase().replace(' ', "_").replace('-', "_");
    // Strip "pattern." prefix if present
    let clean = lower.strip_prefix("pattern.").unwrap_or(&lower);
    PATTERN_NAMES.iter().position(|&p| p == clean)
}

/// The eigenbasis: projects 34-pattern scores into 16-dimensional structural space.
///
/// This is the 1KB brain. Derived from SVD of the 215×34 score matrix.
/// Loaded from the Python formula or computed fresh from a score matrix.
pub struct Eigenbasis {
    /// The eigenbasis matrix: NUM_EIGENDIMS × NUM_PATTERNS (row-major).
    /// E[i][j] = contribution of pattern j to eigendimension i.
    pub matrix: Vec<[f64; NUM_PATTERNS]>,
    /// Mean score vector (column means of the score matrix).
    pub mu: [f64; NUM_PATTERNS],
    /// Number of eigendimensions actually used.
    pub k: usize,
    /// Singular values (for variance analysis).
    pub sigma: Vec<f64>,
    /// Cumulative variance explained.
    pub cumvar: Vec<f64>,
}

impl Eigenbasis {
    /// Create from a raw score matrix (N concepts × 34 patterns).
    /// Performs SVD and extracts K dimensions at variance_threshold.
    pub fn from_scores(scores: &[PatternScores], variance_threshold: f64) -> Self {
        let n = scores.len();
        if n == 0 {
            return Self::empty();
        }

        // Compute column means
        let mut mu = [0.0; NUM_PATTERNS];
        for s in scores {
            for (i, &v) in s.scores.iter().enumerate() {
                mu[i] += v;
            }
        }
        for m in &mut mu {
            *m /= n as f64;
        }

        // Center the matrix
        let mut centered: Vec<[f64; NUM_PATTERNS]> = scores
            .iter()
            .map(|s| {
                let mut row = [0.0; NUM_PATTERNS];
                for (i, &v) in s.scores.iter().enumerate() {
                    row[i] = v - mu[i];
                }
                row
            })
            .collect();

        // Compute covariance matrix (34×34) for eigendecomposition
        // C = M^T * M / (n-1)
        let mut cov = [[0.0f64; NUM_PATTERNS]; NUM_PATTERNS];
        for row in &centered {
            for i in 0..NUM_PATTERNS {
                for j in 0..NUM_PATTERNS {
                    cov[i][j] += row[i] * row[j];
                }
            }
        }
        let divisor = if n > 1 { (n - 1) as f64 } else { 1.0 };
        for i in 0..NUM_PATTERNS {
            for j in 0..NUM_PATTERNS {
                cov[i][j] /= divisor;
            }
        }

        // Power iteration for top-K eigenvectors
        // (Full SVD would be better but this is dependency-free)
        let (eigenvectors, eigenvalues) = power_iteration_eigenvectors(&cov, NUM_EIGENDIMS);

        // Determine K from variance threshold
        let total_var: f64 = eigenvalues.iter().sum();
        let mut cumvar = Vec::new();
        let mut cumsum = 0.0;
        for &ev in &eigenvalues {
            cumsum += ev;
            cumvar.push(cumsum / total_var);
        }

        let k = cumvar
            .iter()
            .position(|&cv| cv >= variance_threshold)
            .map(|i| i + 1)
            .unwrap_or(eigenvectors.len());

        Self {
            matrix: eigenvectors,
            mu,
            k,
            sigma: eigenvalues.iter().map(|ev| ev.sqrt()).collect(),
            cumvar,
        }
    }

    /// Create empty (no data yet).
    pub fn empty() -> Self {
        Self {
            matrix: Vec::new(),
            mu: [0.0; NUM_PATTERNS],
            k: 0,
            sigma: Vec::new(),
            cumvar: Vec::new(),
        }
    }

    /// Load from pre-computed eigenbasis (e.g., from Python formula output).
    /// `rows` is K vectors of 34 values each.
    pub fn from_precomputed(rows: Vec<[f64; NUM_PATTERNS]>, mu: [f64; NUM_PATTERNS]) -> Self {
        let k = rows.len();
        Self {
            matrix: rows,
            mu,
            k,
            sigma: Vec::new(),
            cumvar: Vec::new(),
        }
    }

    /// Project a concept's pattern scores into structural space.
    /// Returns a K-dimensional position vector.
    pub fn project(&self, scores: &PatternScores) -> Vec<f64> {
        let mut centered = [0.0; NUM_PATTERNS];
        for i in 0..NUM_PATTERNS {
            centered[i] = scores.scores[i] - self.mu[i];
        }

        let mut position = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let mut dot = 0.0;
            for j in 0..NUM_PATTERNS {
                dot += self.matrix[i][j] * centered[j];
            }
            position.push(dot);
        }

        position
    }

    /// Structural similarity between two concepts.
    /// R(A, B) = cos(project(A), project(B))
    pub fn relate(&self, a: &PatternScores, b: &PatternScores) -> f64 {
        let va = self.project(a);
        let vb = self.project(b);
        cosine_similarity(&va, &vb)
    }

    /// Variance explained by K dimensions.
    pub fn variance_explained(&self) -> f64 {
        if self.k > 0 && !self.cumvar.is_empty() {
            self.cumvar[self.k.min(self.cumvar.len()) - 1]
        } else {
            0.0
        }
    }

    /// Size in bytes (the "1KB brain").
    pub fn size_bytes(&self) -> usize {
        // eigenbasis: k × 34 × 8 bytes + mu: 34 × 8 bytes
        self.k * NUM_PATTERNS * 8 + NUM_PATTERNS * 8
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a < 1e-15 || norm_b < 1e-15 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Power iteration to find top-K eigenvectors of a symmetric matrix.
/// Simple, dependency-free, O(K * N^2 * iterations).
fn power_iteration_eigenvectors(
    matrix: &[[f64; NUM_PATTERNS]; NUM_PATTERNS],
    k: usize,
) -> (Vec<[f64; NUM_PATTERNS]>, Vec<f64>) {
    let mut eigenvectors = Vec::with_capacity(k);
    let mut eigenvalues = Vec::with_capacity(k);

    // Working copy for deflation
    let mut work = *matrix;

    for _ in 0..k {
        // Initialize random vector
        let mut v = [0.0; NUM_PATTERNS];
        for (i, val) in v.iter_mut().enumerate() {
            *val = ((i as f64 * 2.718281828 + 1.0).sin()) * 1.0; // Deterministic "random"
        }

        // Power iteration: v = M*v / ||M*v|| repeated
        let mut eigenvalue = 0.0;
        for _ in 0..200 {
            let mut mv = [0.0; NUM_PATTERNS];
            for i in 0..NUM_PATTERNS {
                for j in 0..NUM_PATTERNS {
                    mv[i] += work[i][j] * v[j];
                }
            }

            // Normalize
            let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            eigenvalue = norm;
            for i in 0..NUM_PATTERNS {
                v[i] = mv[i] / norm;
            }
        }

        // Deflate: remove this eigenvector's contribution
        for i in 0..NUM_PATTERNS {
            for j in 0..NUM_PATTERNS {
                work[i][j] -= eigenvalue * v[i] * v[j];
            }
        }

        eigenvectors.push(v);
        eigenvalues.push(eigenvalue);
    }

    (eigenvectors, eigenvalues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_index() {
        assert_eq!(pattern_index("transformation"), Some(5));
        assert_eq!(pattern_index("pattern.hierarchy"), Some(6));
        assert_eq!(pattern_index("EMERGENCE"), Some(8));
        assert_eq!(pattern_index("nonexistent"), None);
    }

    #[test]
    fn test_pattern_scores_sparse() {
        let scores = PatternScores::from_sparse(&[
            ("transformation", 0.9),
            ("hierarchy", 0.7),
            ("emergence", 0.8),
        ]);
        assert!((scores.get("transformation") - 0.9).abs() < 0.001);
        assert!((scores.get("hierarchy") - 0.7).abs() < 0.001);
        assert_eq!(scores.num_scored(), 3);
    }

    #[test]
    fn test_structural_similarity() {
        // Build a small score matrix
        let cancer = PatternScores::from_sparse(&[
            ("replication", 0.9),
            ("feedback", 0.8),
            ("emergence", 0.7),
            ("selection", 0.6),
        ]);
        let war = PatternScores::from_sparse(&[
            ("replication", 0.7),
            ("feedback", 0.7),
            ("emergence", 0.8),
            ("selection", 0.5),
        ]);
        let lens = PatternScores::from_sparse(&[
            ("transformation", 0.9),
            ("interface", 0.8),
            ("signal", 0.7),
        ]);

        let scores = vec![cancer.clone(), war.clone(), lens.clone()];
        let basis = Eigenbasis::from_scores(&scores, 0.80);

        // Cancer and War should be more similar than Cancer and Lens
        let sim_cancer_war = basis.relate(&cancer, &war);
        let sim_cancer_lens = basis.relate(&cancer, &lens);

        assert!(
            sim_cancer_war > sim_cancer_lens,
            "cancer~war ({sim_cancer_war:.3}) should be > cancer~lens ({sim_cancer_lens:.3})"
        );
    }

    #[test]
    fn test_eigenbasis_size() {
        let scores: Vec<PatternScores> = (0..10)
            .map(|i| {
                PatternScores::from_sparse(&[
                    ("transformation", (i as f64) / 10.0),
                    ("hierarchy", (10 - i) as f64 / 10.0),
                ])
            })
            .collect();

        let basis = Eigenbasis::from_scores(&scores, 0.80);
        let size = basis.size_bytes();

        // Should be small — the "1KB brain"
        assert!(
            size < 5000,
            "eigenbasis should be small: {} bytes",
            size
        );
    }

    #[test]
    fn test_zero_shot_new_concept() {
        // Train on known concepts
        let dog = PatternScores::from_sparse(&[
            ("hierarchy", 0.5),
            ("replication", 0.3),
            ("cycle", 0.4),
            ("feedback", 0.3),
        ]);
        let cat = PatternScores::from_sparse(&[
            ("hierarchy", 0.5),
            ("replication", 0.3),
            ("cycle", 0.3),
            ("selection", 0.4),
        ]);
        let car = PatternScores::from_sparse(&[
            ("transformation", 0.8),
            ("flow", 0.7),
            ("containment", 0.6),
        ]);

        let basis = Eigenbasis::from_scores(&[dog.clone(), cat.clone(), car.clone()], 0.80);

        // Zero-shot: new concept "wolf" scored on same patterns
        let wolf = PatternScores::from_sparse(&[
            ("hierarchy", 0.6),
            ("replication", 0.3),
            ("cycle", 0.4),
            ("selection", 0.5),
        ]);

        let sim_wolf_dog = basis.relate(&wolf, &dog);
        let sim_wolf_car = basis.relate(&wolf, &car);

        assert!(
            sim_wolf_dog > sim_wolf_car,
            "wolf should be closer to dog ({sim_wolf_dog:.3}) than car ({sim_wolf_car:.3})"
        );
    }

    #[test]
    fn test_project_dimensions() {
        let scores: Vec<PatternScores> = (0..20)
            .map(|i| {
                let mut s = PatternScores::zeros();
                s.scores[i % NUM_PATTERNS] = 1.0;
                s
            })
            .collect();

        let basis = Eigenbasis::from_scores(&scores, 0.80);
        let pos = basis.project(&scores[0]);

        assert_eq!(pos.len(), basis.k);
        assert!(basis.k <= NUM_EIGENDIMS);
    }
}
