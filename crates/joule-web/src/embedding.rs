//! Embedding vectors, distance metrics, kNN search, and quantization.
//!
//! Dense f64 embedding operations: cosine similarity, euclidean distance,
//! dot product, normalization, brute-force kNN, random projection for
//! dimensionality reduction, centroid computation, and product quantization.

// ── EmbeddingVector ─────────────────────────────────────────────

/// A dense embedding vector.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingVector {
    pub data: Vec<f64>,
}

impl EmbeddingVector {
    /// Create an embedding from raw values.
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// Zero vector of given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self { data: vec![0.0; dim] }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// L2 norm (magnitude).
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Normalize to unit vector. Returns zero vector if norm is near zero.
    pub fn normalize(&self) -> EmbeddingVector {
        let n = self.norm();
        if n < 1e-12 {
            return EmbeddingVector::zeros(self.dim());
        }
        EmbeddingVector {
            data: self.data.iter().map(|v| v / n).collect(),
        }
    }

    /// Dot product with another vector.
    pub fn dot(&self, other: &EmbeddingVector) -> f64 {
        assert_eq!(self.dim(), other.dim(), "dimension mismatch");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cosine similarity with another vector. Range: [-1, 1].
    pub fn cosine_similarity(&self, other: &EmbeddingVector) -> f64 {
        let na = self.norm();
        let nb = other.norm();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        self.dot(other) / (na * nb)
    }

    /// Euclidean (L2) distance to another vector.
    pub fn euclidean_distance(&self, other: &EmbeddingVector) -> f64 {
        assert_eq!(self.dim(), other.dim(), "dimension mismatch");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Element-wise addition.
    pub fn add(&self, other: &EmbeddingVector) -> EmbeddingVector {
        assert_eq!(self.dim(), other.dim());
        EmbeddingVector {
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect(),
        }
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> EmbeddingVector {
        EmbeddingVector {
            data: self.data.iter().map(|v| v * s).collect(),
        }
    }
}

// ── Brute-force kNN index ───────────────────────────────────────

/// Distance metric for kNN search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    DotProduct,
}

/// A brute-force embedding index for kNN search.
#[derive(Debug, Clone)]
pub struct EmbeddingIndex {
    pub vectors: Vec<EmbeddingVector>,
    pub metric: DistanceMetric,
}

impl EmbeddingIndex {
    /// Create an empty index.
    pub fn new(metric: DistanceMetric) -> Self {
        Self { vectors: Vec::new(), metric }
    }

    /// Add a vector to the index.
    pub fn add(&mut self, vec: EmbeddingVector) {
        self.vectors.push(vec);
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Compute distance/similarity between two vectors using the index metric.
    fn distance(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => a.euclidean_distance(b),
            DistanceMetric::Cosine => 1.0 - a.cosine_similarity(b), // distance = 1 - similarity
            DistanceMetric::DotProduct => -a.dot(b), // negate so lower = more similar
        }
    }

    /// Find the k nearest neighbors to `query`. Returns (index, distance) pairs
    /// sorted by distance (ascending).
    pub fn knn(&self, query: &EmbeddingVector, k: usize) -> Vec<(usize, f64)> {
        let mut dists: Vec<(usize, f64)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, self.distance(query, v)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }
}

// ── Centroid computation ────────────────────────────────────────

/// Compute the centroid (mean) of a set of embedding vectors.
pub fn centroid(vectors: &[EmbeddingVector]) -> EmbeddingVector {
    assert!(!vectors.is_empty(), "cannot compute centroid of empty set");
    let dim = vectors[0].dim();
    let mut sum = vec![0.0; dim];
    for v in vectors {
        assert_eq!(v.dim(), dim, "dimension mismatch");
        for (i, val) in v.data.iter().enumerate() {
            sum[i] += val;
        }
    }
    let n = vectors.len() as f64;
    EmbeddingVector {
        data: sum.into_iter().map(|s| s / n).collect(),
    }
}

// ── Random projection ───────────────────────────────────────────

/// A random projection matrix for dimensionality reduction.
///
/// Projects from `input_dim` to `output_dim` using a fixed seed
/// for reproducibility (simple LCG-based pseudo-random).
#[derive(Debug, Clone)]
pub struct RandomProjection {
    /// Projection matrix, row-major: [output_dim × input_dim].
    pub matrix: Vec<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl RandomProjection {
    /// Create a random projection matrix with a given seed.
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let n = output_dim * input_dim;
        let mut matrix = Vec::with_capacity(n);
        let scale = 1.0 / (output_dim as f64).sqrt();

        // Simple LCG for reproducible "random" values.
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to {-1, +1} with equal probability.
            let val = if (state >> 33) & 1 == 0 { scale } else { -scale };
            matrix.push(val);
        }

        Self { matrix, input_dim, output_dim }
    }

    /// Project a vector to the lower dimension.
    pub fn project(&self, vec: &EmbeddingVector) -> EmbeddingVector {
        assert_eq!(vec.dim(), self.input_dim, "input dimension mismatch");
        let mut result = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut sum = 0.0;
            for j in 0..self.input_dim {
                sum += self.matrix[i * self.input_dim + j] * vec.data[j];
            }
            result[i] = sum;
        }
        EmbeddingVector::new(result)
    }
}

// ── Product quantization ────────────────────────────────────────

/// Product quantization: split a vector into subvectors and assign each
/// to the nearest codebook entry.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Number of subvector segments.
    pub num_segments: usize,
    /// Dimension of each segment.
    pub segment_dim: usize,
    /// Codebook entries per segment: `[num_segments][codebook_size][segment_dim]`.
    pub codebooks: Vec<Vec<Vec<f64>>>,
}

impl ProductQuantizer {
    /// Create a product quantizer with given codebooks.
    pub fn new(codebooks: Vec<Vec<Vec<f64>>>) -> Self {
        let num_segments = codebooks.len();
        let segment_dim = if num_segments > 0 && !codebooks[0].is_empty() {
            codebooks[0][0].len()
        } else {
            0
        };
        Self { num_segments, segment_dim, codebooks }
    }

    /// Codebook size (entries per segment).
    pub fn codebook_size(&self) -> usize {
        if self.codebooks.is_empty() {
            0
        } else {
            self.codebooks[0].len()
        }
    }

    /// Total input dimension expected.
    pub fn total_dim(&self) -> usize {
        self.num_segments * self.segment_dim
    }

    /// Quantize a vector: returns one codebook index per segment.
    pub fn quantize(&self, vec: &EmbeddingVector) -> Vec<usize> {
        assert_eq!(vec.dim(), self.total_dim(), "dimension mismatch");
        let mut codes = Vec::with_capacity(self.num_segments);
        for seg in 0..self.num_segments {
            let start = seg * self.segment_dim;
            let end = start + self.segment_dim;
            let subvec = &vec.data[start..end];

            let mut best_idx = 0;
            let mut best_dist = f64::MAX;
            for (i, entry) in self.codebooks[seg].iter().enumerate() {
                let dist: f64 = subvec
                    .iter()
                    .zip(entry.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            codes.push(best_idx);
        }
        codes
    }

    /// Reconstruct an approximate vector from quantization codes.
    pub fn reconstruct(&self, codes: &[usize]) -> EmbeddingVector {
        assert_eq!(codes.len(), self.num_segments, "code length mismatch");
        let mut data = Vec::with_capacity(self.total_dim());
        for (seg, &code) in codes.iter().enumerate() {
            data.extend_from_slice(&self.codebooks[seg][code]);
        }
        EmbeddingVector::new(data)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_embedding_norm() {
        let v = EmbeddingVector::new(vec![3.0, 4.0]);
        assert!(approx_eq(v.norm(), 5.0));
    }

    #[test]
    fn test_normalize() {
        let v = EmbeddingVector::new(vec![3.0, 4.0]);
        let n = v.normalize();
        assert!(approx_eq(n.norm(), 1.0));
        assert!(approx_eq(n.data[0], 0.6));
        assert!(approx_eq(n.data[1], 0.8));
    }

    #[test]
    fn test_normalize_zero() {
        let v = EmbeddingVector::zeros(3);
        let n = v.normalize();
        assert!(n.data.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_dot_product() {
        let a = EmbeddingVector::new(vec![1.0, 2.0, 3.0]);
        let b = EmbeddingVector::new(vec![4.0, 5.0, 6.0]);
        assert!(approx_eq(a.dot(&b), 32.0));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = EmbeddingVector::new(vec![1.0, 0.0]);
        let b = EmbeddingVector::new(vec![1.0, 0.0]);
        assert!(approx_eq(a.cosine_similarity(&b), 1.0));

        let c = EmbeddingVector::new(vec![0.0, 1.0]);
        assert!(approx_eq(a.cosine_similarity(&c), 0.0));

        let d = EmbeddingVector::new(vec![-1.0, 0.0]);
        assert!(approx_eq(a.cosine_similarity(&d), -1.0));
    }

    #[test]
    fn test_euclidean_distance() {
        let a = EmbeddingVector::new(vec![0.0, 0.0]);
        let b = EmbeddingVector::new(vec![3.0, 4.0]);
        assert!(approx_eq(a.euclidean_distance(&b), 5.0));
    }

    #[test]
    fn test_knn_euclidean() {
        let mut idx = EmbeddingIndex::new(DistanceMetric::Euclidean);
        idx.add(EmbeddingVector::new(vec![0.0, 0.0]));
        idx.add(EmbeddingVector::new(vec![1.0, 0.0]));
        idx.add(EmbeddingVector::new(vec![10.0, 10.0]));

        let query = EmbeddingVector::new(vec![0.5, 0.0]);
        let results = idx.knn(&query, 2);
        assert_eq!(results.len(), 2);
        // Closest should be index 1 (dist 0.5) or index 0 (dist 0.5)
        assert!(results[0].0 == 0 || results[0].0 == 1);
    }

    #[test]
    fn test_knn_cosine() {
        let mut idx = EmbeddingIndex::new(DistanceMetric::Cosine);
        idx.add(EmbeddingVector::new(vec![1.0, 0.0]));
        idx.add(EmbeddingVector::new(vec![0.0, 1.0]));
        idx.add(EmbeddingVector::new(vec![1.0, 1.0]));

        let query = EmbeddingVector::new(vec![1.0, 0.0]);
        let results = idx.knn(&query, 1);
        assert_eq!(results[0].0, 0); // exact match is closest
    }

    #[test]
    fn test_centroid() {
        let vecs = vec![
            EmbeddingVector::new(vec![1.0, 2.0]),
            EmbeddingVector::new(vec![3.0, 4.0]),
            EmbeddingVector::new(vec![5.0, 6.0]),
        ];
        let c = centroid(&vecs);
        assert!(approx_eq(c.data[0], 3.0));
        assert!(approx_eq(c.data[1], 4.0));
    }

    #[test]
    fn test_random_projection() {
        let proj = RandomProjection::new(100, 10, 42);
        let v = EmbeddingVector::new(vec![1.0; 100]);
        let projected = proj.project(&v);
        assert_eq!(projected.dim(), 10);
        // Projected values should be non-zero (not all cancel out).
        assert!(projected.data.iter().any(|x| x.abs() > 1e-12));
    }

    #[test]
    fn test_random_projection_preserves_relative_distance() {
        let proj = RandomProjection::new(50, 10, 123);
        let a = EmbeddingVector::new(vec![1.0; 50]);
        let mut b_data = vec![1.0; 50];
        b_data[0] = 2.0; // slightly different
        let b = EmbeddingVector::new(b_data);
        let c = EmbeddingVector::new(vec![100.0; 50]); // very different

        let pa = proj.project(&a);
        let pb = proj.project(&b);
        let pc = proj.project(&c);

        // a-b should be closer than a-c.
        assert!(pa.euclidean_distance(&pb) < pa.euclidean_distance(&pc));
    }

    #[test]
    fn test_product_quantization() {
        // 2 segments, each dim=2, codebook size=2
        let codebooks = vec![
            // Segment 0 codebook.
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            // Segment 1 codebook.
            vec![vec![0.0, 0.0], vec![2.0, 2.0]],
        ];
        let pq = ProductQuantizer::new(codebooks);
        assert_eq!(pq.num_segments, 2);
        assert_eq!(pq.segment_dim, 2);
        assert_eq!(pq.codebook_size(), 2);
        assert_eq!(pq.total_dim(), 4);

        let v = EmbeddingVector::new(vec![0.9, 0.9, 1.8, 1.8]);
        let codes = pq.quantize(&v);
        assert_eq!(codes, vec![1, 1]); // nearest to [1,1] and [2,2]

        let recon = pq.reconstruct(&codes);
        assert_eq!(recon.data, vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_add_and_scale() {
        let a = EmbeddingVector::new(vec![1.0, 2.0]);
        let b = EmbeddingVector::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![4.0, 6.0]);

        let d = a.scale(2.0);
        assert_eq!(d.data, vec![2.0, 4.0]);
    }
}
