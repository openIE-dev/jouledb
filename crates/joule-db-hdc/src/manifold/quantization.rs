//! Vector quantization for memory-efficient approximate nearest neighbor search.
//!
//! - **ScalarQuantizer**: Maps f32 to u8 (4x memory compression), computes distances
//!   in quantized space with linear reconstruction.
//! - **ProductQuantizer**: Splits vectors into sub-vectors, learns codebooks via k-means,
//!   and uses lookup-table distance for fast approximate distance computation.

use super::hnsw::DistanceMetric;

// ==================== Scalar Quantizer ====================

/// Scalar Quantizer: maps each f32 dimension to u8 [0, 255].
///
/// Achieves 4x memory compression. Distance is computed on reconstructed
/// (dequantized) vectors for accuracy.
pub struct ScalarQuantizer {
    dimension: usize,
    /// Per-dimension min values (for rescaling)
    mins: Vec<f32>,
    /// Per-dimension scale factors: (max - min) / 255
    scales: Vec<f32>,
    /// Quantized vectors: id → u8 vector
    vectors: Vec<(String, Vec<u8>)>,
    metric: DistanceMetric,
    trained: bool,
}

impl ScalarQuantizer {
    /// Create a new ScalarQuantizer for the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self::with_metric(dimension, DistanceMetric::Euclidean)
    }

    /// Create a ScalarQuantizer with a specific distance metric.
    pub fn with_metric(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            dimension,
            mins: vec![0.0; dimension],
            scales: vec![1.0; dimension],
            vectors: Vec::new(),
            metric,
            trained: false,
        }
    }

    /// Train the quantizer by computing per-dimension min/max from training data.
    pub fn train(&mut self, vectors: &[(String, Vec<f32>)]) -> Result<(), String> {
        if vectors.is_empty() {
            self.trained = true;
            return Ok(());
        }

        let dim = self.dimension;
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];

        for (_, v) in vectors {
            if v.len() != dim {
                return Err(format!(
                    "Dimension mismatch: expected {}, got {}",
                    dim,
                    v.len()
                ));
            }
            for (j, &val) in v.iter().enumerate() {
                if val < mins[j] {
                    mins[j] = val;
                }
                if val > maxs[j] {
                    maxs[j] = val;
                }
            }
        }

        let mut scales = vec![1.0f32; dim];
        for j in 0..dim {
            let range = maxs[j] - mins[j];
            scales[j] = if range > 1e-10 { range / 255.0 } else { 1.0 };
        }

        self.mins = mins;
        self.scales = scales;

        // Quantize all training vectors
        self.vectors.clear();
        for (id, v) in vectors {
            let q = self.quantize(v);
            self.vectors.push((id.clone(), q));
        }

        self.trained = true;
        Ok(())
    }

    /// Quantize a single f32 vector to u8.
    pub fn quantize(&self, v: &[f32]) -> Vec<u8> {
        v.iter()
            .enumerate()
            .map(|(j, &val)| {
                let normalized = (val - self.mins[j]) / self.scales[j];
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    /// Dequantize a u8 vector back to f32.
    pub fn dequantize(&self, q: &[u8]) -> Vec<f32> {
        q.iter()
            .enumerate()
            .map(|(j, &val)| self.mins[j] + val as f32 * self.scales[j])
            .collect()
    }

    /// Insert a vector.
    pub fn insert(&mut self, id: String, point: Vec<f32>) -> Result<(), String> {
        if point.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                point.len()
            ));
        }
        if !self.trained {
            // Auto-train with a single vector to set mins/scales
            self.train(&[(id.clone(), point.clone())])?;
            return Ok(());
        }
        let q = self.quantize(&point);
        self.vectors.push((id, q));
        Ok(())
    }

    /// Query for k nearest neighbors using dequantized distance.
    pub fn query(&self, point: &[f32], k: usize) -> Vec<SQResult> {
        if point.len() != self.dimension || !self.trained {
            return Vec::new();
        }

        let mut candidates: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, (_, q))| {
                let reconstructed = self.dequantize(q);
                (
                    i,
                    super::ivf::compute_distance(&self.metric, point, &reconstructed),
                )
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .into_iter()
            .take(k)
            .map(|(i, dist)| SQResult {
                id: self.vectors[i].0.clone(),
                distance: dist,
            })
            .collect()
    }

    /// Get total vectors stored.
    pub fn size(&self) -> usize {
        self.vectors.len()
    }
    /// Get vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Check if quantizer is trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }
    /// Get configured distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

/// Scalar quantizer query result.
#[derive(Debug, Clone)]
pub struct SQResult {
    /// Point ID.
    pub id: String,
    /// Distance from query.
    pub distance: f32,
}

// ==================== Product Quantizer ====================

/// Product Quantizer: splits vectors into sub-vectors and encodes each
/// sub-vector as a codebook index (u8).
///
/// For a D-dimensional vector split into M sub-vectors, each sub-vector
/// is quantized to one of 256 centroids. Memory: D*4 bytes → M bytes.
pub struct ProductQuantizer {
    dimension: usize,
    /// Number of sub-quantizers (sub-vector segments)
    n_subquantizers: usize,
    /// Sub-vector dimension (dimension / n_subquantizers)
    sub_dim: usize,
    /// Codebooks: n_subquantizers x 256 x sub_dim
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Encoded vectors: id, codes (n_subquantizers u8 values)
    vectors: Vec<(String, Vec<u8>)>,
    metric: DistanceMetric,
    trained: bool,
}

impl ProductQuantizer {
    /// Create a new Product Quantizer.
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension (must be divisible by n_subquantizers)
    /// * `n_subquantizers` - Number of sub-vector segments (e.g., 8)
    pub fn new(dimension: usize, n_subquantizers: usize) -> Self {
        Self::with_metric(dimension, n_subquantizers, DistanceMetric::Euclidean)
    }

    /// Create a Product Quantizer with a specific distance metric.
    pub fn with_metric(dimension: usize, n_subquantizers: usize, metric: DistanceMetric) -> Self {
        let n_sub = n_subquantizers.max(1).min(dimension);
        let sub_dim = dimension / n_sub;
        Self {
            dimension,
            n_subquantizers: n_sub,
            sub_dim,
            codebooks: Vec::new(),
            vectors: Vec::new(),
            metric,
            trained: false,
        }
    }

    /// Train codebooks using k-means on each sub-vector segment.
    /// Each segment gets a codebook of up to 256 centroids.
    pub fn train(&mut self, vectors: &[(String, Vec<f32>)]) -> Result<(), String> {
        if vectors.is_empty() {
            self.trained = true;
            return Ok(());
        }

        let n_codes = 256.min(vectors.len()); // Can't have more codes than vectors
        let mut codebooks = Vec::with_capacity(self.n_subquantizers);

        for m in 0..self.n_subquantizers {
            let start = m * self.sub_dim;
            let end = start + self.sub_dim;

            // Extract sub-vectors for this segment
            let sub_vecs: Vec<Vec<f32>> = vectors
                .iter()
                .map(|(_, v)| v[start..end.min(v.len())].to_vec())
                .collect();

            // K-means on the sub-vectors
            let centroids = self.kmeans_sub(&sub_vecs, n_codes);
            codebooks.push(centroids);
        }

        self.codebooks = codebooks;

        // Encode all training vectors
        self.vectors.clear();
        for (id, v) in vectors {
            let codes = self.encode(v);
            self.vectors.push((id.clone(), codes));
        }

        self.trained = true;
        Ok(())
    }

    /// Encode a vector into PQ codes.
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.n_subquantizers);
        for m in 0..self.n_subquantizers {
            let start = m * self.sub_dim;
            let end = (start + self.sub_dim).min(v.len());
            let sub = &v[start..end];

            // Find nearest codebook entry
            let mut best_code = 0u8;
            let mut best_dist = f32::MAX;
            for (c, centroid) in self.codebooks[m].iter().enumerate() {
                let d = sub_distance(sub, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_code = c as u8;
                }
            }
            codes.push(best_code);
        }
        codes
    }

    /// Insert a vector.
    pub fn insert(&mut self, id: String, point: Vec<f32>) -> Result<(), String> {
        if point.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                point.len()
            ));
        }
        if !self.trained {
            self.train(&[(id.clone(), point.clone())])?;
            return Ok(());
        }
        let codes = self.encode(&point);
        self.vectors.push((id, codes));
        Ok(())
    }

    /// Query for k nearest neighbors using asymmetric distance computation.
    ///
    /// Precomputes a distance lookup table (query sub-vector → each codebook entry),
    /// then sums the lookup values for each database vector.
    pub fn query(&self, point: &[f32], k: usize) -> Vec<PQResult> {
        if point.len() != self.dimension || !self.trained {
            return Vec::new();
        }

        // Build distance lookup table: [m][code] = distance(query_sub_m, codebook[m][code])
        let mut dist_table =
            vec![vec![0.0f32; self.codebooks.first().map_or(0, |c| c.len())]; self.n_subquantizers];
        for m in 0..self.n_subquantizers {
            let start = m * self.sub_dim;
            let end = (start + self.sub_dim).min(point.len());
            let query_sub = &point[start..end];
            for (c, centroid) in self.codebooks[m].iter().enumerate() {
                dist_table[m][c] = sub_distance(query_sub, centroid);
            }
        }

        // Compute approximate distance for each stored vector using lookup table
        let mut candidates: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, (_, codes))| {
                let dist: f32 = codes
                    .iter()
                    .enumerate()
                    .map(|(m, &code)| {
                        dist_table
                            .get(m)
                            .and_then(|t| t.get(code as usize))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .sum();
                // For Euclidean: sum of squared sub-distances, take sqrt
                (i, dist.sqrt())
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .into_iter()
            .take(k)
            .map(|(i, dist)| PQResult {
                id: self.vectors[i].0.clone(),
                distance: dist,
            })
            .collect()
    }

    /// K-means on sub-vectors with up to 256 centroids, 15 iterations.
    fn kmeans_sub(&self, sub_vecs: &[Vec<f32>], n_codes: usize) -> Vec<Vec<f32>> {
        let n = sub_vecs.len();
        if n == 0 {
            return vec![vec![0.0; self.sub_dim]];
        }

        let k = n_codes.min(n);

        // Initialize with first k sub-vectors
        let mut centroids: Vec<Vec<f32>> = sub_vecs.iter().take(k).cloned().collect();

        let mut assignments = vec![0usize; n];

        for _iter in 0..15 {
            // Assign
            for (i, sv) in sub_vecs.iter().enumerate() {
                let mut best = 0;
                let mut best_dist = f32::MAX;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = sub_distance(sv, centroid);
                    if d < best_dist {
                        best_dist = d;
                        best = c;
                    }
                }
                assignments[i] = best;
            }

            // Update
            let dim = self.sub_dim;
            let mut sums = vec![vec![0.0f32; dim]; centroids.len()];
            let mut counts = vec![0usize; centroids.len()];

            for (i, sv) in sub_vecs.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, &val) in sv.iter().enumerate() {
                    if j < dim {
                        sums[c][j] += val;
                    }
                }
            }

            for (c, centroid) in centroids.iter_mut().enumerate() {
                if counts[c] > 0 {
                    for j in 0..dim {
                        centroid[j] = sums[c][j] / counts[c] as f32;
                    }
                }
            }
        }

        centroids
    }

    /// Get total vectors stored.
    pub fn size(&self) -> usize {
        self.vectors.len()
    }
    /// Get vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Get number of sub-quantizers.
    pub fn n_subquantizers(&self) -> usize {
        self.n_subquantizers
    }
    /// Check if quantizer is trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }
    /// Get configured distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

/// Product quantizer query result.
#[derive(Debug, Clone)]
pub struct PQResult {
    /// Point ID.
    pub id: String,
    /// Distance from query.
    pub distance: f32,
}

/// Squared Euclidean distance between two sub-vectors.
fn sub_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Scalar Quantizer Tests ----

    #[test]
    fn test_sq_creation() {
        let sq = ScalarQuantizer::new(4);
        assert_eq!(sq.dimension(), 4);
        assert!(!sq.is_trained());
    }

    #[test]
    fn test_sq_train_and_query() {
        let mut sq = ScalarQuantizer::new(3);
        let vectors = vec![
            ("a".to_string(), vec![1.0, 0.0, 0.0]),
            ("b".to_string(), vec![0.9, 0.1, 0.0]),
            ("c".to_string(), vec![0.0, 1.0, 0.0]),
            ("d".to_string(), vec![-1.0, 0.0, 0.0]),
        ];
        sq.train(&vectors).unwrap();
        assert!(sq.is_trained());
        assert_eq!(sq.size(), 4);

        let results = sq.query(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // First result should be "a" (exact match or very close)
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_sq_quantize_dequantize() {
        let mut sq = ScalarQuantizer::new(2);
        sq.train(&[
            ("a".to_string(), vec![0.0, 0.0]),
            ("b".to_string(), vec![1.0, 1.0]),
        ])
        .unwrap();

        let q = sq.quantize(&[0.5, 0.5]);
        let d = sq.dequantize(&q);
        // Dequantized should be close to original
        assert!((d[0] - 0.5).abs() < 0.01);
        assert!((d[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sq_insert_after_train() {
        let mut sq = ScalarQuantizer::new(3);
        sq.train(&[("a".to_string(), vec![1.0, 0.0, 0.0])]).unwrap();
        sq.insert("b".to_string(), vec![0.5, 0.5, 0.0]).unwrap();
        assert_eq!(sq.size(), 2);
    }

    #[test]
    fn test_sq_dimension_mismatch() {
        let mut sq = ScalarQuantizer::new(3);
        sq.train(&[("a".to_string(), vec![1.0, 0.0, 0.0])]).unwrap();
        let result = sq.insert("bad".to_string(), vec![1.0, 0.0]);
        assert!(result.is_err());
    }

    // ---- Product Quantizer Tests ----

    #[test]
    fn test_pq_creation() {
        let pq = ProductQuantizer::new(8, 4);
        assert_eq!(pq.dimension(), 8);
        assert_eq!(pq.n_subquantizers(), 4);
        assert!(!pq.is_trained());
    }

    #[test]
    fn test_pq_train_and_query() {
        let mut pq = ProductQuantizer::new(4, 2);
        let vectors = vec![
            ("a".to_string(), vec![1.0, 0.0, 0.0, 0.0]),
            ("b".to_string(), vec![0.9, 0.1, 0.0, 0.0]),
            ("c".to_string(), vec![0.0, 1.0, 0.0, 0.0]),
            ("d".to_string(), vec![0.0, 0.0, 1.0, 0.0]),
            ("e".to_string(), vec![-1.0, 0.0, 0.0, 0.0]),
        ];
        pq.train(&vectors).unwrap();
        assert!(pq.is_trained());
        assert_eq!(pq.size(), 5);

        let results = pq.query(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // "a" should be nearest (exact match in quantized space)
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_pq_insert_after_train() {
        let mut pq = ProductQuantizer::new(4, 2);
        pq.train(&[
            ("a".to_string(), vec![1.0, 0.0, 0.0, 0.0]),
            ("b".to_string(), vec![0.0, 1.0, 0.0, 0.0]),
        ])
        .unwrap();
        pq.insert("c".to_string(), vec![0.5, 0.5, 0.0, 0.0])
            .unwrap();
        assert_eq!(pq.size(), 3);
    }

    #[test]
    fn test_pq_dimension_mismatch() {
        let mut pq = ProductQuantizer::new(4, 2);
        pq.train(&[("a".to_string(), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();
        let result = pq.insert("bad".to_string(), vec![1.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pq_approximate_accuracy() {
        // PQ should still rank nearest neighbors correctly in most cases
        let mut pq = ProductQuantizer::new(4, 2);
        let vectors = vec![
            ("near".to_string(), vec![1.0, 0.0, 0.0, 0.0]),
            ("mid".to_string(), vec![0.5, 0.5, 0.0, 0.0]),
            ("far".to_string(), vec![-1.0, -1.0, -1.0, -1.0]),
        ];
        pq.train(&vectors).unwrap();

        let results = pq.query(&[1.0, 0.0, 0.0, 0.0], 3);
        // "near" should be first
        assert_eq!(results[0].id, "near");
        // "far" should be last
        assert_eq!(results[2].id, "far");
    }
}
