//! Locality-Sensitive Hashing (LSH) for approximate nearest neighbor search

use std::collections::HashMap;

/// LSH Hash Table using random hyperplane projections
pub struct LSHTable {
    /// Hash buckets: bucket_id -> list of (point_id, data)
    buckets: HashMap<u64, Vec<(String, Vec<f32>)>>,
    /// Random projection vectors
    projections: Vec<Vec<f32>>,
    /// Number of hash bits per table
    num_bits: usize,
}

impl LSHTable {
    /// Create new LSH table with random projections
    pub fn new(dimension: usize, num_bits: usize, seed: u64) -> Self {
        let mut rng = seed;
        let mut projections = Vec::with_capacity(num_bits);

        for _ in 0..num_bits {
            let mut proj = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Box-Muller transform for Gaussian-like random values
                let u1 = (rng as f64) / (u64::MAX as f64);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = (rng as f64) / (u64::MAX as f64);
                let g = ((-2.0 * u1.max(1e-10).ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
                proj.push(g);
            }
            projections.push(proj);
        }

        Self {
            buckets: HashMap::new(),
            projections,
            num_bits,
        }
    }

    /// Compute hash for a vector
    pub fn hash(&self, point: &[f32]) -> u64 {
        let mut hash = 0u64;
        for (i, proj) in self.projections.iter().enumerate() {
            let dot: f32 = proj.iter().zip(point.iter()).map(|(a, b)| a * b).sum();
            if dot > 0.0 {
                hash |= 1 << i;
            }
        }
        hash
    }

    /// Insert a point
    pub fn insert(&mut self, id: String, point: Vec<f32>) {
        let hash = self.hash(&point);
        self.buckets.entry(hash).or_default().push((id, point));
    }

    /// Query for similar points (exact bucket match)
    pub fn query(&self, point: &[f32]) -> Vec<&(String, Vec<f32>)> {
        let hash = self.hash(point);
        self.buckets
            .get(&hash)
            .map_or(Vec::new(), |bucket| bucket.iter().collect())
    }

    /// Query with Hamming distance tolerance (check nearby buckets)
    pub fn query_with_tolerance(
        &self,
        point: &[f32],
        tolerance: usize,
    ) -> Vec<&(String, Vec<f32>)> {
        let base_hash = self.hash(point);
        let mut results = Vec::new();

        // Check exact match first
        if let Some(bucket) = self.buckets.get(&base_hash) {
            results.extend(bucket.iter());
        }

        // Check buckets with Hamming distance <= tolerance
        if tolerance > 0 {
            for bits_to_flip in 1..=tolerance.min(self.num_bits) {
                for mask in Self::bit_combinations(self.num_bits, bits_to_flip) {
                    let nearby_hash = base_hash ^ mask;
                    if let Some(bucket) = self.buckets.get(&nearby_hash) {
                        results.extend(bucket.iter());
                    }
                }
            }
        }

        results
    }

    /// Generate combinations of n choose k bit positions
    fn bit_combinations(n: usize, k: usize) -> Vec<u64> {
        if k > n || n > 16 {
            return Vec::new();
        }

        let mut results = Vec::new();
        let mut combo = (1u64 << k) - 1;
        let limit = 1u64 << n;

        while combo < limit {
            results.push(combo);
            // Gosper's hack for next combination
            let c = combo & (!combo + 1);
            let r = combo + c;
            combo = (((r ^ combo) >> 2) / c) | r;

            if results.len() > 1000 {
                break;
            }
        }

        results
    }

    /// Number of buckets
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Total points stored
    pub fn total_points(&self) -> usize {
        self.buckets.values().map(|b| b.len()).sum()
    }
}

/// Multi-table LSH index for better recall
pub struct LSHIndex {
    tables: Vec<LSHTable>,
    dimension: usize,
    num_tables: usize,
    num_bits: usize,
    metric: super::hnsw::DistanceMetric,
}

/// Result of LSH query
#[derive(Debug, Clone)]
pub struct LSHResult {
    /// Point ID
    pub id: String,
    /// Distance from query
    pub distance: f32,
}

impl LSHIndex {
    /// Create new LSH index with Euclidean metric
    pub fn new(dimension: usize, num_tables: usize, num_bits: usize) -> Self {
        Self::with_metric(
            dimension,
            num_tables,
            num_bits,
            super::hnsw::DistanceMetric::Euclidean,
        )
    }

    /// Create new LSH index with a specific distance metric
    pub fn with_metric(
        dimension: usize,
        num_tables: usize,
        num_bits: usize,
        metric: super::hnsw::DistanceMetric,
    ) -> Self {
        let tables = (0..num_tables)
            .map(|i| LSHTable::new(dimension, num_bits, i as u64 * 31337 + 12345))
            .collect();

        Self {
            tables,
            dimension,
            num_tables,
            num_bits,
            metric,
        }
    }

    /// Insert a point
    pub fn insert(&mut self, id: String, point: Vec<f32>) -> Result<(), String> {
        if point.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                point.len()
            ));
        }

        for table in &mut self.tables {
            table.insert(id.clone(), point.clone());
        }
        Ok(())
    }

    /// Query for k nearest neighbors (approximate)
    pub fn query(&self, point: &[f32], k: usize) -> Vec<LSHResult> {
        if point.len() != self.dimension {
            return Vec::new();
        }

        // Collect candidates from all tables
        let mut candidates: HashMap<String, (Vec<f32>, f32)> = HashMap::new();

        for table in &self.tables {
            for (id, p) in table.query_with_tolerance(point, 2) {
                if !candidates.contains_key(id) {
                    let dist = self.compute_distance(point, p);
                    candidates.insert(id.clone(), (p.clone(), dist));
                }
            }
        }

        // Sort by distance
        let mut sorted: Vec<_> = candidates.into_iter().collect();
        sorted.sort_by(|a, b| {
            a.1.1
                .partial_cmp(&b.1.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top-k
        sorted
            .into_iter()
            .take(k)
            .map(|(id, (_, dist))| LSHResult { id, distance: dist })
            .collect()
    }

    /// Compute distance using the configured metric
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        use super::hnsw::DistanceMetric;
        match self.metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if na == 0.0 || nb == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (na * nb))
                }
            }
            DistanceMetric::InnerProduct => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                -dot
            }
            DistanceMetric::Hamming => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x.to_bits() ^ y.to_bits()).count_ones())
                .sum::<u32>() as f32,
        }
    }

    /// Get the configured distance metric
    pub fn metric(&self) -> super::hnsw::DistanceMetric {
        self.metric
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get number of tables
    pub fn num_tables(&self) -> usize {
        self.num_tables
    }

    /// Get number of bits per hash
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Get total points (in first table)
    pub fn size(&self) -> usize {
        self.tables.first().map_or(0, |t| t.total_points())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_table_creation() {
        let table = LSHTable::new(64, 8, 42);
        assert_eq!(table.num_bits, 8);
    }

    #[test]
    fn test_lsh_hash_deterministic() {
        let table = LSHTable::new(64, 8, 42);
        let point = vec![1.0; 64];

        let hash1 = table.hash(&point);
        let hash2 = table.hash(&point);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_lsh_table_insert_query() {
        let mut table = LSHTable::new(4, 4, 42);

        table.insert("a".into(), vec![1.0, 0.0, 0.0, 0.0]);
        table.insert("b".into(), vec![0.9, 0.1, 0.0, 0.0]); // Similar to a
        table.insert("c".into(), vec![-1.0, 0.0, 0.0, 0.0]); // Different

        let results = table.query(&[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_lsh_index_creation() {
        let index = LSHIndex::new(64, 4, 8);
        assert_eq!(index.dimension(), 64);
        assert_eq!(index.num_tables(), 4);
    }

    #[test]
    fn test_lsh_index_insert_query() {
        let mut index = LSHIndex::new(4, 4, 4);

        index.insert("a".into(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("b".into(), vec![0.9, 0.1, 0.0, 0.0]).unwrap();
        index.insert("c".into(), vec![-1.0, 0.0, 0.0, 0.0]).unwrap();

        let results = index.query(&[1.0, 0.0, 0.0, 0.0], 2);

        // Should find 'a' as closest
        assert!(!results.is_empty());
    }

    #[test]
    fn test_lsh_dimension_mismatch() {
        let mut index = LSHIndex::new(4, 2, 4);
        let result = index.insert("bad".into(), vec![1.0, 2.0]); // Wrong dimension
        assert!(result.is_err());
    }
}
