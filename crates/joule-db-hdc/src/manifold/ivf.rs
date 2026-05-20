//! Inverted File Index (IVF) for approximate nearest neighbor search.
//!
//! Uses k-means clustering to partition the vector space into Voronoi cells.
//! At query time, only the nearest `n_probe` cells are searched, reducing
//! the search space from O(N) to O(N/n_clusters * n_probe).

use super::hnsw::{DistanceMetric, bytes_to_f32_vec, f32_vec_to_bytes};
use joule_db_core::error::IndexError;
use joule_db_core::index::{
    Bound, Index, IndexEntry, IndexIterator, ScanDirection, SimilarityIndex,
};
use std::collections::HashMap;

/// IVF query result
#[derive(Debug, Clone)]
pub struct IVFResult {
    /// Point ID
    pub id: String,
    /// Distance from query
    pub distance: f32,
}

/// Inverted File Index using k-means partitioning.
pub struct IVFIndex {
    dimension: usize,
    n_clusters: usize,
    n_probe: usize,
    metric: DistanceMetric,
    /// Cluster centroids (n_clusters x dimension)
    centroids: Vec<Vec<f32>>,
    /// Inverted lists: cluster_id -> Vec<(point_id, vector)>
    inverted_lists: Vec<Vec<(String, Vec<f32>)>>,
    /// Whether the index has been trained (centroids computed)
    trained: bool,
}

impl IVFIndex {
    /// Create a new IVF index.
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `n_clusters` - Number of Voronoi cells (clusters)
    /// * `n_probe` - Number of cells to search at query time
    pub fn new(dimension: usize, n_clusters: usize, n_probe: usize) -> Self {
        Self::with_metric(dimension, n_clusters, n_probe, DistanceMetric::Euclidean)
    }

    /// Create with a specific distance metric.
    pub fn with_metric(
        dimension: usize,
        n_clusters: usize,
        n_probe: usize,
        metric: DistanceMetric,
    ) -> Self {
        let n_clusters = n_clusters.max(1);
        let n_probe = n_probe.max(1).min(n_clusters);
        Self {
            dimension,
            n_clusters,
            n_probe,
            metric,
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            trained: false,
        }
    }

    /// Train the index using k-means on the given vectors.
    /// Uses Lloyd's algorithm with 20 iterations.
    pub fn train(&mut self, vectors: &[(String, Vec<f32>)]) -> Result<(), String> {
        if vectors.is_empty() {
            // Empty training set — create one empty cluster
            self.centroids = vec![vec![0.0; self.dimension]];
            self.inverted_lists = vec![Vec::new()];
            self.trained = true;
            return Ok(());
        }

        let n = vectors.len();
        let k = self.n_clusters.min(n); // Can't have more clusters than points

        // Initialize centroids using first k points (simple initialization)
        let mut centroids: Vec<Vec<f32>> = vectors.iter().take(k).map(|(_, v)| v.clone()).collect();

        // If fewer vectors than clusters, pad with copies of the last vector
        while centroids.len() < self.n_clusters {
            centroids.push(centroids.last().unwrap().clone());
        }

        // Lloyd's algorithm — 20 iterations
        let max_iters = 20;
        let mut assignments = vec![0usize; n];

        for _iter in 0..max_iters {
            // Assignment step: assign each point to nearest centroid
            let mut changed = false;
            for (i, (_, v)) in vectors.iter().enumerate() {
                let nearest = self.find_nearest_centroid(v, &centroids);
                if nearest != assignments[i] {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed && _iter > 0 {
                break; // Converged
            }

            // Update step: recompute centroids
            let mut sums = vec![vec![0.0f32; self.dimension]; centroids.len()];
            let mut counts = vec![0usize; centroids.len()];

            for (i, (_, v)) in vectors.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, val) in v.iter().enumerate() {
                    if j < self.dimension {
                        sums[c][j] += val;
                    }
                }
            }

            for (c, centroid) in centroids.iter_mut().enumerate() {
                if counts[c] > 0 {
                    for j in 0..self.dimension {
                        centroid[j] = sums[c][j] / counts[c] as f32;
                    }
                }
            }
        }

        // Build inverted lists from final assignments
        let mut inv_lists = vec![Vec::new(); centroids.len()];
        for (i, (id, v)) in vectors.iter().enumerate() {
            inv_lists[assignments[i]].push((id.clone(), v.clone()));
        }

        self.centroids = centroids;
        self.inverted_lists = inv_lists;
        self.trained = true;
        Ok(())
    }

    /// Insert a point into the trained index.
    pub fn insert(&mut self, id: String, point: Vec<f32>) -> Result<(), String> {
        if point.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                point.len()
            ));
        }

        if !self.trained || self.centroids.is_empty() {
            // Auto-train with a single cluster
            self.centroids = vec![point.clone()];
            self.inverted_lists = vec![vec![(id, point)]];
            self.trained = true;
            return Ok(());
        }

        let nearest = self.find_nearest_centroid(&point, &self.centroids);
        self.inverted_lists[nearest].push((id, point));
        Ok(())
    }

    /// Query for k nearest neighbors.
    /// Searches the `n_probe` nearest clusters and returns the top-k results.
    pub fn query(&self, point: &[f32], k: usize) -> Vec<IVFResult> {
        if point.len() != self.dimension || !self.trained {
            return Vec::new();
        }

        // Find n_probe nearest centroids
        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(&self.metric, point, c)))
            .collect();
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Search the n_probe nearest clusters
        let mut candidates: Vec<(String, f32)> = Vec::new();
        for (cluster_idx, _) in centroid_dists.iter().take(self.n_probe) {
            for (id, v) in &self.inverted_lists[*cluster_idx] {
                let dist = compute_distance(&self.metric, point, v);
                candidates.push((id.clone(), dist));
            }
        }

        // Sort and return top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
            .into_iter()
            .take(k)
            .map(|(id, distance)| IVFResult { id, distance })
            .collect()
    }

    fn find_nearest_centroid(&self, point: &[f32], centroids: &[Vec<f32>]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, c) in centroids.iter().enumerate() {
            let d = compute_distance(&self.metric, point, c);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Get index size (total vectors)
    pub fn size(&self) -> usize {
        self.inverted_lists.iter().map(|l| l.len()).sum()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get configured distance metric
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get number of clusters
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Get n_probe setting
    pub fn n_probe(&self) -> usize {
        self.n_probe
    }

    /// Is the index trained?
    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

/// Shared distance computation.
pub(crate) fn compute_distance(metric: &DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
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

// === Index + SimilarityIndex trait implementations ===

/// Empty iterator for similarity indexes that don't support range queries.
struct EmptyIndexIterator;

impl Iterator for EmptyIndexIterator {
    type Item = Result<IndexEntry, IndexError>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl IndexIterator for EmptyIndexIterator {}

impl Index for IVFIndex {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError> {
        let id = std::str::from_utf8(key).map_err(|_| IndexError::Corrupted {
            reason: "invalid UTF-8 key".to_string(),
        })?;
        for list in &self.inverted_lists {
            for (pid, vector) in list {
                if pid == id {
                    return Ok(Some(f32_vec_to_bytes(vector)));
                }
            }
        }
        Ok(None)
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        let id = std::str::from_utf8(key).map_err(|_| IndexError::Corrupted {
            reason: "invalid UTF-8 key".to_string(),
        })?;
        let vector = bytes_to_f32_vec(value)?;
        IVFIndex::insert(self, id.to_string(), vector)
            .map_err(|e| IndexError::Corrupted { reason: e })?;
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError> {
        let id = std::str::from_utf8(key).map_err(|_| IndexError::Corrupted {
            reason: "invalid UTF-8 key".to_string(),
        })?;
        for list in &mut self.inverted_lists {
            if let Some(pos) = list.iter().position(|(pid, _)| pid == id) {
                list.remove(pos);
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn range(
        &self,
        _start: Bound<&[u8]>,
        _end: Bound<&[u8]>,
        _direction: ScanDirection,
    ) -> Result<Box<dyn IndexIterator + '_>, IndexError> {
        Ok(Box::new(EmptyIndexIterator))
    }
}

impl SimilarityIndex for IVFIndex {
    fn search(&self, query_key: &[u8], limit: usize) -> Result<Vec<(IndexEntry, f32)>, IndexError> {
        let query = bytes_to_f32_vec(query_key)?;
        let results = self.query(&query, limit);
        Ok(results
            .into_iter()
            .map(|r| {
                // Look up vector bytes for the returned ID
                let value = self
                    .inverted_lists
                    .iter()
                    .flat_map(|list| list.iter())
                    .find(|(id, _)| id == &r.id)
                    .map(|(_, v)| f32_vec_to_bytes(v))
                    .unwrap_or_default();
                (IndexEntry::new(r.id.into_bytes(), value), r.distance)
            })
            .collect())
    }

    fn snr(&self) -> f32 {
        if !self.trained || self.inverted_lists.is_empty() {
            return 0.0;
        }
        let total: usize = self.inverted_lists.iter().map(|l| l.len()).sum();
        if total == 0 {
            return 0.0;
        }
        // SNR based on cluster balance: well-balanced clusters = higher SNR
        let avg = total as f32 / self.inverted_lists.len() as f32;
        let variance: f32 = self
            .inverted_lists
            .iter()
            .map(|l| (l.len() as f32 - avg).powi(2))
            .sum::<f32>()
            / self.inverted_lists.len() as f32;
        if variance < 0.001 {
            return 1.0;
        }
        (avg / variance.sqrt()).min(1.0)
    }

    fn estimated_capacity(&self) -> usize {
        10_000_000_usize.saturating_sub(self.size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_creation() {
        let index = IVFIndex::new(4, 3, 2);
        assert_eq!(index.dimension(), 4);
        assert_eq!(index.n_clusters(), 3);
        assert_eq!(index.n_probe(), 2);
        assert!(!index.is_trained());
    }

    #[test]
    fn test_ivf_train_and_query() {
        let mut index = IVFIndex::new(3, 2, 2);
        let vectors = vec![
            ("a".to_string(), vec![1.0, 0.0, 0.0]),
            ("b".to_string(), vec![0.9, 0.1, 0.0]),
            ("c".to_string(), vec![0.0, 1.0, 0.0]),
            ("d".to_string(), vec![0.0, 0.9, 0.1]),
            ("e".to_string(), vec![-1.0, 0.0, 0.0]),
        ];
        index.train(&vectors).unwrap();
        assert!(index.is_trained());
        assert_eq!(index.size(), 5);

        let results = index.query(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_ivf_insert_after_train() {
        let mut index = IVFIndex::new(3, 2, 2);
        let vectors = vec![
            ("a".to_string(), vec![1.0, 0.0, 0.0]),
            ("b".to_string(), vec![0.0, 1.0, 0.0]),
        ];
        index.train(&vectors).unwrap();
        index.insert("c".to_string(), vec![0.5, 0.5, 0.0]).unwrap();
        assert_eq!(index.size(), 3);

        let results = index.query(&[0.5, 0.5, 0.0], 1);
        assert_eq!(results[0].id, "c");
    }

    #[test]
    fn test_ivf_empty_train() {
        let mut index = IVFIndex::new(3, 4, 2);
        index.train(&[]).unwrap();
        assert!(index.is_trained());
        assert_eq!(index.size(), 0);
    }

    #[test]
    fn test_ivf_cosine_metric() {
        let mut index = IVFIndex::with_metric(3, 2, 2, DistanceMetric::Cosine);
        let vectors = vec![
            ("parallel".to_string(), vec![2.0, 0.0, 0.0]),
            ("perp".to_string(), vec![0.0, 1.0, 0.0]),
            ("anti".to_string(), vec![-1.0, 0.0, 0.0]),
        ];
        index.train(&vectors).unwrap();

        let results = index.query(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results[0].id, "parallel");
        assert!(results[0].distance < 0.01);
    }

    #[test]
    fn test_ivf_dimension_mismatch() {
        let mut index = IVFIndex::new(3, 2, 1);
        index
            .train(&[("a".to_string(), vec![1.0, 0.0, 0.0])])
            .unwrap();
        let result = index.insert("bad".to_string(), vec![1.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ivf_kmeans_convergence() {
        // Two well-separated clusters
        let mut index = IVFIndex::new(2, 2, 1);
        let vectors = vec![
            ("a1".to_string(), vec![0.0, 0.0]),
            ("a2".to_string(), vec![0.1, 0.1]),
            ("a3".to_string(), vec![-0.1, 0.0]),
            ("b1".to_string(), vec![10.0, 10.0]),
            ("b2".to_string(), vec![10.1, 10.1]),
            ("b3".to_string(), vec![9.9, 10.0]),
        ];
        index.train(&vectors).unwrap();

        // Query near cluster A — should find a1, a2, a3
        let results = index.query(&[0.0, 0.0], 3);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a1") || ids.contains(&"a2") || ids.contains(&"a3"));
        // With n_probe=1, might miss cluster B
    }
}
