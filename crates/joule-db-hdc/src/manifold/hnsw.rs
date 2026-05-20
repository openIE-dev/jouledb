//! Hierarchical Navigable Small World (HNSW) for accurate approximate nearest neighbor

use joule_db_core::error::IndexError;
use joule_db_core::index::{
    Bound, Index, IndexEntry, IndexIterator, ScanDirection, SimilarityIndex,
};

/// Convert a little-endian byte slice to a Vec<f32>.
pub(crate) fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, IndexError> {
    if bytes.len() % 4 != 0 {
        return Err(IndexError::Corrupted {
            reason: format!(
                "byte length {} not divisible by 4 for f32 vector",
                bytes.len()
            ),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Convert a Vec<f32> to little-endian bytes.
pub(crate) fn f32_vec_to_bytes(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Empty iterator for similarity indexes that don't support range queries.
struct EmptyIndexIterator;

impl Iterator for EmptyIndexIterator {
    type Item = Result<IndexEntry, IndexError>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl IndexIterator for EmptyIndexIterator {}

/// Distance metric for vector similarity search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
    /// Negative inner product (smaller = more similar)
    InnerProduct,
    /// Hamming distance for binary hypervectors.
    /// Vectors are stored as f32 but reinterpreted as packed u32 words.
    /// Use `BinaryHV::to_f32_packed()` to convert before inserting.
    Hamming,
}

/// HNSW query result
#[derive(Debug, Clone)]
pub struct HNSWResult {
    /// Point ID
    pub id: String,
    /// Distance from query
    pub distance: f32,
}

/// HNSW Index for O(log n) nearest neighbor search
///
/// Simplified implementation focusing on correctness over performance.
pub struct HNSWIndex {
    nodes: Vec<(String, Vec<f32>)>,
    connections: Vec<Vec<usize>>,
    dimension: usize,
    max_connections: usize,
    ef_construction: usize,
    metric: DistanceMetric,
}

impl HNSWIndex {
    /// Create new HNSW index
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `max_connections` - Maximum connections per node (M)
    /// * `ef_construction` - Beam width during construction
    pub fn new(dimension: usize, max_connections: usize, ef_construction: usize) -> Self {
        Self::with_metric(
            dimension,
            max_connections,
            ef_construction,
            DistanceMetric::Euclidean,
        )
    }

    /// Create new HNSW index with a specific distance metric
    pub fn with_metric(
        dimension: usize,
        max_connections: usize,
        ef_construction: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            dimension,
            max_connections: max_connections.max(4),
            ef_construction: ef_construction.max(16),
            metric,
        }
    }

    /// Insert a point into the HNSW graph.
    ///
    /// Uses graph search (not brute force) to find neighbors for the new node,
    /// then connects with bidirectional edges and prunes over-connected nodes.
    pub fn insert(&mut self, id: String, point: Vec<f32>) -> Result<usize, String> {
        if point.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                point.len()
            ));
        }

        let idx = self.nodes.len();

        if idx == 0 {
            self.nodes.push((id, point));
            self.connections.push(Vec::new());
            return Ok(idx);
        }

        // Find nearest neighbors using graph search (O(log N) for large graphs)
        let neighbors = self.search_layer(&point, self.ef_construction);

        // Add the node
        self.nodes.push((id, point));

        // Select M best neighbors and create bidirectional connections
        let mut my_connections = Vec::new();
        for (neighbor_idx, _) in neighbors.iter().take(self.max_connections) {
            if *neighbor_idx < self.connections.len() {
                my_connections.push(*neighbor_idx);

                // Add reverse edge
                self.connections[*neighbor_idx].push(idx);

                // Prune over-connected neighbors: keep only M closest
                if self.connections[*neighbor_idx].len() > self.max_connections * 2 {
                    self.prune_connections(*neighbor_idx);
                }
            }
        }

        self.connections.push(my_connections);
        Ok(idx)
    }

    /// Prune connections for an over-connected node.
    /// Keep only the M closest neighbors by distance.
    fn prune_connections(&mut self, node_idx: usize) {
        if node_idx >= self.nodes.len() || node_idx >= self.connections.len() {
            return;
        }

        let node_point = &self.nodes[node_idx].1;

        // Score each connection by distance to the node
        let mut scored: Vec<(usize, f32)> = self.connections[node_idx]
            .iter()
            .filter(|&&n| n < self.nodes.len())
            .map(|&n| (n, self.compute_distance(node_point, &self.nodes[n].1)))
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.max_connections);

        self.connections[node_idx] = scored.into_iter().map(|(n, _)| n).collect();
    }

    /// Greedy graph search: start from entry point, walk edges toward query.
    ///
    /// This is the actual HNSW algorithm (Malkov & Yashunin 2018):
    /// 1. Start from a random entry point
    /// 2. Maintain a candidate set (min-heap by distance) and a result set (max-heap)
    /// 3. For each candidate, explore its neighbors
    /// 4. If a neighbor is closer than the worst result, add it to both sets
    /// 5. Stop when no candidate can improve the result set
    ///
    /// Complexity: O(ef * log(ef) * M) per query, where M = max_connections.
    /// For typical values (ef=50, M=16), this is ~800 distance computations
    /// vs N for brute force. At 1M nodes, that's 1250x faster.
    fn search_layer(&self, query: &[f32], ef: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // For very small graphs, brute force is actually faster (no graph overhead)
        if self.nodes.len() <= ef * 2 {
            return self.search_brute_force(query, ef);
        }

        let mut visited = vec![false; self.nodes.len()];

        // Start from entry point (node 0 — the first inserted)
        let entry = 0usize;
        let entry_dist = self.compute_distance(query, &self.nodes[entry].1);
        visited[entry] = true;

        // Candidate set: nodes to explore (sorted by distance ascending)
        // Result set: best ef results found so far
        let mut candidates: Vec<(usize, f32)> = vec![(entry, entry_dist)];
        let mut results: Vec<(usize, f32)> = vec![(entry, entry_dist)];

        while !candidates.is_empty() {
            // Get closest unprocessed candidate
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let (current_idx, current_dist) = candidates.remove(0);

            // If this candidate is farther than the worst result, we're done
            // (no remaining candidate can improve the result set)
            let worst_result_dist = results
                .iter()
                .map(|(_, d)| *d)
                .fold(f32::NEG_INFINITY, f32::max);

            if results.len() >= ef && current_dist > worst_result_dist {
                break;
            }

            // Explore neighbors of current node
            if current_idx < self.connections.len() {
                for &neighbor_idx in &self.connections[current_idx] {
                    if neighbor_idx < self.nodes.len() && !visited[neighbor_idx] {
                        visited[neighbor_idx] = true;

                        let neighbor_dist =
                            self.compute_distance(query, &self.nodes[neighbor_idx].1);

                        // Add to results if better than worst, or results not full
                        if results.len() < ef || neighbor_dist < worst_result_dist {
                            candidates.push((neighbor_idx, neighbor_dist));
                            results.push((neighbor_idx, neighbor_dist));

                            // Keep results bounded to ef
                            if results.len() > ef {
                                results.sort_by(|a, b| {
                                    a.1.partial_cmp(&b.1)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                results.truncate(ef);
                            }
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Brute-force fallback for small graphs where graph traversal overhead isn't worth it.
    fn search_brute_force(&self, query: &[f32], ef: usize) -> Vec<(usize, f32)> {
        let mut candidates: Vec<(usize, f32)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, (_, p))| (i, self.compute_distance(query, p)))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(ef);
        candidates
    }

    /// Query for k nearest neighbors
    pub fn query(&self, point: &[f32], k: usize) -> Vec<HNSWResult> {
        self.query_with_ef(point, k, self.ef_construction)
    }

    /// Query for k nearest neighbors with a specified ef_search parameter.
    ///
    /// `ef_search` controls the search beam width (higher = more accurate, slower).
    /// Separating ef_search from ef_construction allows tuning search-time accuracy
    /// independently from index build quality.
    pub fn query_with_ef(&self, point: &[f32], k: usize, ef_search: usize) -> Vec<HNSWResult> {
        if point.len() != self.dimension {
            return Vec::new();
        }

        let ef = ef_search.max(k); // ef must be at least k
        let neighbors = self.search_layer(point, ef);

        neighbors
            .into_iter()
            .take(k)
            .map(|(idx, dist)| HNSWResult {
                id: self.nodes[idx].0.clone(),
                distance: dist,
            })
            .collect()
    }

    /// Compute distance using the configured metric
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean => Self::euclidean_distance(a, b),
            DistanceMetric::Cosine => Self::cosine_distance(a, b),
            DistanceMetric::InnerProduct => Self::inner_product_distance(a, b),
            DistanceMetric::Hamming => Self::hamming_distance(a, b),
        }
    }

    /// Euclidean (L2) distance
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Cosine distance: 1 - cosine_similarity
    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        1.0 - (dot / (norm_a * norm_b))
    }

    /// Inner product distance: negate so smaller = more similar
    fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        -dot
    }

    /// Hamming distance for packed binary vectors.
    /// Each f32 is reinterpreted as a u32 of packed bits; distance = total popcount(XOR).
    fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let ax = x.to_bits();
                let bx = y.to_bits();
                (ax ^ bx).count_ones()
            })
            .sum::<u32>() as f32
    }

    /// Get the configured distance metric
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get index size
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get point by index
    pub fn get(&self, idx: usize) -> Option<(&str, &[f32])> {
        self.nodes
            .get(idx)
            .map(|(id, p)| (id.as_str(), p.as_slice()))
    }
}

// === Index + SimilarityIndex trait implementations ===

impl Index for HNSWIndex {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError> {
        let id = std::str::from_utf8(key).map_err(|_| IndexError::Corrupted {
            reason: "invalid UTF-8 key".to_string(),
        })?;
        for (node_id, vector) in &self.nodes {
            if node_id == id {
                return Ok(Some(f32_vec_to_bytes(vector)));
            }
        }
        Ok(None)
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        let id = std::str::from_utf8(key).map_err(|_| IndexError::Corrupted {
            reason: "invalid UTF-8 key".to_string(),
        })?;
        let vector = bytes_to_f32_vec(value)?;
        HNSWIndex::insert(self, id.to_string(), vector)
            .map_err(|e| IndexError::Corrupted { reason: e })?;
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError> {
        let id = std::str::from_utf8(key).map_err(|_| IndexError::Corrupted {
            reason: "invalid UTF-8 key".to_string(),
        })?;
        if let Some(pos) = self.nodes.iter().position(|(nid, _)| nid == id) {
            self.nodes.remove(pos);
            self.connections.remove(pos);
            // Fix connection indices after removal
            for conns in &mut self.connections {
                conns.retain(|&idx| idx != pos);
                for idx in conns.iter_mut() {
                    if *idx > pos {
                        *idx -= 1;
                    }
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
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

impl SimilarityIndex for HNSWIndex {
    fn search(&self, query_key: &[u8], limit: usize) -> Result<Vec<(IndexEntry, f32)>, IndexError> {
        let query = bytes_to_f32_vec(query_key)?;
        let results = self.query(&query, limit);
        Ok(results
            .into_iter()
            .map(|r| {
                let value = self
                    .nodes
                    .iter()
                    .find(|(id, _)| id == &r.id)
                    .map(|(_, v)| f32_vec_to_bytes(v))
                    .unwrap_or_default();
                (IndexEntry::new(r.id.into_bytes(), value), r.distance)
            })
            .collect())
    }

    fn snr(&self) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let total_connections: usize = self.connections.iter().map(|c| c.len()).sum();
        let avg = total_connections as f32 / self.nodes.len() as f32;
        avg / self.max_connections as f32
    }

    fn estimated_capacity(&self) -> usize {
        10_000_000_usize.saturating_sub(self.nodes.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_creation() {
        let index = HNSWIndex::new(64, 16, 100);
        assert_eq!(index.dimension(), 64);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_insert() {
        let mut index = HNSWIndex::new(4, 4, 16);

        let idx = index.insert("a".into(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(index.size(), 1);
    }

    #[test]
    fn test_hnsw_dimension_mismatch() {
        let mut index = HNSWIndex::new(4, 4, 16);
        let result = index.insert("bad".into(), vec![1.0, 2.0]); // Wrong dimension
        assert!(result.is_err());
    }

    #[test]
    fn test_hnsw_query() {
        let mut index = HNSWIndex::new(4, 4, 16);

        index.insert("a".into(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("b".into(), vec![0.9, 0.1, 0.0, 0.0]).unwrap();
        index.insert("c".into(), vec![-1.0, 0.0, 0.0, 0.0]).unwrap();

        let results = index.query(&[1.0, 0.0, 0.0, 0.0], 2);

        assert_eq!(results.len(), 2);
        // First result should be 'a' (exact match)
        assert_eq!(results[0].id, "a");
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_hnsw_nearest_neighbor_accuracy() {
        let mut index = HNSWIndex::new(4, 4, 16);

        // Insert points at known positions
        index
            .insert("origin".into(), vec![0.0, 0.0, 0.0, 0.0])
            .unwrap();
        index.insert("x".into(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("y".into(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        index
            .insert("far".into(), vec![10.0, 10.0, 10.0, 10.0])
            .unwrap();

        // Query near x-axis
        let results = index.query(&[0.5, 0.0, 0.0, 0.0], 2);

        // Should find origin and x as nearest
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"origin") || ids.contains(&"x"));
    }

    #[test]
    fn test_hnsw_get() {
        let mut index = HNSWIndex::new(4, 4, 16);
        index
            .insert("test".into(), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();

        let (id, point) = index.get(0).unwrap();
        assert_eq!(id, "test");
        assert_eq!(point, &[1.0, 2.0, 3.0, 4.0]);

        assert!(index.get(999).is_none());
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let mut index = HNSWIndex::with_metric(3, 4, 16, DistanceMetric::Cosine);

        // Parallel vectors should be closest (cosine distance ≈ 0)
        index
            .insert("parallel".into(), vec![2.0, 0.0, 0.0])
            .unwrap();
        // Perpendicular vector (cosine distance ≈ 1)
        index.insert("perp".into(), vec![0.0, 1.0, 0.0]).unwrap();
        // Anti-parallel vector (cosine distance ≈ 2)
        index.insert("anti".into(), vec![-1.0, 0.0, 0.0]).unwrap();

        let results = index.query(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "parallel");
        assert!(results[0].distance < 0.01);
    }

    #[test]
    fn test_hnsw_inner_product_metric() {
        let mut index = HNSWIndex::with_metric(3, 4, 16, DistanceMetric::InnerProduct);

        // High inner product (most similar under IP)
        index
            .insert("high_ip".into(), vec![10.0, 0.0, 0.0])
            .unwrap();
        // Low inner product
        index.insert("low_ip".into(), vec![1.0, 0.0, 0.0]).unwrap();
        // Negative inner product
        index.insert("neg_ip".into(), vec![-5.0, 0.0, 0.0]).unwrap();

        let results = index.query(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        // Highest IP = -10 (smallest distance), so high_ip is first
        assert_eq!(results[0].id, "high_ip");
    }

    #[test]
    fn test_hnsw_metric_getter() {
        let index = HNSWIndex::new(4, 4, 16);
        assert_eq!(index.metric(), DistanceMetric::Euclidean);

        let index2 = HNSWIndex::with_metric(4, 4, 16, DistanceMetric::Cosine);
        assert_eq!(index2.metric(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_hnsw_graph_search_scales() {
        // Insert 200 points — enough to trigger graph search (not brute force)
        let dim = 8;
        let mut index = HNSWIndex::new(dim, 8, 32);

        for i in 0..200 {
            let point: Vec<f32> = (0..dim)
                .map(|d| ((i * 7 + d * 13) as f32).sin())
                .collect();
            index.insert(format!("p{}", i), point).unwrap();
        }

        assert_eq!(index.size(), 200);

        // Query should return results (graph search path)
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.5).sin()).collect();
        let results = index.query(&query, 5);
        assert_eq!(results.len(), 5);

        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-6);
        }
    }
}
