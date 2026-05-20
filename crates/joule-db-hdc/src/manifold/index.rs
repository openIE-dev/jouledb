//! Information Manifold Index

use super::point::ManifoldPoint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Manifold errors
#[derive(Error, Debug, Clone)]
pub enum ManifoldError {
    /// Dimension mismatch between points
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected point dimension
        expected: usize,
        /// Actual dimension received
        actual: usize,
    },

    /// Point not found
    #[error("Point '{0}' not found")]
    PointNotFound(String),

    /// Lock error
    #[error("Lock poisoned")]
    LockPoisoned,
}

/// Result of neighbor search
#[derive(Debug, Clone)]
pub struct NeighborResult {
    /// Index in the manifold
    pub index: usize,
    /// Point ID
    pub id: String,
    /// Distance from query
    pub distance: f32,
}

/// Statistics about the manifold
#[derive(Debug, Clone)]
pub struct ManifoldStats {
    /// Number of points
    pub num_points: usize,
    /// Dimension
    pub dimension: usize,
    /// Average pairwise distance (sampled)
    pub avg_distance: Option<f32>,
}

/// Information Manifold Index
pub struct InformationManifold {
    points: Arc<RwLock<Vec<(ManifoldPoint, String)>>>,
    id_to_index: Arc<RwLock<HashMap<String, usize>>>,
    dimension: usize,
}

impl InformationManifold {
    /// Create new information manifold
    pub fn new(dimension: usize) -> Self {
        Self {
            points: Arc::new(RwLock::new(Vec::new())),
            id_to_index: Arc::new(RwLock::new(HashMap::new())),
            dimension,
        }
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get number of points
    pub fn len(&self) -> usize {
        self.points.read().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert data point
    pub fn insert(&self, id: &str, data: &[u8]) -> usize {
        let point = ManifoldPoint::from_data(data, self.dimension);
        self.insert_point(id, point)
    }

    /// Insert with embedding
    pub fn insert_embedding(&self, id: &str, embedding: &[f32]) -> Result<usize, ManifoldError> {
        if embedding.len() != self.dimension {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let point = ManifoldPoint::from_embedding(embedding);
        Ok(self.insert_point(id, point))
    }

    /// Insert a point
    fn insert_point(&self, id: &str, point: ManifoldPoint) -> usize {
        let mut points = self.points.write().unwrap();
        let mut id_to_index = self.id_to_index.write().unwrap();

        let index = points.len();
        points.push((point, id.to_string()));
        id_to_index.insert(id.to_string(), index);

        index
    }

    /// Find k nearest neighbors
    pub fn nearest_neighbors(&self, query: &ManifoldPoint, k: usize) -> Vec<NeighborResult> {
        let points = self.points.read().unwrap();

        let mut distances: Vec<(usize, &str, f32)> = points
            .iter()
            .enumerate()
            .map(|(i, (p, id))| (i, id.as_str(), query.geodesic_distance(p)))
            .collect();

        distances.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        distances
            .into_iter()
            .take(k)
            .map(|(index, id, distance)| NeighborResult {
                index,
                id: id.to_string(),
                distance,
            })
            .collect()
    }

    /// Find neighbors by data
    pub fn nearest_by_data(&self, data: &[u8], k: usize) -> Vec<NeighborResult> {
        let query = ManifoldPoint::from_data(data, self.dimension);
        self.nearest_neighbors(&query, k)
    }

    /// Find neighbors by embedding
    pub fn nearest_by_embedding(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<NeighborResult>, ManifoldError> {
        if embedding.len() != self.dimension {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        let query = ManifoldPoint::from_embedding(embedding);
        Ok(self.nearest_neighbors(&query, k))
    }

    /// Range query (all points within radius)
    pub fn range_query(&self, query: &ManifoldPoint, radius: f32) -> Vec<NeighborResult> {
        let points = self.points.read().unwrap();

        points
            .iter()
            .enumerate()
            .filter_map(|(i, (p, id))| {
                let dist = query.geodesic_distance(p);
                if dist <= radius {
                    Some(NeighborResult {
                        index: i,
                        id: id.clone(),
                        distance: dist,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get point by ID
    pub fn get_point(&self, id: &str) -> Option<ManifoldPoint> {
        let id_to_index = self.id_to_index.read().unwrap();
        let points = self.points.read().unwrap();

        id_to_index
            .get(id)
            .and_then(|&idx| points.get(idx))
            .map(|(p, _)| p.clone())
    }

    /// Check if point exists
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index.read().unwrap().contains_key(id)
    }

    /// Get all point IDs
    pub fn list_ids(&self) -> Vec<String> {
        self.id_to_index.read().unwrap().keys().cloned().collect()
    }

    /// Compute centroid of all points
    pub fn centroid(&self) -> Option<ManifoldPoint> {
        let points = self.points.read().unwrap();
        if points.is_empty() {
            return None;
        }

        let mut sum = vec![0.0f32; self.dimension];
        for (p, _) in points.iter() {
            for (i, &c) in p.coords.iter().enumerate() {
                if i < sum.len() {
                    sum[i] += c;
                }
            }
        }

        let n = points.len() as f32;
        let centroid: Vec<f32> = sum.iter().map(|&s| s / n).collect();

        Some(ManifoldPoint::new(centroid))
    }

    /// Compute manifold curvature (local density estimation)
    pub fn local_curvature(&self, point: &ManifoldPoint, k: usize) -> f32 {
        let points = self.points.read().unwrap();
        if points.len() < k {
            return 0.0;
        }

        // Find k nearest neighbors
        let mut distances: Vec<f32> = points
            .iter()
            .map(|(p, _)| point.geodesic_distance(p))
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Average distance to k nearest (inverse of local density)
        let avg_dist: f32 = distances.iter().take(k).sum::<f32>() / k as f32;

        // Curvature ~ 1 / local_radius
        if avg_dist > 0.0 { 1.0 / avg_dist } else { 0.0 }
    }

    /// Get statistics
    pub fn stats(&self) -> ManifoldStats {
        let points = self.points.read().unwrap();

        let avg_distance = if points.len() >= 2 {
            // Sample for efficiency
            let sample_size = points.len().min(100);
            let mut total_dist = 0.0f32;
            let mut count = 0;

            for i in 0..sample_size {
                for j in (i + 1)..sample_size {
                    total_dist += points[i].0.geodesic_distance(&points[j].0);
                    count += 1;
                }
            }

            if count > 0 {
                Some(total_dist / count as f32)
            } else {
                None
            }
        } else {
            None
        };

        ManifoldStats {
            num_points: points.len(),
            dimension: self.dimension,
            avg_distance,
        }
    }

    /// Clear all points
    pub fn clear(&self) {
        self.points.write().unwrap().clear();
        self.id_to_index.write().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifold_creation() {
        let manifold = InformationManifold::new(64);
        assert_eq!(manifold.dimension(), 64);
        assert!(manifold.is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let manifold = InformationManifold::new(32);

        manifold.insert("doc1", b"hello world");
        assert_eq!(manifold.len(), 1);
        assert!(manifold.contains("doc1"));

        let point = manifold.get_point("doc1");
        assert!(point.is_some());
    }

    #[test]
    fn test_insert_embedding() {
        let manifold = InformationManifold::new(4);
        let embedding = vec![1.0, 2.0, 3.0, 4.0];

        let idx = manifold.insert_embedding("vec1", &embedding).unwrap();
        assert_eq!(idx, 0);

        let point = manifold.get_point("vec1").unwrap();
        assert_eq!(point.coordinates(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let manifold = InformationManifold::new(4);
        let wrong_size = vec![1.0, 2.0]; // Wrong dimension

        let result = manifold.insert_embedding("bad", &wrong_size);
        assert!(matches!(
            result,
            Err(ManifoldError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_nearest_neighbors() {
        let manifold = InformationManifold::new(32);

        manifold.insert("a", b"apple");
        manifold.insert("b", b"banana");
        manifold.insert("c", b"cherry");

        let neighbors = manifold.nearest_by_data(b"apple", 2);
        assert_eq!(neighbors.len(), 2);
        // First should be exact match
        assert_eq!(neighbors[0].id, "a");
        assert_eq!(neighbors[0].distance, 0.0);
    }

    #[test]
    fn test_range_query() {
        let manifold = InformationManifold::new(32);

        manifold.insert("a", b"test1");
        manifold.insert("b", b"test2");
        manifold.insert("c", b"completely different data here");

        let query = ManifoldPoint::from_data(b"test1", 32);
        let results = manifold.range_query(&query, 0.1);

        // Should find at least the exact match
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.id == "a"));
    }

    #[test]
    fn test_centroid() {
        let manifold = InformationManifold::new(4);

        manifold
            .insert_embedding("a", &[0.0, 0.0, 0.0, 0.0])
            .unwrap();
        manifold
            .insert_embedding("b", &[2.0, 2.0, 2.0, 2.0])
            .unwrap();

        let centroid = manifold.centroid().unwrap();
        assert_eq!(centroid.coordinates(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_local_curvature() {
        let manifold = InformationManifold::new(4);

        for i in 0..10 {
            manifold
                .insert_embedding(&format!("p{}", i), &[i as f32, 0.0, 0.0, 0.0])
                .unwrap();
        }

        let center = ManifoldPoint::new(vec![5.0, 0.0, 0.0, 0.0]);
        let curvature = manifold.local_curvature(&center, 3);

        assert!(curvature > 0.0);
    }

    #[test]
    fn test_stats() {
        let manifold = InformationManifold::new(32);

        manifold.insert("a", b"hello");
        manifold.insert("b", b"world");

        let stats = manifold.stats();
        assert_eq!(stats.num_points, 2);
        assert_eq!(stats.dimension, 32);
        assert!(stats.avg_distance.is_some());
    }

    #[test]
    fn test_clear() {
        let manifold = InformationManifold::new(32);

        manifold.insert("a", b"hello");
        assert_eq!(manifold.len(), 1);

        manifold.clear();
        assert!(manifold.is_empty());
    }
}
