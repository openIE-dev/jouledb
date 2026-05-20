//! Manifold point representation

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Point on the information manifold
#[derive(Clone, Debug)]
pub struct ManifoldPoint {
    /// Coordinates in embedding space
    pub coords: Vec<f32>,
    dimension: usize,
}

impl ManifoldPoint {
    /// Create new point with given coordinates
    pub fn new(coords: Vec<f32>) -> Self {
        let dimension = coords.len();
        Self { coords, dimension }
    }

    /// Create point from data using random projection (deterministic)
    pub fn from_data(data: &[u8], dimension: usize) -> Self {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let mut rng = hasher.finish();

        let coords: Vec<f32> = (0..dimension)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        Self { coords, dimension }
    }

    /// Create from embedding (e.g., from neural network)
    pub fn from_embedding(embedding: &[f32]) -> Self {
        let dimension = embedding.len();
        Self {
            coords: embedding.to_vec(),
            dimension,
        }
    }

    /// Zero point
    pub fn zero(dimension: usize) -> Self {
        Self {
            coords: vec![0.0; dimension],
            dimension,
        }
    }

    /// Euclidean distance (approximation of geodesic on flat manifold)
    pub fn euclidean_distance(&self, other: &ManifoldPoint) -> f32 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Cosine distance (1 - cosine_similarity)
    pub fn cosine_distance(&self, other: &ManifoldPoint) -> f32 {
        let dot: f32 = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.coords.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.coords.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            1.0 - (dot / (norm_a * norm_b))
        } else {
            1.0
        }
    }

    /// Cosine similarity
    pub fn cosine_similarity(&self, other: &ManifoldPoint) -> f32 {
        1.0 - self.cosine_distance(other)
    }

    /// Geodesic distance (using Euclidean as approximation for flat manifold)
    pub fn geodesic_distance(&self, other: &ManifoldPoint) -> f32 {
        self.euclidean_distance(other)
    }

    /// Manhattan (L1) distance
    pub fn manhattan_distance(&self, other: &ManifoldPoint) -> f32 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).abs())
            .sum()
    }

    /// Get coordinates
    pub fn coordinates(&self) -> &[f32] {
        &self.coords
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Compute magnitude (L2 norm)
    pub fn magnitude(&self) -> f32 {
        self.coords.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(&self) -> ManifoldPoint {
        let mag = self.magnitude();
        if mag > 0.0 {
            ManifoldPoint::new(self.coords.iter().map(|x| x / mag).collect())
        } else {
            self.clone()
        }
    }

    /// Add another point (element-wise)
    pub fn add(&self, other: &ManifoldPoint) -> ManifoldPoint {
        ManifoldPoint::new(
            self.coords
                .iter()
                .zip(other.coords.iter())
                .map(|(a, b)| a + b)
                .collect(),
        )
    }

    /// Scale by a factor
    pub fn scale(&self, factor: f32) -> ManifoldPoint {
        ManifoldPoint::new(self.coords.iter().map(|x| x * factor).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = ManifoldPoint::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.dimension(), 3);
        assert_eq!(p.coordinates(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_data() {
        let p1 = ManifoldPoint::from_data(b"hello", 10);
        let p2 = ManifoldPoint::from_data(b"hello", 10);
        let p3 = ManifoldPoint::from_data(b"world", 10);

        // Same data = same point
        assert_eq!(p1.euclidean_distance(&p2), 0.0);
        // Different data = different point
        assert!(p1.euclidean_distance(&p3) > 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let p1 = ManifoldPoint::new(vec![0.0, 0.0]);
        let p2 = ManifoldPoint::new(vec![3.0, 4.0]);

        assert!((p1.euclidean_distance(&p2) - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_distance() {
        let p1 = ManifoldPoint::new(vec![1.0, 0.0]);
        let p2 = ManifoldPoint::new(vec![1.0, 0.0]);
        let p3 = ManifoldPoint::new(vec![0.0, 1.0]);

        // Same direction = 0 distance
        assert!(p1.cosine_distance(&p2).abs() < 0.0001);
        // Orthogonal = 1 distance
        assert!((p1.cosine_distance(&p3) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_normalize() {
        let p = ManifoldPoint::new(vec![3.0, 4.0]);
        let normalized = p.normalize();

        assert!((normalized.magnitude() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_add_and_scale() {
        let p1 = ManifoldPoint::new(vec![1.0, 2.0]);
        let p2 = ManifoldPoint::new(vec![3.0, 4.0]);

        let sum = p1.add(&p2);
        assert_eq!(sum.coordinates(), &[4.0, 6.0]);

        let scaled = p1.scale(2.0);
        assert_eq!(scaled.coordinates(), &[2.0, 4.0]);
    }
}
