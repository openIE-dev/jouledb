//! HyperVector: High-dimensional random vector for VSA operations

use thiserror::Error;

/// Hyperdimensional computing errors
#[derive(Error, Debug, Clone)]
pub enum HDError {
    /// Dimension mismatch between vectors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected vector dimension
        expected: usize,
        /// Actual dimension received
        actual: usize,
    },

    /// Empty input
    #[error("Cannot operate on empty input")]
    EmptyInput,

    /// Invalid dimension
    #[error("Invalid dimension: {0}")]
    InvalidDimension(usize),
}

/// High-dimensional random vector for symbolic computation
///
/// HyperVectors are the fundamental data structure in Vector Symbolic
/// Architectures. They enable representing symbols, concepts, and
/// structures in a high-dimensional space where:
///
/// - Random vectors are nearly orthogonal
/// - Similarity can be measured by cosine distance
/// - Binding creates associations
/// - Bundling creates set-like collections
#[derive(Clone, Debug)]
pub struct HyperVector {
    components: Vec<f32>,
    dimension: usize,
}

impl HyperVector {
    /// Create a new random hypervector using a seed
    pub fn random(dimension: usize, seed: u64) -> Self {
        let mut components = Vec::with_capacity(dimension);
        let mut rng = seed;

        for _ in 0..dimension {
            // LCG for deterministic random
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1]
            let val = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
            components.push(val as f32);
        }

        Self::from_components_normalize(components)
    }

    /// Create from components (will normalize)
    pub fn from_components(components: Vec<f32>) -> Self {
        Self::from_components_normalize(components)
    }

    /// Create from components and normalize
    fn from_components_normalize(components: Vec<f32>) -> Self {
        let dimension = components.len();
        let mut hv = Self {
            components,
            dimension,
        };
        hv.normalize_in_place();
        hv
    }

    /// Create a zero vector
    pub fn zero(dimension: usize) -> Self {
        Self {
            components: vec![0.0; dimension],
            dimension,
        }
    }

    /// Normalize vector to unit length in place
    fn normalize_in_place(&mut self) {
        let norm: f32 = self.components.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut self.components {
                *x /= norm;
            }
        }
    }

    /// Get a normalized copy
    pub fn normalize(&self) -> Self {
        let mut result = self.clone();
        result.normalize_in_place();
        result
    }

    /// Get components
    pub fn components(&self) -> &[f32] {
        &self.components
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Compute cosine similarity with another vector
    pub fn similarity(&self, other: &HyperVector) -> Result<f32, HDError> {
        if self.dimension != other.dimension {
            return Err(HDError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let dot: f32 = self
            .components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.components.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.components.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            Ok(dot / (norm_a * norm_b))
        } else {
            Ok(0.0)
        }
    }

    /// Binding operation (element-wise XOR for binary, circular convolution for real)
    ///
    /// For real-valued vectors, we use element-wise multiplication which is
    /// a simplified approximation of circular convolution.
    pub fn bind(&self, other: &HyperVector) -> Result<HyperVector, HDError> {
        if self.dimension != other.dimension {
            return Err(HDError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        // Element-wise multiplication (simplified binding)
        let components: Vec<f32> = self
            .components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(HyperVector::from_components(components))
    }

    /// Full circular convolution binding (more expensive)
    pub fn bind_convolution(&self, other: &HyperVector) -> Result<HyperVector, HDError> {
        if self.dimension != other.dimension {
            return Err(HDError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let dim = self.dimension;
        let mut result = vec![0.0; dim];

        // Circular convolution: C[i] = Σ(A[j] * B[(i-j) mod N])
        for i in 0..dim {
            for j in 0..dim {
                let k = ((i as i32 - j as i32 + dim as i32) as usize) % dim;
                result[i] += self.components[j] * other.components[k];
            }
        }

        Ok(HyperVector::from_components(result))
    }

    /// Unbind (inverse of bind) - for element-wise binding
    pub fn unbind(&self, other: &HyperVector) -> Result<HyperVector, HDError> {
        if self.dimension != other.dimension {
            return Err(HDError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        // For element-wise multiplication, unbind is division (with safety)
        let components: Vec<f32> = self
            .components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| if b.abs() > 0.001 { a / b } else { 0.0 })
            .collect();

        Ok(HyperVector::from_components(components))
    }

    /// Bundle multiple vectors (element-wise sum, then normalize)
    pub fn bundle(vectors: &[HyperVector]) -> Result<HyperVector, HDError> {
        if vectors.is_empty() {
            return Err(HDError::EmptyInput);
        }

        let dim = vectors[0].dimension;
        let mut result = vec![0.0; dim];

        for v in vectors {
            if v.dimension != dim {
                return Err(HDError::DimensionMismatch {
                    expected: dim,
                    actual: v.dimension,
                });
            }
            for j in 0..dim {
                result[j] += v.components[j];
            }
        }

        Ok(HyperVector::from_components(result))
    }

    /// Add another vector (for bundling incrementally)
    pub fn add(&self, other: &HyperVector) -> Result<HyperVector, HDError> {
        if self.dimension != other.dimension {
            return Err(HDError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }

        let components: Vec<f32> = self
            .components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(HyperVector::from_components(components))
    }

    /// Permute vector (shift components) - for sequence encoding
    pub fn permute(&self, shift: i32) -> HyperVector {
        let dim = self.dimension as i32;
        let mut components = vec![0.0; self.dimension];

        for i in 0..self.dimension {
            let src = ((i as i32 - shift) % dim + dim) % dim;
            components[i] = self.components[src as usize];
        }

        HyperVector {
            components,
            dimension: self.dimension,
        }
    }

    /// Inverse permute
    pub fn inverse_permute(&self, shift: i32) -> HyperVector {
        self.permute(-shift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_vector() {
        let v = HyperVector::random(1000, 42);
        assert_eq!(v.dimension(), 1000);

        // Should be normalized
        let norm: f32 = v.components().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_similarity_same() {
        let v = HyperVector::random(1000, 42);
        let sim = v.similarity(&v).unwrap();
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_similarity_orthogonal() {
        let v1 = HyperVector::random(10000, 42);
        let v2 = HyperVector::random(10000, 123);
        let sim = v1.similarity(&v2).unwrap();
        // Random high-dim vectors should be nearly orthogonal
        assert!(sim.abs() < 0.1);
    }

    #[test]
    fn test_bind_unbind() {
        let a = HyperVector::random(1000, 1);
        let b = HyperVector::random(1000, 2);

        let bound = a.bind(&b).unwrap();
        let unbound = bound.unbind(&b).unwrap();

        // Unbound should be similar to a
        let sim = unbound.similarity(&a).unwrap();
        assert!(sim > 0.5);
    }

    #[test]
    fn test_bundle() {
        let v1 = HyperVector::random(1000, 1);
        let v2 = HyperVector::random(1000, 2);
        let v3 = HyperVector::random(1000, 3);

        let bundle = HyperVector::bundle(&[v1.clone(), v2.clone(), v3.clone()]).unwrap();

        // Bundle should be similar to all constituents
        assert!(bundle.similarity(&v1).unwrap() > 0.3);
        assert!(bundle.similarity(&v2).unwrap() > 0.3);
        assert!(bundle.similarity(&v3).unwrap() > 0.3);
    }

    #[test]
    fn test_permute() {
        let v = HyperVector::random(100, 42);
        let shifted = v.permute(5);
        let restored = shifted.inverse_permute(5);

        let sim = v.similarity(&restored).unwrap();
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = HyperVector::random(100, 1);
        let v2 = HyperVector::random(200, 2);

        assert!(matches!(
            v1.similarity(&v2),
            Err(HDError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            v1.bind(&v2),
            Err(HDError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_bundle_empty() {
        assert!(matches!(HyperVector::bundle(&[]), Err(HDError::EmptyInput)));
    }
}
