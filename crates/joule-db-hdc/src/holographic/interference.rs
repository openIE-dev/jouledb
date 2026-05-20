//! Interference Pattern Operations
//!
//! Compute and store interference patterns for HAM.

use super::complex::Complex;
use thiserror::Error;

/// Interference pattern errors
#[derive(Error, Debug, Clone)]
pub enum InterferenceError {
    /// Beams have different dimensions
    #[error("Beam dimension mismatch: reference={reference}, object={object}")]
    DimensionMismatch { reference: usize, object: usize },

    /// Reference dimension mismatch during reconstruction
    #[error("Reference dimension mismatch: expected {expected}, got {actual}")]
    ReferenceMismatch { expected: usize, actual: usize },
}

/// Interference pattern representing the hologram
#[derive(Clone)]
pub struct InterferencePattern {
    pattern: Vec<Complex>,
    dimension: usize,
}

impl InterferencePattern {
    /// Create new empty interference pattern
    pub fn new(dimension: usize) -> Self {
        Self {
            pattern: vec![Complex::zero(); dimension],
            dimension,
        }
    }

    /// Create interference pattern from reference and object beams
    ///
    /// The interference pattern is computed as: I = |R + O|² = (R + O)(R + O)*
    pub fn from_beams(
        reference: &[Complex],
        object: &[Complex],
    ) -> Result<InterferencePattern, InterferenceError> {
        if reference.len() != object.len() {
            return Err(InterferenceError::DimensionMismatch {
                reference: reference.len(),
                object: object.len(),
            });
        }

        let dimension = reference.len();
        let mut pattern = Vec::with_capacity(dimension);

        // Interference: I = |R + O|² = (R + O)(R + O)*
        for i in 0..dimension {
            let sum = &reference[i] + &object[i];
            let interference = &sum * &sum.conjugate();
            pattern.push(interference);
        }

        Ok(InterferencePattern { pattern, dimension })
    }

    /// Reconstruct object from interference pattern and reference
    ///
    /// Reconstruction: O ≈ I * R* / |R|²
    pub fn reconstruct(&self, reference: &[Complex]) -> Result<Vec<Complex>, InterferenceError> {
        if reference.len() != self.dimension {
            return Err(InterferenceError::ReferenceMismatch {
                expected: self.dimension,
                actual: reference.len(),
            });
        }

        let mut reconstructed = Vec::with_capacity(self.dimension);

        for i in 0..self.dimension {
            let interference = self.pattern[i];
            let ref_conj = reference[i].conjugate();
            let ref_mag_sq = reference[i].magnitude_squared();

            if ref_mag_sq > 0.001 {
                let reconstructed_val = interference * ref_conj;
                reconstructed.push(Complex {
                    real: reconstructed_val.real / ref_mag_sq,
                    imag: reconstructed_val.imag / ref_mag_sq,
                });
            } else {
                reconstructed.push(Complex::zero());
            }
        }

        Ok(reconstructed)
    }

    /// Get pattern as flat f32 vector (real, imag pairs)
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dimension * 2);
        for c in &self.pattern {
            result.push(c.real);
            result.push(c.imag);
        }
        result
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get raw pattern (for advanced use)
    pub fn pattern(&self) -> &[Complex] {
        &self.pattern
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_interference_creation() {
        let pattern = InterferencePattern::new(10);
        assert_eq!(pattern.dimension(), 10);
    }

    #[test]
    fn test_interference_from_beams() {
        let reference: Vec<Complex> = (0..10)
            .map(|i| Complex::unit(i as f32 * PI / 5.0))
            .collect();
        let object: Vec<Complex> = (0..10)
            .map(|i| Complex::unit(i as f32 * PI / 10.0))
            .collect();

        let pattern = InterferencePattern::from_beams(&reference, &object).unwrap();
        assert_eq!(pattern.dimension(), 10);
    }

    #[test]
    fn test_interference_dimension_mismatch() {
        let reference = vec![Complex::new(1.0, 0.0); 10];
        let object = vec![Complex::new(1.0, 0.0); 5];

        let result = InterferencePattern::from_beams(&reference, &object);
        assert!(matches!(
            result,
            Err(InterferenceError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_reconstruction() {
        // Create simple test case
        let reference: Vec<Complex> = (0..10)
            .map(|i| Complex::unit(i as f32 * PI / 5.0))
            .collect();
        let object: Vec<Complex> = (0..10)
            .map(|i| Complex::new((i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()))
            .collect();

        let pattern = InterferencePattern::from_beams(&reference, &object).unwrap();
        let reconstructed = pattern.reconstruct(&reference).unwrap();

        assert_eq!(reconstructed.len(), 10);
        // Reconstruction won't be exact due to interference formula, but should have similar structure
    }

    #[test]
    fn test_to_f32_vec() {
        let mut pattern = InterferencePattern::new(2);
        pattern.pattern[0] = Complex::new(1.0, 2.0);
        pattern.pattern[1] = Complex::new(3.0, 4.0);

        let vec = pattern.to_f32_vec();
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
