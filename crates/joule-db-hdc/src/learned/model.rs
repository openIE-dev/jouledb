//! Learned Index Model
//!
//! Pure Rust implementation of learned index models using polynomial regression.
//! Replaces WebNN dependency with efficient native algorithms.

use super::{LearnedError, LearnedResult};

/// Type of model to use for learned index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Linear regression: y = ax + b
    Linear,
    /// Quadratic regression: y = ax^2 + bx + c
    Quadratic,
    /// Cubic regression: y = ax^3 + bx^2 + cx + d
    Cubic,
    /// Piecewise linear with specified number of segments
    PiecewiseLinear(usize),
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::Linear
    }
}

/// Learned Index Model
///
/// Uses polynomial regression to learn the cumulative distribution function (CDF)
/// of keys, enabling O(1) position prediction.
#[derive(Debug, Clone)]
pub struct LearnedIndexModel {
    /// Type of model
    model_type: ModelType,
    /// Polynomial coefficients (lowest degree first)
    /// For linear: [b, a] -> y = a*x + b
    /// For quadratic: [c, b, a] -> y = a*x^2 + b*x + c
    coefficients: Vec<f64>,
    /// Piecewise segment boundaries and slopes (for PiecewiseLinear)
    segments: Vec<PiecewiseSegment>,
    /// Minimum key value (for normalization)
    min_key: f64,
    /// Maximum key value (for normalization)
    max_key: f64,
    /// Number of records
    num_records: usize,
    /// Whether model has been trained
    trained: bool,
    /// Mean absolute error from training
    mae: f64,
    /// Max absolute error from training (for error bounds)
    max_error: f64,
}

/// A segment in piecewise linear model
#[derive(Debug, Clone)]
struct PiecewiseSegment {
    /// Start key (normalized)
    start: f64,
    /// End key (normalized)
    end: f64,
    /// Slope
    slope: f64,
    /// Intercept
    intercept: f64,
}

impl LearnedIndexModel {
    /// Create a new learned index model
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            coefficients: Vec::new(),
            segments: Vec::new(),
            min_key: 0.0,
            max_key: 1.0,
            num_records: 0,
            trained: false,
            mae: 0.0,
            max_error: 0.0,
        }
    }

    /// Create a linear model (most common case)
    pub fn linear() -> Self {
        Self::new(ModelType::Linear)
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get mean absolute error from training
    pub fn mean_absolute_error(&self) -> f64 {
        self.mae
    }

    /// Get maximum absolute error from training
    pub fn max_absolute_error(&self) -> f64 {
        self.max_error
    }

    /// Get number of records
    pub fn num_records(&self) -> usize {
        self.num_records
    }

    /// Get key bounds
    pub fn key_bounds(&self) -> (f64, f64) {
        (self.min_key, self.max_key)
    }

    /// Train the model on sorted key-position pairs
    ///
    /// # Arguments
    /// * `data` - Sorted pairs of (key, position) where position is 0-indexed
    ///
    /// # Returns
    /// Training statistics (MAE, max error)
    pub fn train(&mut self, data: &[(f64, usize)]) -> LearnedResult<(f64, f64)> {
        if data.is_empty() {
            return Err(LearnedError::InvalidTrainingData(
                "empty training data".to_string(),
            ));
        }

        if data.len() < 2 {
            return Err(LearnedError::InvalidTrainingData(
                "need at least 2 data points".to_string(),
            ));
        }

        // Extract bounds
        self.min_key = data[0].0;
        self.max_key = data[data.len() - 1].0;
        self.num_records = data.len();

        // Handle edge case where all keys are the same
        if (self.max_key - self.min_key).abs() < f64::EPSILON {
            self.max_key = self.min_key + 1.0;
        }

        // Normalize keys to [0, 1] and positions to [0, 1]
        let normalized: Vec<(f64, f64)> = data
            .iter()
            .map(|(k, p)| {
                let norm_k = (k - self.min_key) / (self.max_key - self.min_key);
                let norm_p = *p as f64 / (self.num_records - 1).max(1) as f64;
                (norm_k, norm_p)
            })
            .collect();

        // Fit model based on type
        match self.model_type {
            ModelType::Linear => self.fit_linear(&normalized)?,
            ModelType::Quadratic => self.fit_polynomial(&normalized, 2)?,
            ModelType::Cubic => self.fit_polynomial(&normalized, 3)?,
            ModelType::PiecewiseLinear(segments) => self.fit_piecewise(&normalized, segments)?,
        }

        // Calculate errors
        let (mae, max_err) = self.calculate_errors(data);
        self.mae = mae;
        self.max_error = max_err;
        self.trained = true;

        Ok((mae, max_err))
    }

    /// Fit linear model using least squares
    fn fit_linear(&mut self, data: &[(f64, f64)]) -> LearnedResult<()> {
        let n = data.len() as f64;

        // Calculate means
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        // Calculate slope and intercept
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (x, y) in data {
            numerator += (x - mean_x) * (y - mean_y);
            denominator += (x - mean_x) * (x - mean_x);
        }

        let slope = if denominator.abs() > f64::EPSILON {
            numerator / denominator
        } else {
            0.0
        };

        let intercept = mean_y - slope * mean_x;

        // Store coefficients [intercept, slope]
        self.coefficients = vec![intercept, slope];

        Ok(())
    }

    /// Fit polynomial using least squares (normal equations)
    fn fit_polynomial(&mut self, data: &[(f64, f64)], degree: usize) -> LearnedResult<()> {
        let _n = data.len();
        let m = degree + 1;

        // Build Vandermonde matrix and target vector
        // For polynomial y = c0 + c1*x + c2*x^2 + ...
        // X[i][j] = x[i]^j

        // Build X^T * X (symmetric matrix)
        let mut xtx = vec![vec![0.0; m]; m];
        let mut xty = vec![0.0; m];

        for (x, y) in data {
            let mut x_power = 1.0;
            for j in 0..m {
                xty[j] += x_power * y;
                let mut x_power2 = 1.0;
                for k in 0..m {
                    xtx[j][k] += x_power * x_power2;
                    x_power2 *= x;
                }
                x_power *= x;
            }
        }

        // Solve using Gaussian elimination
        self.coefficients = self.solve_linear_system(&mut xtx, &mut xty)?;

        Ok(())
    }

    /// Solve linear system Ax = b using Gaussian elimination with partial pivoting
    fn solve_linear_system(&self, a: &mut [Vec<f64>], b: &mut [f64]) -> LearnedResult<Vec<f64>> {
        let n = b.len();

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = a[i][i].abs();
            for k in (i + 1)..n {
                if a[k][i].abs() > max_val {
                    max_val = a[k][i].abs();
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                a.swap(i, max_row);
                b.swap(i, max_row);
            }

            // Check for singular matrix
            if a[i][i].abs() < 1e-12 {
                return Err(LearnedError::FittingFailed(
                    "singular matrix in least squares".to_string(),
                ));
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = a[k][i] / a[i][i];
                b[k] -= factor * b[i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = b[i];
            for j in (i + 1)..n {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }

        Ok(x)
    }

    /// Fit piecewise linear model
    fn fit_piecewise(&mut self, data: &[(f64, f64)], num_segments: usize) -> LearnedResult<()> {
        let num_segments = num_segments.max(1).min(data.len() / 2);
        let segment_size = data.len() / num_segments;

        self.segments.clear();

        for seg in 0..num_segments {
            let start_idx = seg * segment_size;
            let end_idx = if seg == num_segments - 1 {
                data.len()
            } else {
                (seg + 1) * segment_size
            };

            let segment_data: Vec<(f64, f64)> = data[start_idx..end_idx].to_vec();

            if segment_data.len() < 2 {
                continue;
            }

            // Fit linear to this segment
            let n = segment_data.len() as f64;
            let sum_x: f64 = segment_data.iter().map(|(x, _)| x).sum();
            let sum_y: f64 = segment_data.iter().map(|(_, y)| y).sum();
            let mean_x = sum_x / n;
            let mean_y = sum_y / n;

            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for (x, y) in &segment_data {
                numerator += (x - mean_x) * (y - mean_y);
                denominator += (x - mean_x) * (x - mean_x);
            }

            let slope = if denominator.abs() > f64::EPSILON {
                numerator / denominator
            } else {
                0.0
            };
            let intercept = mean_y - slope * mean_x;

            self.segments.push(PiecewiseSegment {
                start: segment_data[0].0,
                end: segment_data[segment_data.len() - 1].0,
                slope,
                intercept,
            });
        }

        Ok(())
    }

    /// Calculate training errors
    fn calculate_errors(&self, data: &[(f64, usize)]) -> (f64, f64) {
        let mut sum_error = 0.0;
        let mut max_error = 0.0f64;

        for (key, actual_pos) in data {
            let predicted = self.predict_position(*key);
            let error = (predicted - *actual_pos as f64).abs();
            sum_error += error;
            max_error = max_error.max(error);
        }

        let mae = sum_error / data.len() as f64;
        (mae, max_error)
    }

    /// Predict position for a key
    ///
    /// Returns the predicted position as a float (may need rounding)
    pub fn predict(&self, key: f64) -> LearnedResult<f64> {
        if !self.trained {
            return Err(LearnedError::NotTrained);
        }

        Ok(self.predict_position(key))
    }

    /// Internal prediction (assumes trained)
    fn predict_position(&self, key: f64) -> f64 {
        // Normalize key
        let norm_key = (key - self.min_key) / (self.max_key - self.min_key);
        let norm_key = norm_key.clamp(0.0, 1.0);

        let norm_pos = match self.model_type {
            ModelType::PiecewiseLinear(_) => self.predict_piecewise(norm_key),
            _ => self.predict_polynomial(norm_key),
        };

        // Convert back to position
        let pos = norm_pos * (self.num_records - 1).max(1) as f64;
        pos.clamp(0.0, (self.num_records - 1).max(0) as f64)
    }

    /// Predict using polynomial coefficients
    fn predict_polynomial(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut x_power = 1.0;
        for coef in &self.coefficients {
            result += coef * x_power;
            x_power *= x;
        }
        result.clamp(0.0, 1.0)
    }

    /// Predict using piecewise linear segments
    fn predict_piecewise(&self, x: f64) -> f64 {
        // Find appropriate segment
        for segment in &self.segments {
            if x >= segment.start && x <= segment.end {
                return (segment.slope * x + segment.intercept).clamp(0.0, 1.0);
            }
        }

        // Fallback to last segment
        if let Some(segment) = self.segments.last() {
            return (segment.slope * x + segment.intercept).clamp(0.0, 1.0);
        }

        x // Linear fallback
    }

    /// Get model coefficients (for debugging/serialization)
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model_uniform() {
        let mut model = LearnedIndexModel::linear();

        // Perfect linear data
        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64, i)).collect();

        let (mae, max_err) = model.train(&data).unwrap();
        assert!(model.is_trained());
        assert!(mae < 1.0, "MAE should be small: {}", mae);
        assert!(max_err < 2.0, "Max error should be small: {}", max_err);

        // Test prediction
        let pred = model.predict(50.0).unwrap();
        assert!(
            (pred - 50.0).abs() < 1.0,
            "Prediction should be close to 50: {}",
            pred
        );
    }

    #[test]
    fn test_linear_model_nonuniform() {
        let mut model = LearnedIndexModel::linear();

        // Quadratic distribution of keys
        let data: Vec<(f64, usize)> = (0..100).map(|i| ((i as f64).powi(2), i)).collect();

        let (mae, max_err) = model.train(&data).unwrap();
        assert!(model.is_trained());
        // Linear model won't fit quadratic data well
        assert!(mae > 0.0);
    }

    #[test]
    fn test_quadratic_model() {
        let mut model = LearnedIndexModel::new(ModelType::Quadratic);

        // Quadratic distribution of keys
        let data: Vec<(f64, usize)> = (0..100).map(|i| ((i as f64).powi(2), i)).collect();

        let (mae, _) = model.train(&data).unwrap();
        assert!(model.is_trained());
        // Quadratic model should fit quadratic data better
        assert!(mae < 5.0, "Quadratic model MAE should be low: {}", mae);
    }

    #[test]
    fn test_piecewise_linear() {
        let mut model = LearnedIndexModel::new(ModelType::PiecewiseLinear(10));

        // Non-uniform data
        let data: Vec<(f64, usize)> = (0..100)
            .map(|i| {
                let key = if i < 50 {
                    i as f64
                } else {
                    50.0 + (i - 50) as f64 * 2.0
                };
                (key, i)
            })
            .collect();

        let (mae, _) = model.train(&data).unwrap();
        assert!(model.is_trained());
        assert!(mae < 5.0, "Piecewise model should fit well: {}", mae);
    }

    #[test]
    fn test_prediction_bounds() {
        let mut model = LearnedIndexModel::linear();
        let data: Vec<(f64, usize)> = (0..100).map(|i| (i as f64 * 10.0, i)).collect();
        model.train(&data).unwrap();

        // Test key below min
        let pred = model.predict(-100.0).unwrap();
        assert!(pred >= 0.0, "Prediction should be clamped to >= 0");

        // Test key above max
        let pred = model.predict(2000.0).unwrap();
        assert!(pred <= 99.0, "Prediction should be clamped to <= 99");
    }

    #[test]
    fn test_untrained_error() {
        let model = LearnedIndexModel::linear();
        assert!(!model.is_trained());
        assert!(model.predict(50.0).is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let mut model = LearnedIndexModel::linear();
        let result = model.train(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_point_error() {
        let mut model = LearnedIndexModel::linear();
        let result = model.train(&[(1.0, 0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_key_bounds() {
        let mut model = LearnedIndexModel::linear();
        let data: Vec<(f64, usize)> = (0..100).map(|i| (100.0 + i as f64, i)).collect();
        model.train(&data).unwrap();

        let (min, max) = model.key_bounds();
        assert_eq!(min, 100.0);
        assert_eq!(max, 199.0);
    }
}
