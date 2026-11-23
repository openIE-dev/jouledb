//! Model Training
//!
//! Trains learned index models from query history

use super::model::ModelType;
use wasm_bindgen::prelude::*;

/// Training data
#[derive(Debug, Clone)]
pub struct TrainingData {
    /// Key-value pairs for training
    pub keys: Vec<String>,
    /// Positions (if known)
    pub positions: Vec<usize>,
}

/// Model trainer
pub struct ModelTrainer {
    model_type: ModelType,
}

impl ModelTrainer {
    /// Create new trainer
    pub fn new(model_type: ModelType) -> Self {
        Self { model_type }
    }

    /// Train model from data
    pub async fn train(&self, data: &TrainingData) -> Result<super::model::LearnedModel, String> {
        match self.model_type {
            ModelType::Linear => self.train_linear(data).await,
            ModelType::Neural => self.train_neural(data).await,
        }
    }

    /// Train linear model
    async fn train_linear(
        &self,
        data: &TrainingData,
    ) -> Result<super::model::LearnedModel, String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        if data.keys.len() != data.positions.len() {
            return Err("Keys and positions must have same length".to_string());
        }

        if data.keys.is_empty() {
            return Err("Training data is empty".to_string());
        }

        // Hash keys to numbers
        let mut hashes = Vec::new();
        for key in &data.keys {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            hashes.push(hasher.finish() as u32);
        }

        // Simple linear regression: position = slope * hash + intercept
        // Using least squares

        let n = hashes.len() as f32;
        let sum_x: f32 = hashes.iter().map(|&h| h as f32).sum();
        let sum_y: f32 = data.positions.iter().map(|&p| p as f32).sum();
        let sum_xy: f32 = hashes
            .iter()
            .zip(data.positions.iter())
            .map(|(&h, &p)| h as f32 * p as f32)
            .sum();
        let sum_x2: f32 = hashes.iter().map(|&h| (h as f32).powi(2)).sum();

        // Calculate slope and intercept
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate accuracy (R-squared)
        let mean_y = sum_y / n;
        let ss_res: f32 = hashes
            .iter()
            .zip(data.positions.iter())
            .map(|(&h, &p)| {
                let predicted = slope * h as f32 + intercept;
                (p as f32 - predicted).powi(2)
            })
            .sum();
        let ss_tot: f32 = data
            .positions
            .iter()
            .map(|&p| (p as f32 - mean_y).powi(2))
            .sum();

        let accuracy = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Create model parameters
        let parameters = super::model::ModelParameters {
            slope,
            intercept,
            accuracy: accuracy.max(0.0).min(1.0),
            neural_weights: None,
        };

        Ok(super::model::LearnedModel {
            model_type: super::model::ModelType::Linear,
            webnn_model: None,
            webnn_graph: None,
            parameters,
            backend: super::model::ComputeBackend::CPU,
        })
    }

    /// Train neural network using simple backpropagation
    ///
    /// Implements a 2-layer neural network:
    /// - Input: hashed key (normalized)
    /// - Hidden layer: 16 neurons with ReLU activation
    /// - Output: predicted position (normalized)
    async fn train_neural(
        &self,
        data: &TrainingData,
    ) -> Result<super::model::LearnedModel, String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        if data.keys.len() != data.positions.len() {
            return Err("Keys and positions must have same length".to_string());
        }

        if data.keys.len() < 10 {
            return Err("Neural training requires at least 10 samples".to_string());
        }

        // Hash keys to numbers and normalize
        let mut hashes: Vec<f32> = Vec::new();
        for key in &data.keys {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            hashes.push((hasher.finish() as f32) / (u64::MAX as f32));
        }

        // Normalize positions
        let max_pos = data.positions.iter().max().copied().unwrap_or(1) as f32;
        let positions: Vec<f32> = data.positions.iter().map(|&p| p as f32 / max_pos).collect();

        // Initialize neural network weights
        const HIDDEN_SIZE: usize = 16;
        const LEARNING_RATE: f32 = 0.01;
        const EPOCHS: usize = 100;

        // Xavier initialization for weights
        let seed = 42u64;
        let mut rng_state = seed;
        let xavier_input = (6.0f32 / (1.0 + HIDDEN_SIZE as f32)).sqrt();
        let xavier_hidden = (6.0f32 / (HIDDEN_SIZE as f32 + 1.0)).sqrt();

        // Input to hidden weights (1 x HIDDEN_SIZE)
        let mut w1: Vec<f32> = (0..HIDDEN_SIZE)
            .map(|_| {
                rng_state = xorshift64(&mut rng_state);
                ((rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0) * xavier_input
            })
            .collect();

        // Hidden biases
        let mut b1: Vec<f32> = vec![0.0; HIDDEN_SIZE];

        // Hidden to output weights (HIDDEN_SIZE x 1)
        let mut w2: Vec<f32> = (0..HIDDEN_SIZE)
            .map(|_| {
                rng_state = xorshift64(&mut rng_state);
                ((rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0) * xavier_hidden
            })
            .collect();

        // Output bias
        let mut b2: f32 = 0.0;

        // Training loop
        for _epoch in 0..EPOCHS {
            let mut total_loss = 0.0f32;

            for (x, &y) in hashes.iter().zip(positions.iter()) {
                // Forward pass
                // Hidden layer: h = ReLU(x * w1 + b1)
                let mut hidden: Vec<f32> = w1
                    .iter()
                    .zip(b1.iter())
                    .map(|(&w, &b)| relu(x * w + b))
                    .collect();

                // Output: y_pred = h · w2 + b2
                let y_pred: f32 = hidden
                    .iter()
                    .zip(w2.iter())
                    .map(|(&h, &w)| h * w)
                    .sum::<f32>()
                    + b2;

                // Loss: MSE
                let error = y_pred - y;
                total_loss += error * error;

                // Backward pass
                // Output gradient
                let d_output = 2.0 * error;

                // Gradients for w2 and b2
                let d_w2: Vec<f32> = hidden.iter().map(|&h| d_output * h).collect();
                let d_b2 = d_output;

                // Gradient through hidden layer
                let d_hidden: Vec<f32> = w2.iter().map(|&w| d_output * w).collect();

                // Apply ReLU derivative
                let d_hidden_relu: Vec<f32> = d_hidden
                    .iter()
                    .zip(hidden.iter())
                    .map(|(&dh, &h)| if h > 0.0 { dh } else { 0.0 })
                    .collect();

                // Gradients for w1 and b1
                let d_w1: Vec<f32> = d_hidden_relu.iter().map(|&dh| dh * x).collect();
                let d_b1: Vec<f32> = d_hidden_relu.clone();

                // Update weights
                for (w, &dw) in w2.iter_mut().zip(d_w2.iter()) {
                    *w -= LEARNING_RATE * dw;
                }
                b2 -= LEARNING_RATE * d_b2;

                for (w, &dw) in w1.iter_mut().zip(d_w1.iter()) {
                    *w -= LEARNING_RATE * dw;
                }
                for (b, &db) in b1.iter_mut().zip(d_b1.iter()) {
                    *b -= LEARNING_RATE * db;
                }
            }

            // Early stopping if loss is very low
            let avg_loss = total_loss / hashes.len() as f32;
            if avg_loss < 1e-6 {
                break;
            }
        }

        // Calculate accuracy on training data
        let mut correct = 0;
        let tolerance = 0.1; // 10% tolerance for position prediction

        for (x, &y) in hashes.iter().zip(positions.iter()) {
            let hidden: Vec<f32> = w1
                .iter()
                .zip(b1.iter())
                .map(|(&w, &b)| relu(x * w + b))
                .collect();
            let y_pred: f32 = hidden
                .iter()
                .zip(w2.iter())
                .map(|(&h, &w)| h * w)
                .sum::<f32>()
                + b2;

            if (y_pred - y).abs() < tolerance {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / hashes.len() as f32;

        // Pack neural weights into a single vector
        // Format: [w1..., b1..., w2..., b2, max_pos]
        let mut neural_weights = Vec::with_capacity(HIDDEN_SIZE * 2 + HIDDEN_SIZE + 2);
        neural_weights.extend_from_slice(&w1);
        neural_weights.extend_from_slice(&b1);
        neural_weights.extend_from_slice(&w2);
        neural_weights.push(b2);
        neural_weights.push(max_pos);

        let parameters = super::model::ModelParameters {
            slope: 0.0,
            intercept: 0.0,
            accuracy,
            neural_weights: Some(neural_weights),
        };

        Ok(super::model::LearnedModel {
            model_type: super::model::ModelType::Neural,
            webnn_model: None,
            webnn_graph: None,
            parameters,
            backend: super::model::ComputeBackend::CPU,
        })
    }
}

/// ReLU activation function
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Simple xorshift64 PRNG for weight initialization
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
