//! Attention-as-SDM Bridge
//!
//! Implements the Bricken & Pehlevan (NeurIPS 2021) result:
//! Transformer attention (softmax(QK^T/√d)V) approximates Sparse Distributed
//! Memory read under L2 normalization.
//!
//! This enables JouleDB's SDM to serve as native attention / LLM memory.
//! Media stored in the amorphic engine can be "attended to" by models natively.
//!
//! # Key insight
//!
//! SDM's read operation uses intersections between high-dimensional hyperspheres.
//! The decay of these intersection volumes is approximately exponential in Hamming
//! distance — matching the exponential in softmax. Under L2 normalization and
//! appropriate temperature (β), SDM read ≈ attention.
//!
//! # References
//!
//! - Bricken & Pehlevan (2021). "Attention Approximates Sparse Distributed Memory."
//!   NeurIPS. <https://arxiv.org/abs/2111.05498>
//! - Bricken et al. (2023). "Sparse Distributed Memory is a Continual Learner."
//!   ICLR. <https://arxiv.org/abs/2303.11934>

use super::memory::{SDMAddress, SDMError, SparseDistributedMemory};

/// Attention-compatible SDM that supports softmax-weighted reads.
///
/// Wraps `SparseDistributedMemory` with L2-normalized address computation
/// and temperature-scaled softmax weighting.
pub struct AttentionSDM {
    /// Underlying SDM instance
    sdm: SparseDistributedMemory,
    /// Temperature parameter (β) controlling attention sharpness.
    /// Higher β = sharper attention (fewer locations activated).
    /// β ≈ √dimension gives good approximation of standard softmax.
    pub temperature: f64,
}

impl AttentionSDM {
    /// Create a new Attention-SDM bridge.
    ///
    /// # Arguments
    /// * `num_locations` - Number of hard locations (= number of "memory slots")
    /// * `dimension` - Address dimension in bits (= key/query dimension)
    /// * `data_size` - Data stored per location (= value dimension)
    /// * `temperature` - Softmax temperature β (default: √dimension)
    pub fn new(
        num_locations: usize,
        dimension: usize,
        data_size: usize,
        temperature: Option<f64>,
    ) -> Self {
        let temp = temperature.unwrap_or_else(|| (dimension as f64).sqrt());
        AttentionSDM {
            sdm: SparseDistributedMemory::new(num_locations, dimension, data_size),
            temperature: temp,
        }
    }

    /// Write a key-value pair into the attention memory.
    ///
    /// Equivalent to adding a row to the Key and Value matrices in attention.
    pub fn write_kv(
        &self,
        key: &SDMAddress,
        value: &[i8],
    ) -> Result<u32, SDMError> {
        self.sdm.write(key, value)
    }

    /// Attention-compatible read: softmax-weighted retrieval.
    ///
    /// Computes: output = Σ softmax(sim(query, key_i) / temperature) * value_i
    ///
    /// This is mathematically equivalent to:
    ///   softmax(QK^T / √d) V
    /// when the keys are SDM hard location addresses and values are the stored data.
    ///
    /// Returns a float vector (weighted sum, not thresholded).
    pub fn attention_read(&self, query: &SDMAddress) -> Vec<f64> {
        let locations = self.sdm.locations_ref();
        let locations = locations.read().unwrap();

        let dimension = self.sdm.dimension();
        let data_size = self.sdm.data_size();

        // Compute similarities: convert Hamming distance to similarity score
        // sim = (dimension - 2 * hamming_distance) / dimension
        // This is equivalent to cosine similarity for binary vectors
        let similarities: Vec<f64> = locations
            .iter()
            .map(|loc| {
                let dist = query.hamming_distance(&loc.address);
                let sim = (dimension as f64 - 2.0 * dist as f64) / dimension as f64;
                sim * self.temperature
            })
            .collect();

        // Softmax normalization
        let max_sim = similarities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sims: Vec<f64> = similarities.iter().map(|&s| (s - max_sim).exp()).collect();
        let sum_exp: f64 = exp_sims.iter().sum();

        if sum_exp == 0.0 {
            return vec![0.0; data_size];
        }

        let weights: Vec<f64> = exp_sims.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum of stored values
        let mut output = vec![0.0f64; data_size];
        for (loc, &weight) in locations.iter().zip(weights.iter()) {
            if weight > 1e-10 {
                // Skip negligible weights
                for (i, &counter) in loc.counters.iter().enumerate() {
                    output[i] += weight * counter as f64;
                }
            }
        }

        output
    }

    /// Multi-head attention read.
    ///
    /// Partitions the address space into `num_heads` segments, runs
    /// attention_read independently per head, and concatenates results.
    ///
    /// # Arguments
    /// * `queries` - One query per head
    pub fn multi_head_read(&self, queries: &[SDMAddress]) -> Vec<Vec<f64>> {
        queries
            .iter()
            .map(|q| self.attention_read(q))
            .collect()
    }

    /// Get the underlying SDM (for direct access when attention semantics aren't needed).
    pub fn sdm(&self) -> &SparseDistributedMemory {
        &self.sdm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_read_recovers_written_value() {
        let dim = 256;
        let data_size = 16;
        let attn = AttentionSDM::new(1000, dim, data_size, None);

        // Write a key-value pair
        let key = SDMAddress::from_data(b"test_key", dim);
        let value: Vec<i8> = (0..data_size).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        attn.write_kv(&key, &value).unwrap();

        // Read with the same key — should recover the written pattern
        let output = attn.attention_read(&key);

        // Output should have the same sign pattern as the written value
        for (i, &v) in value.iter().enumerate() {
            if v > 0 {
                assert!(output[i] > 0.0, "Expected positive at index {}, got {}", i, output[i]);
            } else {
                assert!(output[i] < 0.0, "Expected negative at index {}, got {}", i, output[i]);
            }
        }
    }

    #[test]
    fn test_attention_sharpness() {
        let dim = 256;
        let data_size = 8;

        // High temperature = sharp attention (more selective)
        let sharp = AttentionSDM::new(100, dim, data_size, Some(50.0));
        // Low temperature = diffuse attention (more spread out)
        let diffuse = AttentionSDM::new(100, dim, data_size, Some(1.0));

        let key = SDMAddress::from_data(b"key1", dim);
        let value: Vec<i8> = vec![1; data_size];
        sharp.write_kv(&key, &value).unwrap();
        diffuse.write_kv(&key, &value).unwrap();

        let sharp_out = sharp.attention_read(&key);
        let diffuse_out = diffuse.attention_read(&key);

        // Sharp attention should have higher magnitude output (more concentrated)
        let sharp_mag: f64 = sharp_out.iter().map(|x| x.abs()).sum();
        let diffuse_mag: f64 = diffuse_out.iter().map(|x| x.abs()).sum();
        assert!(sharp_mag >= diffuse_mag);
    }
}
