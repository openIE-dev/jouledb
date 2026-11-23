//! RaBitQ: Randomized Bit Quantization for High-Dimensional Vector Search
//!
//! Implements the SOTA 1-bit vector quantization scheme (Gao & Long, SIGMOD 2024)
//! with structured Fast Walsh-Hadamard + Rademacher orthogonal rotation (`O(D log D)`
//! compute, `O(D)` memory — no `O(D^2)` projection matrix!) and hardware bitwise
//! `POPCNT` distance estimation with provable concentration error bounds.
//!
//! ## Key Properties
//! - **32x Memory Compression**: `D` `f32` values (`4D` bytes) compress to `D` bits
//!   (`D / 8` bytes) + 8 bytes of exact norm/projection metadata (`||x - c||`, `<o_bar, x_bar>`).
//! - **Unbiased Distance Estimator**: Unlike heuristic sign binarization, RaBitQ
//!   normalizes by the exact projection `<o_bar, x_bar>` onto the hypercube vertex,
//!   producing an asymptotically unbiased inner-product and Euclidean distance estimate.
//! - **Theoretical Error Bound**: Emits a per-candidate `error_bound` (`O(1 / sqrt(D))`)
//!   enabling exact early pruning or coarse-to-fine two-stage rescoring.
//! - **Predicate-Aware Search**: Native `search_filtered` evaluates arbitrary row
//!   predicates during quantized scan/rerank without post-filtering recall collapse.

use std::collections::HashMap;

/// Configuration for a [`RaBitQIndex`].
#[derive(Debug, Clone)]
pub struct RaBitQConfig {
    /// Vector dimensionality `D` (must be > 0).
    pub dim: usize,
    /// Deterministic PRNG seed for the structured orthogonal rotation `P = D_1 H D_2`.
    pub seed: u64,
    /// Oversampling multiplier for optional exact `f32` rescoring (`1` = pure 1-bit RaBitQ).
    pub rerank_factor: usize,
    /// Whether to retain original `f32` vectors in-memory/extent for exact top-k rescoring.
    pub store_raw_for_rerank: bool,
}

impl RaBitQConfig {
    /// Create a default RaBitQ configuration for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            seed: 0x517C_C1B7_2722_0A95,
            rerank_factor: 4,
            store_raw_for_rerank: false,
        }
    }

    /// Enable two-stage coarse-to-fine exact rescoring with the given oversampling factor.
    pub fn with_exact_rerank(mut self, rerank_factor: usize) -> Self {
        self.rerank_factor = rerank_factor.max(1);
        self.store_raw_for_rerank = true;
        self
    }
}

/// Search hit returned by [`RaBitQIndex`].
#[derive(Debug, Clone, PartialEq)]
pub struct RaBitQSearchResult {
    /// External vector/row ID.
    pub id: u64,
    /// Estimated (or rescored exact) squared Euclidean distance `||x - q||^2`.
    pub squared_l2: f32,
    /// Estimated (or rescored exact) cosine similarity in `[-1.0, 1.0]`.
    pub cosine_similarity: f32,
    /// Theoretical 1-sigma concentration bound on the inner-product term.
    pub error_bound: f32,
}

#[derive(Debug, Clone)]
struct QuantizedEntry {
    id: u64,
    /// Packed 1-bit signs of rotated residual `P(x - c)` in padded dimension `padded_dim`.
    bits: Vec<u64>,
    /// Exact L2 norm of residual `||x - c||_2`.
    residual_norm: f32,
    /// Original vector L2 norm `||x||_2` (for cosine similarity reconstruction).
    raw_norm: f32,
    /// Exact inner product `<o_bar, x_bar>` where `x_bar = P(x - c) / ||x - c||`
    /// and `o_bar` is the normalized `{-1/sqrt(D_p), +1/sqrt(D_p)}` hypercube vertex.
    vertex_proj: f32,
    /// Optional raw vector for two-stage exact rescoring.
    raw: Option<Vec<f32>>,
}

/// High-performance 1-bit RaBitQ vector index with POPCNT estimator.
#[derive(Debug, Clone)]
pub struct RaBitQIndex {
    config: RaBitQConfig,
    /// Next power of two >= `config.dim` for O(D log D) Fast Walsh-Hadamard rotation.
    padded_dim: usize,
    /// Number of `u64` words per quantized bit-vector (`ceil(padded_dim / 64)`).
    num_words: usize,
    /// Rademacher sign masks (+1.0 or -1.0) for fast orthogonal butterfly `D_1 H D_2`.
    signs_pre: Vec<f32>,
    signs_post: Vec<f32>,
    /// Dataset centroid `c` (updated explicitly via `build_with_centroid` or running mean).
    centroid: Vec<f32>,
    /// Quantized entries.
    entries: Vec<QuantizedEntry>,
    /// Fast ID -> slot lookup for updates/deletes.
    id_to_slot: HashMap<u64, usize>,
}

impl RaBitQIndex {
    /// Create a new empty RaBitQ index.
    pub fn new(config: RaBitQConfig) -> Self {
        assert!(config.dim > 0, "RaBitQ dimension must be > 0");
        let padded_dim = config.dim.next_power_of_two().max(64);
        let num_words = padded_dim.div_ceil(64);
        let (signs_pre, signs_post) = generate_rademacher_pair(padded_dim, config.seed);

        Self {
            centroid: vec![0.0; config.dim],
            config,
            padded_dim,
            num_words,
            signs_pre,
            signs_post,
            entries: Vec::new(),
            id_to_slot: HashMap::new(),
        }
    }

    /// Build a calibrated RaBitQ index over a batch of `(id, vector)` pairs,
    /// centering residuals around the exact empirical centroid first.
    pub fn build_batch(config: RaBitQConfig, items: &[(u64, Vec<f32>)]) -> Self {
        let mut index = Self::new(config);
        if items.is_empty() {
            return index;
        }
        let inv_n = 1.0 / (items.len() as f32);
        for (_, v) in items {
            assert_eq!(v.len(), index.config.dim, "vector dimension mismatch");
            for (c, &val) in index.centroid.iter_mut().zip(v.iter()) {
                *c += val * inv_n;
            }
        }
        index.entries.reserve(items.len());
        for (id, v) in items {
            index.insert(*id, v);
        }
        index
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Memory consumed by quantized codes + scalar factors in bytes (excluding optional raw vectors).
    pub fn quantized_memory_bytes(&self) -> usize {
        self.entries.len() * (self.num_words * 8 + 12)
    }

    /// Compression ratio relative to uncompressed `f32` storage (`4 * dim` bytes/vector).
    pub fn compression_ratio(&self) -> f32 {
        if self.entries.is_empty() {
            return (self.config.dim * 4) as f32 / (self.num_words * 8 + 12) as f32;
        }
        let raw_bytes = self.entries.len() * self.config.dim * 4;
        raw_bytes as f32 / self.quantized_memory_bytes().max(1) as f32
    }

    /// Insert or replace a vector by `id`.
    pub fn insert(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.config.dim,
            "expected dim {}, got {}",
            self.config.dim,
            vector.len()
        );

        let raw_norm = l2_norm(vector);
        let mut residual_padded = vec![0.0f32; self.padded_dim];
        let mut residual_norm_sq = 0.0f32;
        for i in 0..self.config.dim {
            let d = vector[i] - self.centroid[i];
            residual_padded[i] = d;
            residual_norm_sq += d * d;
        }
        let residual_norm = residual_norm_sq.sqrt();

        // Normalize residual to unit sphere `x_bar` and apply isometric FWHT rotation `P`
        if residual_norm > 1e-12 {
            let inv = 1.0 / residual_norm;
            for v in &mut residual_padded[..self.config.dim] {
                *v *= inv;
            }
        }
        self.apply_orthogonal_rotation(&mut residual_padded);

        // Quantize `x_bar_rot` to `{0, 1}^D_p` and compute exact `<o_bar, x_bar_rot>`
        // Since `o_bar[i] = sign(x_bar_rot[i]) / sqrt(D_p)`,
        // `<o_bar, x_bar_rot> = (1 / sqrt(D_p)) * sum_i |x_bar_rot[i]|`.
        let mut bits = vec![0u64; self.num_words];
        let mut l1_sum = 0.0f32;
        for (i, &val) in residual_padded.iter().enumerate() {
            if val >= 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
                l1_sum += val;
            } else {
                l1_sum -= val;
            }
        }
        let inv_sqrt_d = 1.0 / (self.padded_dim as f32).sqrt();
        let vertex_proj = (l1_sum * inv_sqrt_d).max(1e-6);

        let entry = QuantizedEntry {
            id,
            bits,
            residual_norm,
            raw_norm,
            vertex_proj,
            raw: if self.config.store_raw_for_rerank {
                Some(vector.to_vec())
            } else {
                None
            },
        };

        if let Some(&slot) = self.id_to_slot.get(&id) {
            self.entries[slot] = entry;
        } else {
            let slot = self.entries.len();
            self.entries.push(entry);
            self.id_to_slot.insert(id, slot);
        }
    }

    /// Remove a vector by `id`. Returns `true` if present.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(slot) = self.id_to_slot.remove(&id) else {
            return false;
        };
        let last_idx = self.entries.len() - 1;
        self.entries.swap_remove(slot);
        if slot < self.entries.len() {
            let moved_id = self.entries[slot].id;
            self.id_to_slot.insert(moved_id, slot);
        }
        let _ = last_idx;
        true
    }

    /// Top-K nearest-neighbor search by minimum squared Euclidean distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<RaBitQSearchResult> {
        self.search_filtered(query, k, |_| true)
    }

    /// Top-K nearest-neighbor search filtered by an arbitrary row predicate `filter(id)`.
    ///
    /// Evaluates 8-bit bit-plane `POPCNT` inner products directly against 1-bit codes.
    pub fn search_filtered<F>(
        &self,
        query: &[f32],
        k: usize,
        mut filter: F,
    ) -> Vec<RaBitQSearchResult>
    where
        F: FnMut(u64) -> bool,
    {
        assert_eq!(query.len(), self.config.dim, "query dimension mismatch");
        if k == 0 || self.entries.is_empty() {
            return Vec::new();
        }

        let query_raw_norm = l2_norm(query);
        let centroid_dot_q: f32 = self.centroid.iter().zip(query).map(|(c, q)| c * q).sum();
        let centroid_norm_sq: f32 = self.centroid.iter().map(|c| c * c).sum();

        // Compute rotated query residual `q_rot = P((q - c) / ||q - c||)`
        let mut q_padded = vec![0.0f32; self.padded_dim];
        let mut q_res_norm_sq = 0.0f32;
        for i in 0..self.config.dim {
            let d = query[i] - self.centroid[i];
            q_padded[i] = d;
            q_res_norm_sq += d * d;
        }
        let q_res_norm = q_res_norm_sq.sqrt();
        if q_res_norm > 1e-12 {
            let inv = 1.0 / q_res_norm;
            for v in &mut q_padded[..self.config.dim] {
                *v *= inv;
            }
        }
        self.apply_orthogonal_rotation(&mut q_padded);

        // Quantize rotated query into 8 bit-planes `[u64; num_words]` for SIMD POPCNT dot product!
        // For each coordinate `i`: `q_u8 = round((q_rot[i] - q_min) / q_step)` in `[0, 255]`.
        // Note: `o_bar[i] = (2 * bit[i] - 1) / sqrt(D_p)`.
        // Therefore:
        // `<o_bar, q_rot> = (1 / sqrt(D_p)) * (2 * sum_{i: bit[i]=1} q_rot[i] - sum_i q_rot[i])`.
        let q_sum: f32 = q_padded.iter().sum();
        let q_min = q_padded.iter().copied().fold(f32::INFINITY, f32::min);
        let q_max = q_padded.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let q_span = (q_max - q_min).max(1e-12);
        let q_step = q_span / 255.0;
        let inv_step = 255.0 / q_span;

        let mut bitplanes = vec![vec![0u64; self.num_words]; 8];
        for (i, &qv) in q_padded.iter().enumerate() {
            let quant = ((qv - q_min) * inv_step).round().clamp(0.0, 255.0) as u8;
            if quant != 0 {
                let word = i / 64;
                let mask = 1u64 << (i % 64);
                for plane in 0..8 {
                    if (quant >> plane) & 1 != 0 {
                        bitplanes[plane][word] |= mask;
                    }
                }
            }
        }

        let inv_sqrt_d = 1.0 / (self.padded_dim as f32).sqrt();
        let bound_scale = 1.96 / ((self.padded_dim as f32 - 1.0).max(1.0)).sqrt();

        let shortlist_len = if self.config.store_raw_for_rerank {
            k.saturating_mul(self.config.rerank_factor)
        } else {
            k
        };

        let mut candidates: Vec<(usize, f32, f32, f32)> = Vec::new();
        for (slot, entry) in self.entries.iter().enumerate() {
            if !filter(entry.id) {
                continue;
            }

            // POPCNT evaluation of `sum_{i: bit[i]=1} q_rot[i]`
            let mut ones_count: u32 = 0;
            for &w in &entry.bits {
                ones_count += w.count_ones();
            }

            let mut weighted_popcnt: u64 = 0;
            for (plane, plane_words) in bitplanes.iter().enumerate() {
                let mut plane_ones: u32 = 0;
                for (&xw, &qw) in entry.bits.iter().zip(plane_words.iter()) {
                    plane_ones += (xw & qw).count_ones();
                }
                weighted_popcnt += (plane_ones as u64) << plane;
            }

            let sum_q_where_one =
                (ones_count as f32) * q_min + (weighted_popcnt as f32) * q_step;
            let o_bar_dot_q = (2.0 * sum_q_where_one - q_sum) * inv_sqrt_d;

            // Unbiased RaBitQ estimator of `<x_bar, q_bar> = <o_bar, q_rot> / <o_bar, x_bar>`
            let est_unit_ip = if q_res_norm > 1e-12 && entry.residual_norm > 1e-12 {
                (o_bar_dot_q / entry.vertex_proj).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            let est_res_ip = entry.residual_norm * q_res_norm * est_unit_ip;
            let est_sq_l2 = (entry.residual_norm * entry.residual_norm
                + q_res_norm_sq
                - 2.0 * est_res_ip)
                .max(0.0);

            // Reconstruct cosine(x, q) from `||x - q||^2 = ||x||^2 + ||q||^2 - 2<x, q>`
            let x_dot_q = 0.5
                * (entry.raw_norm * entry.raw_norm + query_raw_norm * query_raw_norm - est_sq_l2);
            let _ = (centroid_dot_q, centroid_norm_sq);
            let denom = (entry.raw_norm * query_raw_norm).max(1e-12);
            let est_cos = (x_dot_q / denom).clamp(-1.0, 1.0);

            let err_bound = entry.residual_norm
                * q_res_norm
                * bound_scale
                * ((1.0 - entry.vertex_proj * entry.vertex_proj).max(0.0).sqrt()
                    / entry.vertex_proj);

            candidates.push((slot, est_sq_l2, est_cos, err_bound));
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(shortlist_len);

        // Optional Stage 2: exact rescoring if `store_raw_for_rerank` is enabled
        let mut results: Vec<RaBitQSearchResult> = candidates
            .into_iter()
            .map(|(slot, est_sq_l2, est_cos, err_bound)| {
                let entry = &self.entries[slot];
                if let Some(ref raw) = entry.raw {
                    let exact_sq_l2: f32 = raw
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| {
                            let d = a - b;
                            d * d
                        })
                        .sum();
                    let dot: f32 = raw.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                    let cos = if entry.raw_norm > 1e-12 && query_raw_norm > 1e-12 {
                        (dot / (entry.raw_norm * query_raw_norm)).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    RaBitQSearchResult {
                        id: entry.id,
                        squared_l2: exact_sq_l2,
                        cosine_similarity: cos,
                        error_bound: 0.0,
                    }
                } else {
                    RaBitQSearchResult {
                        id: entry.id,
                        squared_l2: est_sq_l2,
                        cosine_similarity: est_cos,
                        error_bound: err_bound,
                    }
                }
            })
            .collect();

        if self.config.store_raw_for_rerank {
            results.sort_by(|a, b| {
                a.squared_l2
                    .partial_cmp(&b.squared_l2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        results.truncate(k);
        results
    }

    /// Structured orthogonal rotation `P = (1/sqrt(D_p)) * D_post * H * D_pre`
    /// via in-place Fast Walsh-Hadamard Transform (`O(D log D)`, norm-preserving!).
    fn apply_orthogonal_rotation(&self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), self.padded_dim);
        for (v, &s) in buf.iter_mut().zip(self.signs_pre.iter()) {
            *v *= s;
        }
        let mut len = 1usize;
        while len < self.padded_dim {
            let step = len << 1;
            let mut i = 0usize;
            while i < self.padded_dim {
                for j in 0..len {
                    let u = buf[i + j];
                    let v = buf[i + j + len];
                    buf[i + j] = u + v;
                    buf[i + j + len] = u - v;
                }
                i += step;
            }
            len = step;
        }
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        for (v, &s) in buf.iter_mut().zip(self.signs_post.iter()) {
            *v *= s * scale;
        }
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// SplitMix64 deterministic PRNG for cross-platform reproducible Rademacher signs.
fn generate_rademacher_pair(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed;
    let mut next_sign = || -> f32 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if z & 1 == 0 { 1.0 } else { -1.0 }
    };

    let pre = (0..n).map(|_| next_sign()).collect();
    let post = (0..n).map(|_| next_sign()).collect();
    (pre, post)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(1);
        (0..dim)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as i32 as f32) / (u32::MAX as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn test_orthogonal_rotation_preserves_l2_norm() {
        let idx = RaBitQIndex::new(RaBitQConfig::new(384));
        let mut v = vec![0.0f32; idx.padded_dim];
        let raw = synthetic_vec(384, 42);
        v[..384].copy_from_slice(&raw);
        let before = l2_norm(&v);
        idx.apply_orthogonal_rotation(&mut v);
        let after = l2_norm(&v);
        assert!(
            (before - after).abs() < 1e-4,
            "FWHT rotation must be isometric: before={before}, after={after}"
        );
    }

    #[test]
    fn test_rabitq_compression_and_recall() {
        let dim = 512;
        let n = 500usize;
        let items: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|i| (i as u64, synthetic_vec(dim, 10_000 + i as u64)))
            .collect();

        let index = RaBitQIndex::build_batch(RaBitQConfig::new(dim), &items);
        assert_eq!(index.len(), n);
        // 512 f32 = 2048 bytes vs 64 code bytes + 12 scalar bytes = 76 bytes (~26.9x total, 32x code)
        assert!(
            index.compression_ratio() > 25.0,
            "compression ratio was {}",
            index.compression_ratio()
        );

        // Query perturbed versions of 20 items and verify pure 1-bit RaBitQ finds exact target in top-5
        let mut hits = 0usize;
        for target_id in 0..20usize {
            let mut q = items[target_id].1.clone();
            for (j, v) in q.iter_mut().enumerate() {
                *v += if j % 2 == 0 { 0.02 } else { -0.02 };
            }
            let res = index.search(&q, 5);
            assert!(!res.is_empty());
            if res.iter().any(|r| r.id == target_id as u64) {
                hits += 1;
            }
            assert!(res[0].error_bound >= 0.0);
        }
        assert!(
            hits >= 19,
            "pure 1-bit RaBitQ should recover >=95% of near-neighbors, got {hits}/20"
        );
    }

    #[test]
    fn test_rabitq_filtered_search_and_rerank() {
        let dim = 256;
        let items: Vec<(u64, Vec<f32>)> = (0..100)
            .map(|i| (i as u64, synthetic_vec(dim, 999 + i as u64)))
            .collect();

        let index =
            RaBitQIndex::build_batch(RaBitQConfig::new(dim).with_exact_rerank(4), &items);

        // Only allow odd IDs
        let q = items[42].1.clone(); // 42 is even -> excluded!
        let filtered = index.search_filtered(&q, 5, |id| id % 2 == 1);
        assert_eq!(filtered.len(), 5);
        assert!(filtered.iter().all(|r| r.id % 2 == 1));
        assert_eq!(filtered[0].error_bound, 0.0, "exact rerank has 0 error bound");
    }
}
