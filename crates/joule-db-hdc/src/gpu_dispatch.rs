//! GPU-accelerated HDC operations via ComputeBackend dispatch
//!
//! Provides a `GpuDispatcher` that routes batch HDC operations (similarity search,
//! bind, bundle) to a GPU backend when the batch size exceeds configurable thresholds.
//! Below those thresholds, callers should fall back to CPU-based operations.
//!
//! ## Architecture
//!
//! ```text
//! GpuDispatcher  →  dyn ComputeBackend  ←  WgpuComputeBackend (joule-db-gpu)
//! ```
//!
//! `joule-db-hdc` never depends on `joule-db-gpu` directly. Callers inject a
//! `dyn ComputeBackend` at construction time, enabling any backend (GPU, NPU, FPGA).
//!
//! ## Data Layout
//!
//! CPU (`BinaryHyperVector` / `BinaryHV`): `Vec<u64>` — 64 bits per word.
//! GPU shaders: `array<u32>` — 32 bits per word.
//!
//! Conversion is handled transparently via `bytemuck::cast_slice`. On little-endian
//! platforms (all modern targets), this is zero-copy. The `ComputeOp::BinaryHD*`
//! variants document `num_words` as u64-word count; the backend converts to u32-word
//! count internally before dispatching to shaders.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use joule_db_core::StorageError;
use joule_db_core::persistence::compute::{
    BufferHandle, BufferUsage, ComputeBackend, ComputeOp, DeviceType,
};

/// Thread-safe handle to a GPU compute backend.
pub type GpuBackend = Arc<Mutex<dyn ComputeBackend>>;

/// Configuration for GPU dispatch thresholds.
///
/// Operations below these thresholds are better served by CPU due to GPU dispatch
/// overhead (~50-200us for buffer creation, upload, kernel launch, readback).
#[derive(Debug, Clone)]
pub struct GpuHdcConfig {
    /// Minimum number of candidate vectors for GPU similarity search (default: 64)
    pub similarity_threshold: usize,
    /// Minimum number of vectors for GPU bundle (default: 32)
    pub bundle_threshold: usize,
    /// Minimum number of pairs for GPU bind (default: 128)
    pub bind_threshold: usize,
}

impl Default for GpuHdcConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 64,
            bundle_threshold: 32,
            bind_threshold: 128,
        }
    }
}

/// Dynamic threshold overrides from the adaptive controller.
///
/// When the controller detects CPU pressure and GPU idle capacity,
/// it lowers these thresholds to offload more work. When GPU is
/// overloaded or thermal, it raises them.
#[derive(Debug, Clone, Copy)]
pub struct DynamicThresholds {
    /// Similarity search threshold (num vectors).
    pub similarity: usize,
    /// Bundle threshold (num vectors).
    pub bundle: usize,
    /// Bind threshold (num pairs).
    pub bind: usize,
}

/// Persistent GPU-resident buffer for data that is uploaded once and reused across calls.
struct PersistentCodebook {
    buffer_handle: BufferHandle,
    num_vectors: usize,
    dim_u64_words: usize,
    total_bits: usize,
}

/// Pre-allocated workspace buffers for decode operations.
/// Avoids repeated GPU buffer allocation/deallocation per decode call.
struct DecodeWorkspace {
    query_buffer: BufferHandle,
    query_capacity_bytes: u64,
}

/// GPU dispatcher for batch HDC operations.
///
/// Wraps a `dyn ComputeBackend` and provides high-level batch methods that handle
/// buffer management, data conversion (u64 ↔ u32), and GPU dispatch.
///
/// # Example
///
/// ```rust,ignore
/// use joule_db_hdc::gpu_dispatch::{GpuDispatcher, GpuBackend};
///
/// // Create a GPU backend (from joule-db-gpu)
/// let backend: GpuBackend = Arc::new(Mutex::new(wgpu_backend));
/// let gpu = GpuDispatcher::new(backend);
///
/// // Batch similarity search: query vs 1000 vectors, each 512 dimensions (8 u64 words)
/// let scores = gpu.batch_similarity(query_words, all_vectors_flat, 1000, 8)?;
/// ```
pub struct GpuDispatcher {
    backend: GpuBackend,
    config: GpuHdcConfig,
    /// Dynamic threshold overrides from the adaptive controller.
    /// 0 = no override (use static config defaults).
    dynamic_similarity_threshold: AtomicUsize,
    dynamic_bundle_threshold: AtomicUsize,
    dynamic_bind_threshold: AtomicUsize,
    /// Persistent codebook buffer (uploaded once, reused across calls)
    codebook: Mutex<Option<PersistentCodebook>>,
    /// Pre-allocated workspace for decode operations
    workspace: Mutex<Option<DecodeWorkspace>>,
}

impl GpuDispatcher {
    /// Create a new GPU dispatcher with default thresholds.
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            backend,
            config: GpuHdcConfig::default(),
            dynamic_similarity_threshold: AtomicUsize::new(0),
            dynamic_bundle_threshold: AtomicUsize::new(0),
            dynamic_bind_threshold: AtomicUsize::new(0),
            codebook: Mutex::new(None),
            workspace: Mutex::new(None),
        }
    }

    /// Create a GPU dispatcher with custom configuration.
    pub fn with_config(backend: GpuBackend, config: GpuHdcConfig) -> Self {
        Self {
            backend,
            dynamic_similarity_threshold: AtomicUsize::new(0),
            dynamic_bundle_threshold: AtomicUsize::new(0),
            dynamic_bind_threshold: AtomicUsize::new(0),
            config,
            codebook: Mutex::new(None),
            workspace: Mutex::new(None),
        }
    }

    /// Get the dispatch configuration.
    pub fn config(&self) -> &GpuHdcConfig {
        &self.config
    }

    /// Check whether the backend is a real GPU (not CPU fallback).
    pub fn is_gpu_available(&self) -> bool {
        let backend = self.backend.lock().unwrap();
        matches!(
            backend.capabilities().device_type,
            DeviceType::Gpu | DeviceType::Metal | DeviceType::Cuda | DeviceType::Vulkan
        )
    }

    /// Update dynamic dispatch thresholds from the adaptive controller.
    ///
    /// Call this periodically (e.g., after each `AdaptiveController::tick()`).
    /// Pass `None` to revert to static config defaults.
    pub fn update_thresholds(&self, overrides: Option<DynamicThresholds>) {
        match overrides {
            Some(ov) => {
                self.dynamic_similarity_threshold
                    .store(ov.similarity, Ordering::Release);
                self.dynamic_bundle_threshold
                    .store(ov.bundle, Ordering::Release);
                self.dynamic_bind_threshold
                    .store(ov.bind, Ordering::Release);
            }
            None => {
                self.dynamic_similarity_threshold
                    .store(0, Ordering::Release);
                self.dynamic_bundle_threshold.store(0, Ordering::Release);
                self.dynamic_bind_threshold.store(0, Ordering::Release);
            }
        }
    }

    /// Check if the batch size meets the threshold for GPU similarity search.
    ///
    /// Uses dynamic threshold if set by the adaptive controller,
    /// otherwise falls back to the static config default.
    pub fn should_dispatch_similarity(&self, num_vectors: usize) -> bool {
        num_vectors >= self.effective_similarity_threshold()
    }

    /// Check if the batch size meets the threshold for GPU bundle.
    pub fn should_dispatch_bundle(&self, num_vectors: usize) -> bool {
        num_vectors >= self.effective_bundle_threshold()
    }

    /// Check if the batch size meets the threshold for GPU bind.
    pub fn should_dispatch_bind(&self, num_pairs: usize) -> bool {
        num_pairs >= self.effective_bind_threshold()
    }

    /// Get the effective similarity threshold (dynamic if set, else static).
    fn effective_similarity_threshold(&self) -> usize {
        let dyn_val = self.dynamic_similarity_threshold.load(Ordering::Acquire);
        if dyn_val > 0 {
            dyn_val
        } else {
            self.config.similarity_threshold
        }
    }

    /// Get the effective bundle threshold (dynamic if set, else static).
    fn effective_bundle_threshold(&self) -> usize {
        let dyn_val = self.dynamic_bundle_threshold.load(Ordering::Acquire);
        if dyn_val > 0 {
            dyn_val
        } else {
            self.config.bundle_threshold
        }
    }

    /// Get the effective bind threshold (dynamic if set, else static).
    fn effective_bind_threshold(&self) -> usize {
        let dyn_val = self.dynamic_bind_threshold.load(Ordering::Acquire);
        if dyn_val > 0 {
            dyn_val
        } else {
            self.config.bind_threshold
        }
    }

    /// Batch similarity search: compute Hamming similarity of a query against many vectors.
    ///
    /// Returns similarity scores as `u32` values (number of matching bits) for each vector.
    /// To convert to normalized similarity: `score as f64 / (dim_u64_words * 64) as f64`.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector as `&[u64]` with `dim_u64_words` elements
    /// * `vectors` - Flat concatenation of all candidate vectors (`num_vectors * dim_u64_words` u64s)
    /// * `num_vectors` - Number of candidate vectors
    /// * `dim_u64_words` - Number of u64 words per vector (e.g., 8 for 512 dimensions)
    ///
    /// # Returns
    ///
    /// `Vec<u32>` of length `num_vectors`, where each entry is the similarity score
    /// (total bits - Hamming distance = number of matching bits).
    pub fn batch_similarity(
        &self,
        query: &[u64],
        vectors: &[u64],
        num_vectors: usize,
        dim_u64_words: usize,
    ) -> Result<Vec<u32>, StorageError> {
        debug_assert_eq!(query.len(), dim_u64_words);
        debug_assert_eq!(vectors.len(), num_vectors * dim_u64_words);

        let mut backend = self
            .backend
            .lock()
            .map_err(|e| StorageError::Backend(format!("Failed to lock GPU backend: {}", e)))?;

        // Upload query vector
        let query_bytes = bytemuck::cast_slice::<u64, u8>(query);
        let query_buf = backend.create_buffer(
            query_bytes.len() as u64,
            BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
            Some("hdc_similarity_query"),
        )?;
        backend.write_buffer(query_buf, 0, query_bytes)?;

        // Upload candidate vectors
        let vectors_bytes = bytemuck::cast_slice::<u64, u8>(vectors);
        let vectors_buf = backend.create_buffer(
            vectors_bytes.len() as u64,
            BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
            Some("hdc_similarity_vectors"),
        )?;
        backend.write_buffer(vectors_buf, 0, vectors_bytes)?;

        // Execute similarity search
        let result = backend.execute(
            ComputeOp::BinaryHDSimilarity {
                num_vectors: num_vectors as u32,
                num_words: dim_u64_words as u32,
            },
            &[query_buf, vectors_buf],
        )?;

        // Read back similarity scores (u32 per vector)
        let output_bytes =
            backend.read_buffer(result.output_buffer, 0, (num_vectors as u64) * 4)?;

        // Convert bytes to u32 scores
        let scores: Vec<u32> = output_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Clean up buffers
        backend.destroy_buffer(query_buf)?;
        backend.destroy_buffer(vectors_buf)?;
        backend.destroy_buffer(result.output_buffer)?;

        Ok(scores)
    }

    /// Batch bind: XOR pairs of vectors together on GPU in a single dispatch.
    ///
    /// Uploads all pairs at once and processes them in one GPU kernel launch,
    /// avoiding the per-pair dispatch overhead that dominated the previous implementation.
    ///
    /// # Arguments
    ///
    /// * `vectors_a` - First vectors, flat concatenation (`num_pairs * dim_u64_words` u64s)
    /// * `vectors_b` - Second vectors, flat concatenation (`num_pairs * dim_u64_words` u64s)
    /// * `num_pairs` - Number of vector pairs to bind
    /// * `dim_u64_words` - Number of u64 words per vector
    ///
    /// # Returns
    ///
    /// `Vec<u64>` of length `num_pairs * dim_u64_words` — the XOR results.
    pub fn batch_bind(
        &self,
        vectors_a: &[u64],
        vectors_b: &[u64],
        num_pairs: usize,
        dim_u64_words: usize,
    ) -> Result<Vec<u64>, StorageError> {
        debug_assert_eq!(vectors_a.len(), num_pairs * dim_u64_words);
        debug_assert_eq!(vectors_b.len(), num_pairs * dim_u64_words);

        let mut backend = self
            .backend
            .lock()
            .map_err(|e| StorageError::Backend(format!("Failed to lock GPU backend: {}", e)))?;

        // Upload ALL vectors_a in one shot
        let a_bytes = bytemuck::cast_slice::<u64, u8>(vectors_a);
        let a_buf = backend.create_buffer(
            a_bytes.len() as u64,
            BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
            Some("hdc_batch_bind_a"),
        )?;
        backend.write_buffer(a_buf, 0, a_bytes)?;

        // Upload ALL vectors_b in one shot
        let b_bytes = bytemuck::cast_slice::<u64, u8>(vectors_b);
        let b_buf = backend.create_buffer(
            b_bytes.len() as u64,
            BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
            Some("hdc_batch_bind_b"),
        )?;
        backend.write_buffer(b_buf, 0, b_bytes)?;

        // Single dispatch for ALL pairs
        let result = backend.execute(
            ComputeOp::BinaryHDBatchBind {
                num_pairs: num_pairs as u32,
                num_words: dim_u64_words as u32,
            },
            &[a_buf, b_buf],
        )?;

        // Single readback
        let total_bytes = (num_pairs * dim_u64_words) as u64 * 8;
        let output_bytes = backend.read_buffer(result.output_buffer, 0, total_bytes)?;

        // Reassemble u64s from raw bytes
        let result_words: Vec<u64> = output_bytes
            .chunks_exact(8)
            .map(|chunk| {
                u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        // Clean up
        backend.destroy_buffer(a_buf)?;
        backend.destroy_buffer(b_buf)?;
        backend.destroy_buffer(result.output_buffer)?;

        Ok(result_words)
    }

    /// Batch bundle: majority-vote multiple vectors into one on GPU.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Flat concatenation of all vectors (`num_vectors * dim_u64_words` u64s)
    /// * `num_vectors` - Number of vectors to bundle
    /// * `dim_u64_words` - Number of u64 words per vector
    ///
    /// # Returns
    ///
    /// `Vec<u64>` of length `dim_u64_words` — the majority-voted result.
    pub fn batch_bundle(
        &self,
        vectors: &[u64],
        num_vectors: usize,
        dim_u64_words: usize,
    ) -> Result<Vec<u64>, StorageError> {
        debug_assert_eq!(vectors.len(), num_vectors * dim_u64_words);

        let mut backend = self
            .backend
            .lock()
            .map_err(|e| StorageError::Backend(format!("Failed to lock GPU backend: {}", e)))?;

        // Upload all vectors
        let vectors_bytes = bytemuck::cast_slice::<u64, u8>(vectors);
        let vectors_buf = backend.create_buffer(
            vectors_bytes.len() as u64,
            BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
            Some("hdc_bundle_vectors"),
        )?;
        backend.write_buffer(vectors_buf, 0, vectors_bytes)?;

        // Execute bundle
        let result = backend.execute(
            ComputeOp::BinaryHDBundle {
                num_vectors: num_vectors as u32,
                num_words: dim_u64_words as u32,
            },
            &[vectors_buf],
        )?;

        // Read back result vector
        let output_bytes =
            backend.read_buffer(result.output_buffer, 0, (dim_u64_words as u64) * 8)?;

        // Reassemble u64s from raw bytes
        let result_words: Vec<u64> = output_bytes
            .chunks_exact(8)
            .map(|chunk| {
                u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        // Clean up
        backend.destroy_buffer(vectors_buf)?;
        backend.destroy_buffer(result.output_buffer)?;

        Ok(result_words)
    }

    /// Decode a single byte position from a noisy holographic vector on GPU.
    ///
    /// Given an already-unshifted query vector, compute similarity against all 256
    /// byte codebook vectors in a single GPU dispatch. Returns the best-matching byte
    /// and its bipolar similarity score.
    ///
    /// # Arguments
    ///
    /// * `unshifted_query` - The noisy HV after position un-permutation (`dim_u64_words` u64s)
    /// * `codebook_flat` - All 256 byte vectors concatenated (`256 * dim_u64_words` u64s)
    /// * `dim_u64_words` - Number of u64 words per vector
    /// * `total_bits` - Total number of bits (dimensions) per vector
    ///
    /// # Returns
    ///
    /// `(best_byte, bipolar_similarity)` where bipolar_similarity is in [-1.0, 1.0].
    pub fn decode_byte_gpu(
        &self,
        unshifted_query: &[u64],
        codebook_flat: &[u64],
        dim_u64_words: usize,
        total_bits: usize,
    ) -> Result<(u8, f32), StorageError> {
        debug_assert_eq!(unshifted_query.len(), dim_u64_words);
        debug_assert_eq!(codebook_flat.len(), 256 * dim_u64_words);

        let scores = self.batch_similarity(unshifted_query, codebook_flat, 256, dim_u64_words)?;

        // Find the byte with the highest similarity score
        let mut best_byte = 0u8;
        let mut best_score = 0u32;

        for (byte, &score) in scores.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_byte = byte as u8;
            }
        }

        // Convert similarity score to bipolar similarity: 1.0 - 2.0 * hamming / total_bits
        // similarity_score = total_bits - hamming_distance
        // hamming_distance = total_bits - similarity_score
        // bipolar = 1.0 - 2.0 * (total_bits - similarity_score) / total_bits
        //         = 2.0 * similarity_score / total_bits - 1.0
        let bipolar_sim = 2.0 * best_score as f32 / total_bits as f32 - 1.0;

        Ok((best_byte, bipolar_sim))
    }

    /// Upload a codebook to GPU memory for persistent reuse across decode calls.
    ///
    /// The codebook buffer stays in GPU memory until the dispatcher is dropped
    /// or `upload_codebook` is called again with a new codebook.
    ///
    /// # Arguments
    ///
    /// * `codebook_flat` - All codebook vectors concatenated (`num_vectors * dim_u64_words` u64s)
    /// * `num_vectors` - Number of codebook entries (typically 256 for byte codebook)
    /// * `dim_u64_words` - Number of u64 words per vector
    /// * `total_bits` - Total number of bits (dimensions) per vector
    pub fn upload_codebook(
        &self,
        codebook_flat: &[u64],
        num_vectors: usize,
        dim_u64_words: usize,
        total_bits: usize,
    ) {
        debug_assert_eq!(codebook_flat.len(), num_vectors * dim_u64_words);

        let mut backend = match self.backend.lock() {
            Ok(b) => b,
            Err(_) => return,
        };

        // Destroy previous codebook if any
        let mut codebook_guard = self.codebook.lock().unwrap();
        if let Some(prev) = codebook_guard.take() {
            let _ = backend.destroy_buffer(prev.buffer_handle);
        }

        // Upload codebook
        let codebook_bytes = bytemuck::cast_slice::<u64, u8>(codebook_flat);
        let buf = match backend.create_buffer(
            codebook_bytes.len() as u64,
            BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
            Some("hdc_persistent_codebook"),
        ) {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.write_buffer(buf, 0, codebook_bytes).is_err() {
            let _ = backend.destroy_buffer(buf);
            return;
        }

        *codebook_guard = Some(PersistentCodebook {
            buffer_handle: buf,
            num_vectors,
            dim_u64_words,
            total_bits,
        });
    }

    /// Check whether a persistent codebook is uploaded.
    pub fn has_codebook(&self) -> bool {
        self.codebook.lock().unwrap().is_some()
    }

    /// Batch decode: compute similarity of M query vectors against the persistent
    /// codebook in a single GPU dispatch, then find the best-matching byte per query.
    ///
    /// This replaces M separate `decode_byte_gpu` calls (M GPU round-trips) with
    /// ONE multi-query similarity dispatch.
    ///
    /// # Arguments
    ///
    /// * `unshifted_queries` - All query vectors concatenated (`num_positions * dim_u64_words` u64s)
    /// * `num_positions` - Number of byte positions to decode
    ///
    /// # Returns
    ///
    /// `Vec<(u8, f32)>` of length `num_positions` — `(best_byte, bipolar_similarity)` per position.
    pub fn decode_value_batch(
        &self,
        unshifted_queries: &[u64],
        num_positions: usize,
    ) -> Result<Vec<(u8, f32)>, StorageError> {
        let codebook_guard = self.codebook.lock().unwrap();
        let codebook = codebook_guard
            .as_ref()
            .ok_or_else(|| StorageError::Backend("No persistent codebook uploaded".to_string()))?;

        let dim_u64_words = codebook.dim_u64_words;
        let num_vectors = codebook.num_vectors;
        let total_bits = codebook.total_bits;
        let codebook_handle = codebook.buffer_handle;

        debug_assert_eq!(unshifted_queries.len(), num_positions * dim_u64_words);

        let mut backend = self
            .backend
            .lock()
            .map_err(|e| StorageError::Backend(format!("Failed to lock GPU backend: {}", e)))?;

        // Reuse or create persistent query buffer
        let queries_bytes = bytemuck::cast_slice::<u64, u8>(unshifted_queries);
        let required_bytes = queries_bytes.len() as u64;

        let mut ws_guard = self.workspace.lock().unwrap();
        let queries_buf = if let Some(ref ws) = *ws_guard {
            if ws.query_capacity_bytes >= required_bytes {
                // Reuse existing buffer
                backend.write_buffer(ws.query_buffer, 0, queries_bytes)?;
                ws.query_buffer
            } else {
                // Too small — destroy and recreate
                let _ = backend.destroy_buffer(ws.query_buffer);
                let buf = backend.create_buffer(
                    required_bytes,
                    BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                    Some("hdc_decode_queries_ws"),
                )?;
                backend.write_buffer(buf, 0, queries_bytes)?;
                *ws_guard = Some(DecodeWorkspace {
                    query_buffer: buf,
                    query_capacity_bytes: required_bytes,
                });
                buf
            }
        } else {
            // First call — create workspace
            let buf = backend.create_buffer(
                required_bytes,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("hdc_decode_queries_ws"),
            )?;
            backend.write_buffer(buf, 0, queries_bytes)?;
            *ws_guard = Some(DecodeWorkspace {
                query_buffer: buf,
                query_capacity_bytes: required_bytes,
            });
            buf
        };
        drop(ws_guard);

        // Single dispatch: num_positions queries × num_vectors codebook entries
        let result = backend.execute(
            ComputeOp::BinaryHDMultiSimilarity {
                num_queries: num_positions as u32,
                num_vectors: num_vectors as u32,
                num_words: dim_u64_words as u32,
            },
            &[queries_buf, codebook_handle],
        )?;

        // Read back all scores: num_positions × num_vectors u32s
        let total_scores = num_positions * num_vectors;
        let output_bytes =
            backend.read_buffer(result.output_buffer, 0, (total_scores as u64) * 4)?;

        // Find the best byte per position
        let scores: Vec<u32> = output_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut results = Vec::with_capacity(num_positions);
        for pos in 0..num_positions {
            let row = &scores[pos * num_vectors..(pos + 1) * num_vectors];
            let mut best_byte = 0u8;
            let mut best_score = 0u32;
            for (byte, &score) in row.iter().enumerate() {
                if score > best_score {
                    best_score = score;
                    best_byte = byte as u8;
                }
            }
            let bipolar_sim = 2.0 * best_score as f32 / total_bits as f32 - 1.0;
            results.push((best_byte, bipolar_sim));
        }

        // Only destroy output buffer (query buffer is persistent workspace)
        backend.destroy_buffer(result.output_buffer)?;

        Ok(results)
    }
}

impl Drop for GpuDispatcher {
    fn drop(&mut self) {
        if let Ok(mut backend) = self.backend.lock() {
            // Clean up persistent codebook buffer
            if let Ok(mut codebook_guard) = self.codebook.lock() {
                if let Some(codebook) = codebook_guard.take() {
                    let _ = backend.destroy_buffer(codebook.buffer_handle);
                }
            }
            // Clean up workspace buffers
            if let Ok(mut ws_guard) = self.workspace.lock() {
                if let Some(ws) = ws_guard.take() {
                    let _ = backend.destroy_buffer(ws.query_buffer);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_core::persistence::compute::CpuComputeBackend;

    fn make_cpu_dispatcher() -> GpuDispatcher {
        let backend: GpuBackend = Arc::new(Mutex::new(CpuComputeBackend::new()));
        GpuDispatcher::new(backend)
    }

    #[test]
    fn test_config_defaults() {
        let config = GpuHdcConfig::default();
        assert_eq!(config.similarity_threshold, 64);
        assert_eq!(config.bundle_threshold, 32);
        assert_eq!(config.bind_threshold, 128);
    }

    #[test]
    fn test_threshold_checks() {
        let dispatcher = make_cpu_dispatcher();

        assert!(!dispatcher.should_dispatch_similarity(63));
        assert!(dispatcher.should_dispatch_similarity(64));
        assert!(dispatcher.should_dispatch_similarity(1000));

        assert!(!dispatcher.should_dispatch_bundle(31));
        assert!(dispatcher.should_dispatch_bundle(32));

        assert!(!dispatcher.should_dispatch_bind(127));
        assert!(dispatcher.should_dispatch_bind(128));
    }

    #[test]
    fn test_is_gpu_available_with_cpu_backend() {
        let dispatcher = make_cpu_dispatcher();
        assert!(!dispatcher.is_gpu_available());
    }

    #[test]
    fn test_custom_config() {
        let config = GpuHdcConfig {
            similarity_threshold: 16,
            bundle_threshold: 8,
            bind_threshold: 32,
        };
        let backend: GpuBackend = Arc::new(Mutex::new(CpuComputeBackend::new()));
        let dispatcher = GpuDispatcher::with_config(backend, config);

        assert_eq!(dispatcher.config().similarity_threshold, 16);
        assert!(dispatcher.should_dispatch_similarity(16));
        assert!(!dispatcher.should_dispatch_similarity(15));
    }

    #[test]
    fn test_u64_u8_roundtrip() {
        let original: Vec<u64> = vec![0xDEADBEEF_CAFEBABE, 0x12345678_9ABCDEF0];
        let bytes = bytemuck::cast_slice::<u64, u8>(&original);

        // Verify bytemuck preserves the data
        let recovered: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| {
                u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        assert_eq!(original, recovered);
    }

    #[test]
    fn test_batch_similarity_buffer_lifecycle() {
        // CpuComputeBackend doesn't actually compute, but this tests the
        // buffer creation/write/execute/read/destroy lifecycle doesn't panic
        let dispatcher = make_cpu_dispatcher();

        let dim_u64 = 8; // 512 dimensions
        let num_vectors = 4;
        let query = vec![0u64; dim_u64];
        let vectors = vec![0u64; num_vectors * dim_u64];

        let result = dispatcher.batch_similarity(&query, &vectors, num_vectors, dim_u64);
        assert!(result.is_ok());
        let scores = result.unwrap();
        assert_eq!(scores.len(), num_vectors);
    }

    #[test]
    fn test_batch_bundle_buffer_lifecycle() {
        let dispatcher = make_cpu_dispatcher();

        let dim_u64 = 8;
        let num_vectors = 3;
        let vectors = vec![0u64; num_vectors * dim_u64];

        let result = dispatcher.batch_bundle(&vectors, num_vectors, dim_u64);
        assert!(result.is_ok());
        let bundled = result.unwrap();
        assert_eq!(bundled.len(), dim_u64);
    }

    #[test]
    fn test_batch_bind_buffer_lifecycle() {
        let dispatcher = make_cpu_dispatcher();

        let dim_u64 = 8;
        let num_pairs = 2;
        let vectors_a = vec![0u64; num_pairs * dim_u64];
        let vectors_b = vec![0u64; num_pairs * dim_u64];

        let result = dispatcher.batch_bind(&vectors_a, &vectors_b, num_pairs, dim_u64);
        assert!(result.is_ok());
        let bound = result.unwrap();
        assert_eq!(bound.len(), num_pairs * dim_u64);
    }

    #[test]
    fn test_decode_byte_gpu_lifecycle() {
        let dispatcher = make_cpu_dispatcher();

        let dim_u64 = 8;
        let total_bits = 512;
        let query = vec![0u64; dim_u64];
        let codebook = vec![0u64; 256 * dim_u64];

        let result = dispatcher.decode_byte_gpu(&query, &codebook, dim_u64, total_bits);
        assert!(result.is_ok());
        let (byte, sim) = result.unwrap();
        // With CpuComputeBackend (no actual compute), scores are all 0
        assert_eq!(byte, 0);
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_codebook_not_uploaded_initially() {
        let dispatcher = make_cpu_dispatcher();
        assert!(!dispatcher.has_codebook());
    }

    #[test]
    fn test_upload_codebook_lifecycle() {
        let dispatcher = make_cpu_dispatcher();

        let dim_u64 = 8;
        let num_vectors = 256;
        let total_bits = 512;
        let codebook = vec![0u64; num_vectors * dim_u64];

        dispatcher.upload_codebook(&codebook, num_vectors, dim_u64, total_bits);
        assert!(dispatcher.has_codebook());
    }

    #[test]
    fn test_upload_codebook_replaces_previous() {
        let dispatcher = make_cpu_dispatcher();

        let dim_u64 = 8;
        let total_bits = 512;
        let codebook1 = vec![0u64; 256 * dim_u64];
        let codebook2 = vec![1u64; 256 * dim_u64];

        dispatcher.upload_codebook(&codebook1, 256, dim_u64, total_bits);
        assert!(dispatcher.has_codebook());

        // Upload a second codebook — should replace without panic
        dispatcher.upload_codebook(&codebook2, 256, dim_u64, total_bits);
        assert!(dispatcher.has_codebook());
    }

    #[test]
    fn test_decode_value_batch_no_codebook() {
        let dispatcher = make_cpu_dispatcher();

        let dim_u64 = 8;
        let queries = vec![0u64; 4 * dim_u64];

        let result = dispatcher.decode_value_batch(&queries, 4);
        assert!(result.is_err(), "Should fail without uploaded codebook");
    }

    #[test]
    fn test_decode_value_batch_lifecycle() {
        let dispatcher = make_cpu_dispatcher();

        let dim_u64 = 8;
        let total_bits = 512;
        let codebook = vec![0u64; 256 * dim_u64];
        dispatcher.upload_codebook(&codebook, 256, dim_u64, total_bits);

        let num_positions = 4;
        let queries = vec![0u64; num_positions * dim_u64];

        let result = dispatcher.decode_value_batch(&queries, num_positions);
        assert!(result.is_ok());
        let decoded = result.unwrap();
        assert_eq!(decoded.len(), num_positions);
        for &(byte, sim) in &decoded {
            assert!(byte <= 255);
            assert!(sim >= -1.0 && sim <= 1.0);
        }
    }

    // --- Dynamic threshold tests ---

    #[test]
    fn test_dynamic_threshold_lowers_bar() {
        let dispatcher = make_cpu_dispatcher();

        // Static defaults: similarity=64, bundle=32, bind=128
        assert!(!dispatcher.should_dispatch_similarity(32));
        assert!(!dispatcher.should_dispatch_bundle(16));
        assert!(!dispatcher.should_dispatch_bind(64));

        // Controller detects CPU overload, GPU idle → lower thresholds
        dispatcher.update_thresholds(Some(DynamicThresholds {
            similarity: 16,
            bundle: 8,
            bind: 32,
        }));

        // Now smaller batches dispatch to GPU
        assert!(dispatcher.should_dispatch_similarity(16));
        assert!(dispatcher.should_dispatch_similarity(32));
        assert!(dispatcher.should_dispatch_bundle(8));
        assert!(dispatcher.should_dispatch_bundle(16));
        assert!(dispatcher.should_dispatch_bind(32));
        assert!(dispatcher.should_dispatch_bind(64));

        // Still below new thresholds → don't dispatch
        assert!(!dispatcher.should_dispatch_similarity(15));
        assert!(!dispatcher.should_dispatch_bundle(7));
        assert!(!dispatcher.should_dispatch_bind(31));
    }

    #[test]
    fn test_dynamic_threshold_revert() {
        let dispatcher = make_cpu_dispatcher();

        // Lower thresholds
        dispatcher.update_thresholds(Some(DynamicThresholds {
            similarity: 16,
            bundle: 8,
            bind: 32,
        }));
        assert!(dispatcher.should_dispatch_similarity(16));

        // Revert to static defaults
        dispatcher.update_thresholds(None);

        // Static defaults restored: 64/32/128
        assert!(!dispatcher.should_dispatch_similarity(16));
        assert!(!dispatcher.should_dispatch_similarity(63));
        assert!(dispatcher.should_dispatch_similarity(64));

        assert!(!dispatcher.should_dispatch_bundle(31));
        assert!(dispatcher.should_dispatch_bundle(32));

        assert!(!dispatcher.should_dispatch_bind(127));
        assert!(dispatcher.should_dispatch_bind(128));
    }

    #[test]
    fn test_dynamic_threshold_concurrent() {
        use std::thread;

        let backend: GpuBackend = Arc::new(Mutex::new(CpuComputeBackend::new()));
        let dispatcher = Arc::new(GpuDispatcher::new(backend));

        // 8 readers + 1 writer, no data races
        let writer = {
            let d = Arc::clone(&dispatcher);
            thread::spawn(move || {
                for i in 0..1000 {
                    if i % 2 == 0 {
                        d.update_thresholds(Some(DynamicThresholds {
                            similarity: 16,
                            bundle: 8,
                            bind: 32,
                        }));
                    } else {
                        d.update_thresholds(None);
                    }
                }
            })
        };

        let readers: Vec<_> = (0..8)
            .map(|_| {
                let d = Arc::clone(&dispatcher);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        // These must never panic or return inconsistent results
                        let _sim = d.should_dispatch_similarity(32);
                        let _bun = d.should_dispatch_bundle(16);
                        let _bind = d.should_dispatch_bind(64);
                    }
                })
            })
            .collect();

        writer.join().unwrap();
        for r in readers {
            r.join().unwrap();
        }

        // After all threads complete, verify we can still update cleanly
        dispatcher.update_thresholds(Some(DynamicThresholds {
            similarity: 100,
            bundle: 50,
            bind: 200,
        }));
        assert!(dispatcher.should_dispatch_similarity(100));
        assert!(!dispatcher.should_dispatch_similarity(99));
    }
}
