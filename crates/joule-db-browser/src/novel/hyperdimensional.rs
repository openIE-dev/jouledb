//! Browser Integration for Hyperdimensional Computing
//!
//! Provides GPU-accelerated hyperdimensional operations and WASM bindings.
//! Core algorithms remain in joule-db-novel (pure Rust).

use joule_db_core::persistence::compute::{BufferHandle, BufferUsage, ComputeBackend};
#[cfg(feature = "novel")]
use joule_db_hdc::hyperdimensional::{HyperVector, HyperdimensionalStorage};
use std::sync::Arc;

use joule_db_gpu::WgpuComputeBackend;

/// GPU-accelerated hyperdimensional operations
pub struct GpuHyperdimensionalOps {
    /// GPU compute backend (optional)
    gpu_backend: Option<Arc<WgpuComputeBackend>>,
    /// Core storage (pure Rust, no WASM)
    storage: HyperdimensionalStorage,
}

impl GpuHyperdimensionalOps {
    /// Create with GPU acceleration
    pub async fn new_with_gpu(dimension: usize) -> Result<Self, String> {
        let gpu_backend = WgpuComputeBackend::new()
            .await
            .map_err(|e| format!("GPU init failed: {}", e))?;

        Ok(Self {
            gpu_backend: Some(Arc::new(gpu_backend)),
            storage: HyperdimensionalStorage::new(dimension),
        })
    }

    /// Create without GPU (CPU fallback)
    pub fn new(dimension: usize) -> Self {
        Self {
            gpu_backend: None,
            storage: HyperdimensionalStorage::new(dimension),
        }
    }

    /// Bind two vectors (GPU-accelerated if available)
    pub async fn bind(&self, a: &HyperVector, b: &HyperVector) -> Result<HyperVector, String> {
        if let Some(ref gpu) = self.gpu_backend {
            return self.bind_gpu(gpu, a, b).await;
        }

        // CPU fallback
        a.bind(b).map_err(|e| e.to_string())
    }

    /// GPU-accelerated bind operation
    async fn bind_gpu(
        &self,
        _gpu: &WgpuComputeBackend,
        a: &HyperVector,
        b: &HyperVector,
    ) -> Result<HyperVector, String> {
        // For now, use CPU fallback since HyperVector uses f32, not binary
        // Binary hyperdimensional operations would use BinaryHV from turbo_holographic
        a.bind(b).map_err(|e| e.to_string())
    }

    /// Bundle multiple vectors (GPU-accelerated if available)
    pub async fn bundle(&self, vectors: &[HyperVector]) -> Result<Option<HyperVector>, String> {
        if let Some(ref gpu) = self.gpu_backend {
            return self.bundle_gpu(gpu, vectors).await;
        }

        // CPU fallback - bundle returns Result, so convert to Option on success
        match HyperVector::bundle(vectors) {
            Ok(v) => Ok(Some(v)),
            Err(_) => Ok(None),
        }
    }

    /// GPU-accelerated bundle operation
    async fn bundle_gpu(
        &self,
        _gpu: &WgpuComputeBackend,
        vectors: &[HyperVector],
    ) -> Result<Option<HyperVector>, String> {
        // For now, use CPU fallback since HyperVector uses f32, not binary
        // Binary hyperdimensional operations would use BinaryHV from turbo_holographic
        match HyperVector::bundle(vectors) {
            Ok(v) => Ok(Some(v)),
            Err(_) => Ok(None),
        }
    }

    /// Add vector to storage
    pub fn add_vector(
        &self,
        vector: HyperVector,
    ) -> Result<usize, joule_db_hdc::hyperdimensional::HDError> {
        self.storage.add_vector(vector)
    }

    /// Get number of stored vectors
    pub fn vector_count(&self) -> usize {
        self.storage.len()
    }

    /// Get a vector by index
    pub fn get_vector(&self, index: usize) -> Option<HyperVector> {
        self.storage.get(index)
    }

    /// Similarity search (GPU-accelerated if available)
    pub async fn similarity_search(
        &self,
        query: &HyperVector,
        k: usize,
    ) -> Result<Vec<(HyperVector, f32)>, String> {
        if let Some(ref gpu) = self.gpu_backend {
            return self.similarity_search_gpu(gpu, query, k).await;
        }

        // CPU fallback - SimilarityMatch has index and similarity, need to get vectors by index
        self.storage
            .similarity_search(query, k)
            .map_err(|e| e.to_string())
            .map(|results| {
                results
                    .into_iter()
                    .filter_map(|m| self.storage.get(m.index).map(|v| (v, m.similarity)))
                    .collect()
            })
    }

    /// GPU-accelerated similarity search
    async fn similarity_search_gpu(
        &self,
        _gpu: &WgpuComputeBackend,
        query: &HyperVector,
        k: usize,
    ) -> Result<Vec<(HyperVector, f32)>, String> {
        // For now, use CPU fallback since HyperVector uses f32, not binary
        // Binary hyperdimensional operations would use BinaryHV from turbo_holographic
        self.storage
            .similarity_search(query, k)
            .map_err(|e| e.to_string())
            .map(|results| {
                results
                    .into_iter()
                    .filter_map(|m| self.storage.get(m.index).map(|v| (v, m.similarity)))
                    .collect()
            })
    }
}

/// Browser-specific hyperdimensional search wrapper
pub struct BrowserHyperdimensionalSearch {
    ops: GpuHyperdimensionalOps,
}

impl BrowserHyperdimensionalSearch {
    /// Create new search instance
    pub fn new(dimension: usize) -> Self {
        Self {
            ops: GpuHyperdimensionalOps::new(dimension),
        }
    }

    /// Create with GPU acceleration
    pub async fn new_with_gpu(dimension: usize) -> Result<Self, String> {
        Ok(Self {
            ops: GpuHyperdimensionalOps::new_with_gpu(dimension).await?,
        })
    }

    /// Add vector to storage
    pub fn add_vector(&mut self, vector: HyperVector) -> Result<usize, String> {
        // Add vector to the internal storage
        self.ops
            .add_vector(vector)
            .map_err(|e| format!("Failed to add vector: {}", e))
    }

    /// Get number of stored vectors
    pub fn vector_count(&self) -> usize {
        self.ops.vector_count()
    }

    /// Get a vector by index
    pub fn get_vector(&self, index: usize) -> Option<HyperVector> {
        self.ops.get_vector(index)
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        query: &HyperVector,
        k: usize,
    ) -> Result<Vec<(HyperVector, f32)>, String> {
        self.ops.similarity_search(query, k).await
    }
}
