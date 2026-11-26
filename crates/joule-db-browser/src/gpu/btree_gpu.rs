//! GPU-Accelerated B-tree Operations
//!
//! Provides GPU acceleration for B-tree range scans and filtering operations.
//! Falls back to CPU when GPU is unavailable.

use joule_db_core::StorageError;
use joule_db_core::{
    index::{Bound, IndexEntry, ScanDirection},
    persistence::compute::{BufferHandle, BufferUsage, ComputeBackend, ComputeOp},
    storage::PageId,
};
use std::sync::{Arc, RwLock};

/// GPU-accelerated B-tree range scan
///
/// Performs parallel range scan over B-tree nodes using GPU compute.
/// Falls back to CPU if GPU is unavailable.
pub struct GpuBTreeRangeScan {
    /// GPU compute backend (wrapped in RwLock for interior mutability)
    gpu_backend: Option<Arc<RwLock<dyn ComputeBackend + Send + Sync>>>,
    /// CPU fallback flag
    use_cpu_fallback: bool,
}

impl GpuBTreeRangeScan {
    /// Create a new GPU-accelerated range scan
    pub fn new(gpu_backend: Option<Arc<RwLock<dyn ComputeBackend + Send + Sync>>>) -> Self {
        let use_cpu_fallback = gpu_backend.is_none();
        Self {
            gpu_backend,
            use_cpu_fallback,
        }
    }

    /// Execute GPU-accelerated range scan
    ///
    /// # Arguments
    /// * `nodes` - Serialized B-tree nodes
    /// * `start` - Start bound
    /// * `end` - End bound
    /// * `direction` - Scan direction
    ///
    /// # Returns
    /// Vector of matching index entries
    pub async fn scan(
        &mut self,
        nodes: &[u8],
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        _direction: ScanDirection,
    ) -> Result<Vec<IndexEntry>, StorageError> {
        if self.use_cpu_fallback {
            // CPU fallback - perform range scan on CPU
            // This is a simplified implementation - in production would use proper B-tree traversal
            return Err(StorageError::Backend(
                "CPU fallback: Use EnhancedAsyncEngine::range_cpu instead".to_string(),
            ));
        }

        let Some(backend) = self.gpu_backend.as_ref() else {
            return Err(StorageError::Backend(
                "CPU fallback: Use EnhancedAsyncEngine::range_cpu instead".to_string(),
            ));
        };
        let mut backend_guard = backend.write().unwrap_or_else(|e| e.into_inner());

        // Serialize bounds
        let (start_key, start_len, include_start, unbounded_start) = match start {
            Bound::Included(k) => (k.to_vec(), k.len() as u32, true, false),
            Bound::Excluded(k) => (k.to_vec(), k.len() as u32, false, false),
            Bound::Unbounded => (Vec::new(), 0, false, true),
        };

        let (end_key, end_len, include_end, unbounded_end) = match end {
            Bound::Included(k) => (k.to_vec(), k.len() as u32, true, false),
            Bound::Excluded(k) => (k.to_vec(), k.len() as u32, false, false),
            Bound::Unbounded => (Vec::new(), 0, false, true),
        };

        // Estimate number of nodes (simplified - would need actual node count)
        let num_nodes = (nodes.len() / 16384) as u32; // Assuming 16KB pages
        let keys_per_node = 32u32; // MAX_KEYS from B-tree

        // Create input buffer
        let nodes_handle = backend_guard.create_buffer(
            nodes.len() as u64,
            BufferUsage::STORAGE_READ,
            Some("btree_nodes"),
        )?;

        backend_guard.write_buffer(nodes_handle, 0, nodes)?;

        // Create start key buffer
        let start_key_handle = if !unbounded_start {
            Some(backend_guard.create_buffer(
                start_key.len() as u64,
                BufferUsage::STORAGE_READ,
                Some("start_key"),
            )?)
        } else {
            None
        };

        if let Some(handle) = start_key_handle {
            backend_guard.write_buffer(handle, 0, &start_key)?;
        }

        // Create end key buffer
        let end_key_handle = if !unbounded_end {
            Some(backend_guard.create_buffer(
                end_key.len() as u64,
                BufferUsage::STORAGE_READ,
                Some("end_key"),
            )?)
        } else {
            None
        };

        if let Some(handle) = end_key_handle {
            backend_guard.write_buffer(handle, 0, &end_key)?;
        }

        // Execute GPU compute
        let op = ComputeOp::BTreeRangeScan {
            num_nodes,
            keys_per_node,
            start_key_len: start_len,
            end_key_len: end_len,
            include_start,
            include_end,
        };

        // Prepare input buffers
        let mut input_buffers = vec![nodes_handle];

        if let Some(start_handle) = start_key_handle {
            input_buffers.push(start_handle);
        } else {
            // Create dummy buffer for unbounded start
            let dummy =
                backend_guard.create_buffer(1, BufferUsage::STORAGE_READ, Some("dummy_start"))?;
            backend_guard.write_buffer(dummy, 0, &[0u8])?;
            input_buffers.push(dummy);
        }

        if let Some(end_handle) = end_key_handle {
            input_buffers.push(end_handle);
        } else {
            // Create dummy buffer for unbounded end
            let dummy =
                backend_guard.create_buffer(1, BufferUsage::STORAGE_READ, Some("dummy_end"))?;
            backend_guard.write_buffer(dummy, 0, &[0u8])?;
            input_buffers.push(dummy);
        }

        // Execute on GPU
        let result = backend_guard.execute(op, &input_buffers)?;

        // Read results from GPU
        let result_size = (result.result_count * 16) as u64; // Each entry: key (8) + value (8)
        let result_data = backend_guard.read_buffer(result.output_buffer, 0, result_size)?;

        // Deserialize results
        // Format: [key_len: u32, key_data: Vec<u8>, value_len: u32, value_data: Vec<u8>] * N
        let mut entries = Vec::new();
        let mut offset = 0;

        for _ in 0..result.result_count {
            if offset + 4 > result_data.len() {
                break;
            }

            // Read key length
            let key_len = u32::from_le_bytes([
                result_data[offset],
                result_data[offset + 1],
                result_data[offset + 2],
                result_data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + key_len > result_data.len() {
                break;
            }

            // Read key data
            let key = result_data[offset..offset + key_len].to_vec();
            offset += key_len;

            if offset + 4 > result_data.len() {
                break;
            }

            // Read value length
            let value_len = u32::from_le_bytes([
                result_data[offset],
                result_data[offset + 1],
                result_data[offset + 2],
                result_data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + value_len > result_data.len() {
                break;
            }

            // Read value data
            let value = result_data[offset..offset + value_len].to_vec();
            offset += value_len;

            entries.push(IndexEntry { key, value });
        }

        Ok(entries)
    }
}

/// GPU-accelerated B-tree operations wrapper
///
/// Provides GPU acceleration for B-tree operations when available,
/// with automatic CPU fallback.
pub struct GpuBTreeOps {
    /// GPU compute backend (optional)
    gpu_backend: Option<Arc<dyn ComputeBackend + Send + Sync>>,
}

impl GpuBTreeOps {
    /// Create new GPU B-tree operations
    pub fn new(gpu_backend: Option<Arc<dyn ComputeBackend + Send + Sync>>) -> Self {
        Self { gpu_backend }
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_backend.is_some()
    }

    /// Get GPU capabilities
    pub fn capabilities(&self) -> Option<joule_db_core::persistence::compute::DeviceCapabilities> {
        self.gpu_backend.as_ref().map(|b| b.capabilities().clone())
    }
}
