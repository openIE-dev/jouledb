//! Platform-agnostic compute backend traits
//!
//! Provides abstractions for computation across different hardware:
//! - CPU: Standard Rust computation
//! - GPU: WebGPU, CUDA, Metal, Vulkan
//! - NPU/TPU: Neural processing units
//! - SIMD: Vectorized CPU operations for MCUs
//!
//! ## Design Philosophy
//!
//! JouleDB is designed from the ground up to leverage hardware acceleration.
//! Database operations that benefit from parallelism are expressed as compute
//! kernels that can be executed on the most appropriate hardware.
//!
//! ## Supported Operations
//!
//! - Vector similarity search (HNSW, IVF)
//! - Batch key lookups
//! - Index building/rebuilding
//! - Aggregations (COUNT, SUM, AVG, etc.)
//! - Sorting and filtering
//! - Hash computations

use crate::error::StorageError;

/// Compute device type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU (default fallback)
    Cpu,
    /// GPU via WebGPU/wgpu (cross-platform)
    Gpu,
    /// NVIDIA GPU via CUDA
    Cuda,
    /// Apple GPU via Metal
    Metal,
    /// Vulkan compute
    Vulkan,
    /// Neural Processing Unit
    Npu,
    /// Tensor Processing Unit
    Tpu,
    /// Language Processing Unit (Groq/xAI inference chips)
    Lpu,
    /// SIMD-optimized CPU (for MCUs)
    Simd,
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
    }
}

/// Device capabilities
#[derive(Debug, Clone, Default)]
pub struct DeviceCapabilities {
    /// Device type
    pub device_type: DeviceType,
    /// Device name
    pub name: String,
    /// Total memory in bytes (0 if unknown)
    pub total_memory: u64,
    /// Available memory in bytes (0 if unknown)
    pub available_memory: u64,
    /// Maximum work group size
    pub max_workgroup_size: u32,
    /// Maximum buffer size
    pub max_buffer_size: u64,
    /// Supports f16 operations
    pub supports_f16: bool,
    /// Supports f64 operations
    pub supports_f64: bool,
    /// Supports atomic operations
    pub supports_atomics: bool,
    /// Number of compute units (cores/SMs)
    pub compute_units: u32,
}

/// Buffer usage flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage(u32);

impl BufferUsage {
    /// Buffer can be read from shader
    pub const STORAGE_READ: Self = Self(0x01);
    /// Buffer can be written to from shader
    pub const STORAGE_WRITE: Self = Self(0x02);
    /// Buffer can be used as uniform
    pub const UNIFORM: Self = Self(0x04);
    /// Buffer can be copied from
    pub const COPY_SRC: Self = Self(0x08);
    /// Buffer can be copied to
    pub const COPY_DST: Self = Self(0x10);
    /// Buffer is mapped for CPU access
    pub const MAP_READ: Self = Self(0x20);
    /// Buffer is mapped for CPU write
    pub const MAP_WRITE: Self = Self(0x40);

    /// Create from bits
    pub fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Get bits
    pub fn bits(self) -> u32 {
        self.0
    }

    /// Combine usage flags
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Compute buffer handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u64);

/// Compute pipeline handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineHandle(pub u64);

/// Bind group handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupHandle(pub u64);

/// Compute operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeOp {
    /// Vector similarity search
    VectorSearch {
        /// Number of vectors to search
        num_queries: u32,
        /// Vector dimension
        dimension: u32,
        /// Top-K results per query
        top_k: u32,
    },
    /// Batch key lookup
    BatchLookup {
        /// Number of keys
        num_keys: u32,
    },
    /// Build index
    BuildIndex {
        /// Number of items
        num_items: u32,
        /// Item size in bytes
        item_size: u32,
    },
    /// Aggregation
    Aggregate {
        /// Aggregation type
        agg_type: AggregationType,
        /// Number of items
        num_items: u32,
    },
    /// Sort
    Sort {
        /// Number of items
        num_items: u32,
        /// Item size in bytes
        item_size: u32,
        /// Sort descending
        descending: bool,
    },
    /// Filter
    Filter {
        /// Number of items to filter
        num_items: u32,
    },
    /// Hash computation
    Hash {
        /// Number of items to hash
        num_items: u32,
        /// Hash algorithm
        algorithm: HashAlgorithm,
    },
    /// B-tree range scan (GPU-accelerated)
    BTreeRangeScan {
        /// Number of B-tree nodes to scan
        num_nodes: u32,
        /// Keys per node
        keys_per_node: u32,
        /// Start key length (0 = unbounded)
        start_key_len: u32,
        /// End key length (0 = unbounded)
        end_key_len: u32,
        /// Include start bound
        include_start: bool,
        /// Include end bound
        include_end: bool,
    },
    /// Binary hyperdimensional bind operation (XOR, single pair)
    BinaryHDBind {
        /// Number of u64 words per vector
        num_words: u32,
    },
    /// Binary hyperdimensional batch bind (XOR multiple pairs in one dispatch)
    BinaryHDBatchBind {
        /// Number of vector pairs to bind
        num_pairs: u32,
        /// Number of u64 words per vector
        num_words: u32,
    },
    /// Binary hyperdimensional bundle operation (majority voting)
    BinaryHDBundle {
        /// Number of vectors to bundle
        num_vectors: u32,
        /// Number of u64 words per vector
        num_words: u32,
    },
    /// Binary hyperdimensional similarity search (Hamming distance)
    BinaryHDSimilarity {
        /// Number of vectors in database
        num_vectors: u32,
        /// Number of u64 words per vector
        num_words: u32,
    },
    /// Binary hyperdimensional multi-query similarity (M queries × N vectors)
    BinaryHDMultiSimilarity {
        /// Number of query vectors
        num_queries: u32,
        /// Number of database vectors
        num_vectors: u32,
        /// Number of u64 words per vector
        num_words: u32,
    },
    /// Custom compute shader
    Custom {
        /// Shader name/identifier
        name: &'static str,
        /// Workgroup size
        workgroup_size: [u32; 3],
    },
}

/// Aggregation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationType {
    /// Count items
    Count,
    /// Sum values
    Sum,
    /// Average value
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Standard deviation
    StdDev,
    /// Variance
    Variance,
}

/// Hash algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// XXHash64
    XxHash64,
    /// CRC32
    Crc32,
    /// MurmurHash3
    Murmur3,
    /// SipHash
    SipHash,
    /// SHA256 (for cryptographic purposes)
    Sha256,
}

/// Compute result
#[derive(Debug)]
pub struct ComputeResult {
    /// Output buffer handle
    pub output_buffer: BufferHandle,
    /// Number of results
    pub result_count: u32,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// Compute backend trait
///
/// Platform-specific implementations provide actual compute capabilities.
pub trait ComputeBackend: Send + Sync {
    /// Get device capabilities
    fn capabilities(&self) -> &DeviceCapabilities;

    /// Create a buffer
    fn create_buffer(
        &mut self,
        size: u64,
        usage: BufferUsage,
        label: Option<&str>,
    ) -> Result<BufferHandle, StorageError>;

    /// Destroy a buffer
    fn destroy_buffer(&mut self, buffer: BufferHandle) -> Result<(), StorageError>;

    /// Write data to a buffer
    fn write_buffer(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<(), StorageError>;

    /// Read data from a buffer
    fn read_buffer(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, StorageError>;

    /// Copy between buffers
    fn copy_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), StorageError>;

    /// Execute a compute operation
    fn execute(
        &mut self,
        op: ComputeOp,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError>;

    /// Wait for all operations to complete
    fn synchronize(&mut self) -> Result<(), StorageError>;

    /// Check if a compute operation is supported
    fn supports_operation(&self, op: &ComputeOp) -> bool;
}

/// CPU compute backend (fallback implementation)
pub struct CpuComputeBackend {
    capabilities: DeviceCapabilities,
    buffers: std::collections::HashMap<u64, Vec<u8>>,
    next_buffer_id: u64,
}

impl CpuComputeBackend {
    /// Create a new CPU compute backend
    pub fn new() -> Self {
        Self {
            capabilities: DeviceCapabilities {
                device_type: DeviceType::Cpu,
                name: "CPU".to_string(),
                total_memory: 0, // Unknown
                available_memory: 0,
                max_workgroup_size: 1,
                max_buffer_size: u64::MAX,
                supports_f16: false,
                supports_f64: true,
                supports_atomics: true,
                compute_units: num_cpus(),
            },
            buffers: std::collections::HashMap::new(),
            next_buffer_id: 1,
        }
    }
}

impl Default for CpuComputeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuComputeBackend {
    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn create_buffer(
        &mut self,
        size: u64,
        _usage: BufferUsage,
        _label: Option<&str>,
    ) -> Result<BufferHandle, StorageError> {
        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(id, vec![0u8; size as usize]);
        Ok(BufferHandle(id))
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) -> Result<(), StorageError> {
        self.buffers.remove(&buffer.0);
        Ok(())
    }

    fn write_buffer(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<(), StorageError> {
        let buf = self
            .buffers
            .get_mut(&buffer.0)
            .ok_or_else(|| StorageError::Backend("Buffer not found".to_string()))?;

        let start = offset as usize;
        let end = start + data.len();

        if end > buf.len() {
            return Err(StorageError::Backend("Write out of bounds".to_string()));
        }

        buf[start..end].copy_from_slice(data);
        Ok(())
    }

    fn read_buffer(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, StorageError> {
        let buf = self
            .buffers
            .get(&buffer.0)
            .ok_or_else(|| StorageError::Backend("Buffer not found".to_string()))?;

        let start = offset as usize;
        let end = start + size as usize;

        if end > buf.len() {
            return Err(StorageError::Backend("Read out of bounds".to_string()));
        }

        Ok(buf[start..end].to_vec())
    }

    fn copy_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), StorageError> {
        // Read from source
        let data = self.read_buffer(src, src_offset, size)?;
        // Write to destination
        self.write_buffer(dst, dst_offset, &data)
    }

    fn execute(
        &mut self,
        op: ComputeOp,
        _inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = std::time::Instant::now();

        // Create output buffer based on operation
        let (output_size, result_count) = match op {
            ComputeOp::VectorSearch {
                num_queries, top_k, ..
            } => {
                // Output: indices (u32) and distances (f32) for each query
                let size = (num_queries * top_k * 8) as u64;
                (size, num_queries * top_k)
            }
            ComputeOp::BatchLookup { num_keys } => {
                // Output: found flags (u8) for each key
                (num_keys as u64, num_keys)
            }
            ComputeOp::Aggregate { .. } => {
                // Output: single aggregated value (f64)
                (8, 1)
            }
            ComputeOp::Sort {
                num_items,
                item_size: _,
                ..
            } => {
                // Output: sorted indices (u32)
                ((num_items * 4) as u64, num_items)
            }
            ComputeOp::Filter { num_items } => {
                // Output: filter mask (u8)
                (num_items as u64, num_items)
            }
            ComputeOp::BTreeRangeScan {
                num_nodes,
                keys_per_node,
                start_key_len: _,
                end_key_len: _,
                ..
            } => {
                // Output: result keys and values (offsets as u32)
                // Estimate max results (all keys in all nodes)
                let max_results = (num_nodes * keys_per_node) as u64;
                let size = (max_results * 8) as u64; // 4 bytes key offset + 4 bytes value offset per result
                (size, max_results as u32)
            }
            ComputeOp::Hash {
                num_items,
                algorithm,
            } => {
                let hash_size = match algorithm {
                    HashAlgorithm::XxHash64 | HashAlgorithm::SipHash => 8,
                    HashAlgorithm::Crc32 | HashAlgorithm::Murmur3 => 4,
                    HashAlgorithm::Sha256 => 32,
                };
                ((num_items * hash_size) as u64, num_items)
            }
            ComputeOp::BuildIndex { num_items, .. } => {
                // Output: index structure (simplified)
                ((num_items * 8) as u64, num_items)
            }
            ComputeOp::BinaryHDBind { num_words } => {
                // Output: bound vector (u64 per word)
                let size = (num_words * 8) as u64;
                (size, num_words)
            }
            ComputeOp::BinaryHDBatchBind {
                num_pairs,
                num_words,
            } => {
                // Output: bound vectors (num_pairs * num_words u64s)
                let size = (num_pairs as u64) * (num_words as u64) * 8;
                (size, num_pairs * num_words)
            }
            ComputeOp::BinaryHDBundle {
                num_vectors: _,
                num_words,
            } => {
                // Output: bundled vector (u64 per word)
                let size = (num_words * 8) as u64;
                (size, num_words)
            }
            ComputeOp::BinaryHDSimilarity { num_vectors, .. } => {
                // Output: similarity scores (u32 per vector)
                let size = (num_vectors * 4) as u64;
                (size, num_vectors)
            }
            ComputeOp::BinaryHDMultiSimilarity {
                num_queries,
                num_vectors,
                ..
            } => {
                // Output: similarity scores (u32 per query-vector pair)
                let total = num_queries * num_vectors;
                let size = (total * 4) as u64;
                (size, total)
            }
            ComputeOp::Custom { .. } => {
                // Custom ops need to specify their output size
                (1024, 1)
            }
        };

        let output_buffer = self.create_buffer(output_size, BufferUsage::STORAGE_WRITE, None)?;

        // CPU implementation would go here - for now just return empty results
        // Real implementations would process the input buffers

        let execution_time_us = start.elapsed().as_micros() as u64;

        Ok(ComputeResult {
            output_buffer,
            result_count,
            execution_time_us,
        })
    }

    fn synchronize(&mut self) -> Result<(), StorageError> {
        // CPU is always synchronous
        Ok(())
    }

    fn supports_operation(&self, _op: &ComputeOp) -> bool {
        // CPU supports all operations (as fallback)
        true
    }
}

/// Get number of CPU cores
fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1)
}

/// Compute context for managing compute operations
pub struct ComputeContext<B: ComputeBackend> {
    backend: B,
    /// Statistics
    total_operations: u64,
    total_time_us: u64,
}

impl<B: ComputeBackend> ComputeContext<B> {
    /// Create a new compute context
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            total_operations: 0,
            total_time_us: 0,
        }
    }

    /// Get backend capabilities
    pub fn capabilities(&self) -> &DeviceCapabilities {
        self.backend.capabilities()
    }

    /// Execute a compute operation
    pub fn execute(
        &mut self,
        op: ComputeOp,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let result = self.backend.execute(op, inputs)?;
        self.total_operations += 1;
        self.total_time_us += result.execution_time_us;
        Ok(result)
    }

    /// Create a buffer
    pub fn create_buffer(
        &mut self,
        size: u64,
        usage: BufferUsage,
    ) -> Result<BufferHandle, StorageError> {
        self.backend.create_buffer(size, usage, None)
    }

    /// Write to a buffer
    pub fn write_buffer(&mut self, buffer: BufferHandle, data: &[u8]) -> Result<(), StorageError> {
        self.backend.write_buffer(buffer, 0, data)
    }

    /// Read from a buffer
    pub fn read_buffer(
        &mut self,
        buffer: BufferHandle,
        size: u64,
    ) -> Result<Vec<u8>, StorageError> {
        self.backend.read_buffer(buffer, 0, size)
    }

    /// Destroy a buffer
    pub fn destroy_buffer(&mut self, buffer: BufferHandle) -> Result<(), StorageError> {
        self.backend.destroy_buffer(buffer)
    }

    /// Synchronize with device
    pub fn synchronize(&mut self) -> Result<(), StorageError> {
        self.backend.synchronize()
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, u64) {
        (self.total_operations, self.total_time_us)
    }

    /// Get mutable access to backend
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_create_buffer() {
        let mut backend = CpuComputeBackend::new();
        let buffer = backend
            .create_buffer(1024, BufferUsage::STORAGE_WRITE, Some("test"))
            .unwrap();
        assert_eq!(buffer.0, 1);
    }

    #[test]
    fn test_cpu_backend_write_read() {
        let mut backend = CpuComputeBackend::new();
        let buffer = backend
            .create_buffer(1024, BufferUsage::STORAGE_WRITE, None)
            .unwrap();

        let data = b"hello world";
        backend.write_buffer(buffer, 0, data).unwrap();

        let read = backend.read_buffer(buffer, 0, data.len() as u64).unwrap();
        assert_eq!(read, data);
    }

    #[test]
    fn test_cpu_backend_copy() {
        let mut backend = CpuComputeBackend::new();
        let src = backend
            .create_buffer(1024, BufferUsage::COPY_SRC, None)
            .unwrap();
        let dst = backend
            .create_buffer(1024, BufferUsage::COPY_DST, None)
            .unwrap();

        let data = b"copy test";
        backend.write_buffer(src, 0, data).unwrap();
        backend
            .copy_buffer(src, 0, dst, 0, data.len() as u64)
            .unwrap();

        let read = backend.read_buffer(dst, 0, data.len() as u64).unwrap();
        assert_eq!(read, data);
    }

    #[test]
    fn test_cpu_backend_execute() {
        let mut backend = CpuComputeBackend::new();

        let result = backend
            .execute(
                ComputeOp::Aggregate {
                    agg_type: AggregationType::Count,
                    num_items: 100,
                },
                &[],
            )
            .unwrap();

        assert_eq!(result.result_count, 1);
    }

    #[test]
    fn test_compute_context() {
        let backend = CpuComputeBackend::new();
        let mut ctx = ComputeContext::new(backend);

        assert_eq!(ctx.capabilities().device_type, DeviceType::Cpu);

        let buffer = ctx.create_buffer(256, BufferUsage::STORAGE_WRITE).unwrap();
        ctx.write_buffer(buffer, b"test").unwrap();

        let data = ctx.read_buffer(buffer, 4).unwrap();
        assert_eq!(data, b"test");
    }

    #[test]
    fn test_buffer_usage() {
        let usage = BufferUsage::STORAGE_READ.union(BufferUsage::STORAGE_WRITE);
        assert_eq!(usage.bits(), 0x03);
    }

    #[test]
    fn test_device_capabilities() {
        let caps = DeviceCapabilities::default();
        assert_eq!(caps.device_type, DeviceType::Cpu);
    }
}
