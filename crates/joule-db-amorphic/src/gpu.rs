//! GPU-accelerated operations for Amorphic Database
//!
//! Provides massive parallelism for similarity search using WebGPU.
//!
//! ## Performance Analysis
//!
//! ### When to Use GPU
//!
//! The GPU performs **exhaustive brute-force search** in O(n) time, computing
//! similarity against every vector in parallel. This is ideal for:
//!
//! - **Exact k-NN search**: When approximate LSH results aren't acceptable
//! - **Batch queries**: Many queries amortize the GPU dispatch overhead
//! - **Real-time streaming**: When vectors change too fast for index updates
//!
//! ### When CPU is Faster
//!
//! The CPU uses **LSH indexing** for ~O(1) approximate nearest neighbor lookup,
//! which beats brute-force GPU for most single-query workloads:
//!
//! | Dataset Size | CPU (LSH) | GPU (brute) |
//! |--------------|-----------|-------------|
//! | 10K vectors  | ~765µs    | ~1.4ms      |
//! | 100K vectors | ~570µs    | ~2.1ms      |
//!
//! CPU scales sub-linearly due to LSH; GPU scales linearly with vector count.
//!
//! ### Optimization: Persistent GPU Store
//!
//! Use `GpuVectorStore` to pre-upload vectors (one-time cost), then
//! `compute_similarities_fast()` avoids re-upload overhead (3x faster).

use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::{AmorphicError, AmorphicResult, DIMENSION};
use joule_db_hdc::BinaryHV;

/// GPU context for accelerated operations
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    similarity_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    batch_pipeline: wgpu::ComputePipeline,
    batch_bind_group_layout: wgpu::BindGroupLayout,
}

/// Pre-uploaded vector store on GPU for fast repeated queries
///
/// Avoids re-uploading vectors each query - vectors persist on GPU memory.
pub struct GpuVectorStore {
    vectors_buffer: wgpu::Buffer,
    num_vectors: usize,
    num_words: usize,
}

impl GpuVectorStore {
    /// Upload vectors to GPU memory (one-time cost)
    pub fn upload(gpu: &GpuContext, vectors: &[BinaryHV]) -> AmorphicResult<Self> {
        if vectors.is_empty() {
            return Err(AmorphicError::QueryError(
                "Cannot upload empty vectors".to_string(),
            ));
        }

        let num_vectors = vectors.len();
        let num_words = (DIMENSION + 63) / 64;

        // Prepare vectors data (all vectors concatenated)
        let mut vectors_data: Vec<u32> = Vec::with_capacity(num_vectors * num_words * 2);
        for hv in vectors {
            for &word in hv.as_words() {
                vectors_data.push(word as u32);
                vectors_data.push((word >> 32) as u32);
            }
        }

        let vectors_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Persistent Vectors Buffer"),
                contents: bytemuck::cast_slice(&vectors_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        Ok(Self {
            vectors_buffer,
            num_vectors,
            num_words,
        })
    }

    /// Number of vectors stored on GPU
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }
}

/// Parameters for similarity computation
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimilarityParams {
    num_vectors: u32,
    num_words: u32,
    _padding: [u32; 2],
}

/// Parameters for batched similarity computation (2D dispatch)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchSimilarityParams {
    num_vectors: u32,
    num_words: u32,
    num_queries: u32,
    _padding: u32,
}

/// WGSL shader for parallel similarity computation
const SIMILARITY_SHADER: &str = r#"
struct SimilarityParams {
    num_vectors: u32,
    num_words: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> query: array<u32>;
@group(0) @binding(1) var<storage, read> vectors: array<u32>;
@group(0) @binding(2) var<storage, read_write> similarities: array<u32>;
@group(0) @binding(3) var<uniform> params: SimilarityParams;

// Optimized popcount using parallel bit counting
fn popcount(x: u32) -> u32 {
    var v = x;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    v = v + (v >> 8u);
    v = v + (v >> 16u);
    return v & 0x3Fu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vec_idx = global_id.x;
    if (vec_idx >= params.num_vectors) {
        return;
    }

    // Compute Hamming distance between query and this vector
    var distance: u32 = 0u;
    let base_offset = vec_idx * params.num_words;

    // Process 4 words at a time for better memory coalescing
    let num_chunks = params.num_words / 4u;
    for (var chunk = 0u; chunk < num_chunks; chunk++) {
        let word_base = chunk * 4u;
        let d0 = popcount(query[word_base] ^ vectors[base_offset + word_base]);
        let d1 = popcount(query[word_base + 1u] ^ vectors[base_offset + word_base + 1u]);
        let d2 = popcount(query[word_base + 2u] ^ vectors[base_offset + word_base + 2u]);
        let d3 = popcount(query[word_base + 3u] ^ vectors[base_offset + word_base + 3u]);
        distance += d0 + d1 + d2 + d3;
    }

    // Handle remainder
    let remainder_start = num_chunks * 4u;
    for (var word_idx = remainder_start; word_idx < params.num_words; word_idx++) {
        distance += popcount(query[word_idx] ^ vectors[base_offset + word_idx]);
    }

    // Store similarity score (dimension - distance = number of matching bits)
    // Higher is more similar
    similarities[vec_idx] = (params.num_words * 32u) - distance;
}
"#;

/// WGSL shader for batched parallel similarity computation (2D dispatch)
///
/// Uses `global_invocation_id.x` for vector index, `.y` for query index.
/// Output is laid out as `similarities[query_idx * num_vectors + vec_idx]`.
const BATCH_SIMILARITY_SHADER: &str = r#"
struct BatchParams {
    num_vectors: u32,
    num_words: u32,
    num_queries: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> queries: array<u32>;
@group(0) @binding(1) var<storage, read> vectors: array<u32>;
@group(0) @binding(2) var<storage, read_write> similarities: array<u32>;
@group(0) @binding(3) var<uniform> params: BatchParams;

fn popcount(x: u32) -> u32 {
    var v = x;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    v = v + (v >> 8u);
    v = v + (v >> 16u);
    return v & 0x3Fu;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vec_idx = global_id.x;
    let query_idx = global_id.y;

    if (vec_idx >= params.num_vectors || query_idx >= params.num_queries) {
        return;
    }

    var distance: u32 = 0u;
    let vec_base = vec_idx * params.num_words;
    let query_base = query_idx * params.num_words;

    let num_chunks = params.num_words / 4u;
    for (var chunk = 0u; chunk < num_chunks; chunk++) {
        let w = chunk * 4u;
        let d0 = popcount(queries[query_base + w] ^ vectors[vec_base + w]);
        let d1 = popcount(queries[query_base + w + 1u] ^ vectors[vec_base + w + 1u]);
        let d2 = popcount(queries[query_base + w + 2u] ^ vectors[vec_base + w + 2u]);
        let d3 = popcount(queries[query_base + w + 3u] ^ vectors[vec_base + w + 3u]);
        distance += d0 + d1 + d2 + d3;
    }

    let remainder_start = num_chunks * 4u;
    for (var w = remainder_start; w < params.num_words; w++) {
        distance += popcount(queries[query_base + w] ^ vectors[vec_base + w]);
    }

    similarities[query_idx * params.num_vectors + vec_idx] = (params.num_words * 32u) - distance;
}
"#;

impl GpuContext {
    /// Create a new GPU context
    pub async fn new() -> AmorphicResult<Self> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
            memory_budget_thresholds: Default::default(),
        });

        // Request adapter (wgpu 28 returns Result)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                AmorphicError::QueryError(format!("Failed to get GPU adapter: {:?}", e))
            })?;

        // Get device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Amorphic GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| AmorphicError::QueryError(format!("Failed to get GPU device: {}", e)))?;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Similarity Shader"),
            source: wgpu::ShaderSource::Wgsl(SIMILARITY_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Similarity Bind Group Layout"),
            entries: &[
                // Query vector
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Database vectors
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output similarities
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Parameters
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout (wgpu 28 uses immediate_size instead of push_constant_ranges)
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Similarity Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create compute pipeline
        let similarity_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Similarity Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create batched shader module (2D dispatch)
        let batch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Batch Similarity Shader"),
            source: wgpu::ShaderSource::Wgsl(BATCH_SIMILARITY_SHADER.into()),
        });

        // Batch bind group layout (same bindings: queries, vectors, similarities, params)
        let batch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Batch Similarity Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let batch_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Batch Similarity Pipeline Layout"),
                bind_group_layouts: &[Some(&batch_bind_group_layout)],
                immediate_size: 0,
            });

        let batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batch Similarity Pipeline"),
            layout: Some(&batch_pipeline_layout),
            module: &batch_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            similarity_pipeline,
            bind_group_layout,
            batch_pipeline,
            batch_bind_group_layout,
        })
    }

    /// Compute similarities between a query vector and all database vectors
    /// Returns similarity scores (higher = more similar)
    pub fn compute_similarities(
        &self,
        query: &BinaryHV,
        vectors: &[BinaryHV],
    ) -> AmorphicResult<Vec<u32>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let num_vectors = vectors.len();
        let num_words = (DIMENSION + 63) / 64;

        // Prepare query data (convert u64 to u32 pairs)
        let query_words = query.as_words();
        let mut query_data: Vec<u32> = Vec::with_capacity(num_words * 2);
        for &word in query_words {
            query_data.push(word as u32);
            query_data.push((word >> 32) as u32);
        }

        // Prepare vectors data (all vectors concatenated)
        let mut vectors_data: Vec<u32> = Vec::with_capacity(num_vectors * num_words * 2);
        for hv in vectors {
            for &word in hv.as_words() {
                vectors_data.push(word as u32);
                vectors_data.push((word >> 32) as u32);
            }
        }

        // Create buffers
        let query_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Query Buffer"),
                contents: bytemuck::cast_slice(&query_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let vectors_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vectors Buffer"),
                contents: bytemuck::cast_slice(&vectors_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let similarities_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Similarities Buffer"),
            size: (num_vectors * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = SimilarityParams {
            num_vectors: num_vectors as u32,
            num_words: (num_words * 2) as u32, // We split u64 into pairs of u32
            _padding: [0; 2],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Similarity Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: similarities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Similarity Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Similarity Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.similarity_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with enough workgroups to cover all vectors
            let workgroups = (num_vectors as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy results to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (num_vectors * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &similarities_buffer,
            0,
            &staging_buffer,
            0,
            (num_vectors * 4) as u64,
        );

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Wait for GPU to complete (wgpu 28 uses PollType)
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|_| AmorphicError::QueryError("GPU channel closed".to_string()))?
            .map_err(|e| AmorphicError::QueryError(format!("Buffer map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let similarities: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(similarities)
    }

    /// Create GPU context using pollster for synchronous creation
    pub fn new_sync() -> AmorphicResult<Self> {
        pollster::block_on(Self::new())
    }

    /// Compute similarities using pre-uploaded vectors (FAST path)
    ///
    /// This is much faster than `compute_similarities` because vectors
    /// are already on GPU - only the query needs to be uploaded.
    pub fn compute_similarities_fast(
        &self,
        query: &BinaryHV,
        store: &GpuVectorStore,
    ) -> AmorphicResult<Vec<u32>> {
        let num_vectors = store.num_vectors;
        let num_words = store.num_words;

        // Prepare query data (convert u64 to u32 pairs)
        let query_words = query.as_words();
        let mut query_data: Vec<u32> = Vec::with_capacity(num_words * 2);
        for &word in query_words {
            query_data.push(word as u32);
            query_data.push((word >> 32) as u32);
        }

        // Only need to create query buffer (vectors already on GPU)
        let query_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Query Buffer"),
                contents: bytemuck::cast_slice(&query_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let similarities_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Similarities Buffer"),
            size: (num_vectors * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = SimilarityParams {
            num_vectors: num_vectors as u32,
            num_words: (num_words * 2) as u32,
            _padding: [0; 2],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group using persistent vectors buffer
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Similarity Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: store.vectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: similarities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Similarity Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Similarity Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.similarity_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (num_vectors as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy results to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (num_vectors * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &similarities_buffer,
            0,
            &staging_buffer,
            0,
            (num_vectors * 4) as u64,
        );

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|_| AmorphicError::QueryError("GPU channel closed".to_string()))?
            .map_err(|e| AmorphicError::QueryError(format!("Buffer map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let similarities: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(similarities)
    }

    /// Compute similarities for a batch of queries (BATCHED path)
    ///
    /// Uses a single 2D GPU dispatch (`x` = vector index, `y` = query index)
    /// to compute all query×vector similarities in parallel, amortizing buffer
    /// creation and command submission overhead across the entire batch.
    pub fn compute_similarities_batch(
        &self,
        queries: &[BinaryHV],
        store: &GpuVectorStore,
    ) -> AmorphicResult<Vec<Vec<u32>>> {
        if queries.is_empty() {
            return Ok(vec![]);
        }

        let num_queries = queries.len();
        let num_vectors = store.num_vectors;
        let num_words = store.num_words;

        // Pack all queries into a single contiguous buffer
        let mut queries_data: Vec<u32> = Vec::with_capacity(num_queries * num_words * 2);
        for q in queries {
            for &word in q.as_words() {
                queries_data.push(word as u32);
                queries_data.push((word >> 32) as u32);
            }
        }

        let queries_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batch Queries Buffer"),
                contents: bytemuck::cast_slice(&queries_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Output: num_queries × num_vectors similarity scores
        let total_output = num_queries * num_vectors;
        let similarities_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Similarities Buffer"),
            size: (total_output * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = BatchSimilarityParams {
            num_vectors: num_vectors as u32,
            num_words: (num_words * 2) as u32,
            num_queries: num_queries as u32,
            _padding: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batch Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Batch Similarity Bind Group"),
            layout: &self.batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: queries_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: store.vectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: similarities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Similarity Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batch Similarity Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.batch_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // 2D dispatch: x covers vectors, y covers queries
            let workgroups_x = (num_vectors as u32 + 255) / 256;
            let workgroups_y = num_queries as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Staging Buffer"),
            size: (total_output * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &similarities_buffer,
            0,
            &staging_buffer,
            0,
            (total_output * 4) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|_| AmorphicError::QueryError("GPU batch channel closed".to_string()))?
            .map_err(|e| AmorphicError::QueryError(format!("Batch buffer map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let flat: &[u32] = bytemuck::cast_slice(&data);

        // Split flat output into per-query result vectors
        let results: Vec<Vec<u32>> = (0..num_queries)
            .map(|q| flat[q * num_vectors..(q + 1) * num_vectors].to_vec())
            .collect();

        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}
