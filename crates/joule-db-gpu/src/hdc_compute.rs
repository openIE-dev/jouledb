//! GPU-accelerated Hyperdimensional Computing (HDC) encoding via WebGPU compute shaders.
//!
//! This module provides GPU-accelerated operations for binary hyperdimensional computing:
//! - XOR binding of binary hypervectors (packed as `u32` words)
//! - Population count (Hamming distance) computation
//! - Batch encoding of key-value pairs into hypervectors
//!
//! Vectors are represented as arrays of `u32`, where each `u32` packs 32 bits of
//! the binary hypervector. A 512-dimensional vector requires 16 `u32` words.
//!
//! # Example
//!
//! ```rust,ignore
//! use joule_db_gpu::hdc_compute::{HdcGpuConfig, HdcGpuEncoder};
//!
//! // Create encoder with default config (dim=512, batch=10000, workgroup=256)
//! let config = HdcGpuConfig::default();
//! let encoder = HdcGpuEncoder::new(&device);
//!
//! // XOR-bind two batches of vectors on the GPU
//! let result = encoder.bind_batch(&device, &queue, &a_words, &b_words, dim_words);
//! ```

use std::sync::mpsc;

/// WGSL compute shader for XOR binding of two binary hypervectors.
///
/// Each invocation processes one `u32` word, XOR-ing the corresponding words
/// from vectors `a` and `b` into the `result` buffer.
pub const HDC_XOR_BIND_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(256)
fn xor_bind(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx < arrayLength(&a) {
        result[idx] = a[idx] ^ b[idx];
    }
}
"#;

/// WGSL compute shader for population count (Hamming distance).
///
/// Computes the Hamming distance between pairs of binary hypervectors.
/// Each invocation handles one vector pair identified by `id.x`. The shader
/// iterates over the `dim_words` packed `u32` words of each vector, XOR-ing
/// corresponding words and counting the resulting set bits with `countOneBits()`.
///
/// Buffer layout:
/// - `a`: contiguous packed vectors, each `dim_words` u32 elements
/// - `b`: contiguous packed vectors, same layout as `a`
/// - `distances`: one `u32` per vector pair, holding the Hamming distance
/// - `params.x`: number of `u32` words per vector (`dim_words`)
/// - `params.y`: total number of vector pairs (`count`)
pub const HDC_POPCOUNT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> distances: array<u32>;
@group(0) @binding(3) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn popcount(@builtin(global_invocation_id) id: vec3<u32>) {
    let vec_idx = id.x;
    let dim_words = params.x;
    let count = params.y;

    if vec_idx >= count {
        return;
    }

    let base = vec_idx * dim_words;
    var dist: u32 = 0u;
    for (var w = 0u; w < dim_words; w = w + 1u) {
        let xor_val = a[base + w] ^ b[base + w];
        dist = dist + countOneBits(xor_val);
    }
    distances[vec_idx] = dist;
}
"#;

/// WGSL compute shader for batch encoding of key-value pairs into hypervectors.
///
/// For each record (identified by `id.x`), the shader reads a key hypervector and
/// a value hypervector from the `keys` and `values` buffers, XOR-binds them, and
/// writes the result into the `encoded` buffer. This implements the standard HDC
/// record encoding: `encode(k, v) = k XOR v`.
///
/// Buffer layout:
/// - `keys`: contiguous packed key hypervectors, each `dim_words` u32 elements
/// - `values`: contiguous packed value hypervectors, same layout
/// - `encoded`: output buffer, same layout
/// - `params.x`: number of `u32` words per vector (`dim_words`)
/// - `params.y`: number of records to encode (`num_records`)
pub const HDC_BATCH_ENCODE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> encoded: array<u32>;
@group(0) @binding(3) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn batch_encode(@builtin(global_invocation_id) id: vec3<u32>) {
    let record_idx = id.x;
    let dim_words = params.x;
    let num_records = params.y;

    if record_idx >= num_records {
        return;
    }

    let base = record_idx * dim_words;
    for (var w = 0u; w < dim_words; w = w + 1u) {
        encoded[base + w] = keys[base + w] ^ values[base + w];
    }
}
"#;

/// Configuration for HDC GPU encoding operations.
#[derive(Debug, Clone)]
pub struct HdcGpuConfig {
    /// Hypervector dimension in bits (default: 512).
    pub dimension: usize,
    /// Maximum number of vectors to process in a single GPU dispatch (default: 10000).
    pub batch_size: usize,
    /// Number of invocations per workgroup (default: 256). Must match the
    /// `@workgroup_size` declared in the WGSL shaders.
    pub workgroup_size: u32,
}

impl Default for HdcGpuConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            batch_size: 10000,
            workgroup_size: 256,
        }
    }
}

/// GPU-accelerated HDC encoder that manages compiled compute pipelines for
/// binary hypervector operations.
///
/// `HdcGpuEncoder` compiles three compute pipelines on construction:
/// - **xor_bind** -- element-wise XOR of packed `u32` arrays
/// - **popcount** -- Hamming distance via `countOneBits()` reduction
/// - **batch_encode** -- per-record key XOR value encoding
///
/// All methods accept explicit `wgpu::Device` and `wgpu::Queue` references
/// so callers can share a single GPU context across subsystems.
pub struct HdcGpuEncoder {
    /// Compiled pipeline for the XOR bind shader.
    xor_bind_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for the XOR bind pipeline (3 storage buffers).
    xor_bind_layout: wgpu::BindGroupLayout,

    /// Compiled pipeline for the popcount / Hamming distance shader.
    popcount_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for the popcount pipeline (3 storage + 1 uniform).
    popcount_layout: wgpu::BindGroupLayout,

    /// Compiled pipeline for the batch encode shader.
    batch_encode_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for the batch encode pipeline (3 storage + 1 uniform).
    batch_encode_layout: wgpu::BindGroupLayout,
}

impl HdcGpuEncoder {
    /// Create a new `HdcGpuEncoder`, compiling all three compute pipelines.
    ///
    /// This is a relatively expensive operation (shader compilation) and should
    /// be done once, reusing the encoder for many dispatches.
    pub fn new(device: &wgpu::Device) -> Self {
        // --- XOR bind pipeline (3 storage buffers, no uniform) ---
        let xor_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdc_xor_bind_layout"),
            entries: &[
                // binding 0: a (read)
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
                // binding 1: b (read)
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
                // binding 2: result (read_write)
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
            ],
        });

        let xor_bind_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hdc_xor_bind_shader"),
            source: wgpu::ShaderSource::Wgsl(HDC_XOR_BIND_SHADER.into()),
        });

        let xor_bind_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hdc_xor_bind_pipeline_layout"),
                bind_group_layouts: &[Some(&xor_bind_layout)],
                immediate_size: 0,
            });

        let xor_bind_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hdc_xor_bind_pipeline"),
            layout: Some(&xor_bind_pipeline_layout),
            module: &xor_bind_shader,
            entry_point: Some("xor_bind"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // --- Popcount pipeline (3 storage + 1 uniform) ---
        let popcount_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdc_popcount_layout"),
            entries: &[
                // binding 0: a (read)
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
                // binding 1: b (read)
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
                // binding 2: distances (read_write)
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
                // binding 3: params uniform (vec2<u32>)
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

        let popcount_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hdc_popcount_shader"),
            source: wgpu::ShaderSource::Wgsl(HDC_POPCOUNT_SHADER.into()),
        });

        let popcount_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hdc_popcount_pipeline_layout"),
                bind_group_layouts: &[Some(&popcount_layout)],
                immediate_size: 0,
            });

        let popcount_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hdc_popcount_pipeline"),
            layout: Some(&popcount_pipeline_layout),
            module: &popcount_shader,
            entry_point: Some("popcount"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // --- Batch encode pipeline (3 storage + 1 uniform) ---
        let batch_encode_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hdc_batch_encode_layout"),
                entries: &[
                    // binding 0: keys (read)
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
                    // binding 1: values (read)
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
                    // binding 2: encoded (read_write)
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
                    // binding 3: params uniform (vec2<u32>)
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

        let batch_encode_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hdc_batch_encode_shader"),
            source: wgpu::ShaderSource::Wgsl(HDC_BATCH_ENCODE_SHADER.into()),
        });

        let batch_encode_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hdc_batch_encode_pipeline_layout"),
                bind_group_layouts: &[Some(&batch_encode_layout)],
                immediate_size: 0,
            });

        let batch_encode_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hdc_batch_encode_pipeline"),
                layout: Some(&batch_encode_pipeline_layout),
                module: &batch_encode_shader,
                entry_point: Some("batch_encode"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Self {
            xor_bind_pipeline,
            xor_bind_layout,
            popcount_pipeline,
            popcount_layout,
            batch_encode_pipeline,
            batch_encode_layout,
        }
    }

    /// XOR-bind two batches of packed binary hypervectors on the GPU.
    ///
    /// Both `a` and `b` must contain the same number of `u32` elements.
    /// The element count must be a multiple of `dim_words` (i.e. each slice
    /// holds `n` vectors of `dim_words` words each, laid out contiguously).
    ///
    /// Returns a `Vec<u32>` of the same length as the inputs, containing
    /// `a[i] ^ b[i]` for every element.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create buffers and encode commands.
    /// * `queue` - The wgpu queue to submit work.
    /// * `a` - Packed binary vectors (first operand).
    /// * `b` - Packed binary vectors (second operand), same length as `a`.
    /// * `dim_words` - Number of `u32` words per single hypervector (unused in
    ///   the flat XOR but kept for API consistency and future batching).
    pub fn bind_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        a: &[u32],
        b: &[u32],
        _dim_words: u32,
    ) -> Vec<u32> {
        assert_eq!(a.len(), b.len(), "a and b must have the same length");

        let total_words = a.len() as u64;
        let buf_size = total_words * 4;

        // Upload input buffers
        let a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_xor_a"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));

        let b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_xor_b"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));

        // Output buffer
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_xor_result"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdc_xor_bind_group"),
            layout: &self.xor_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroup_count = ((total_words as u32) + 255) / 256;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdc_xor_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hdc_xor_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.xor_bind_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Read back via staging buffer
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_xor_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging, 0, buf_size);

        queue.submit(std::iter::once(encoder.finish()));

        Self::read_back_u32(device, &staging, total_words as usize)
    }

    /// Compute the Hamming distance between corresponding pairs of packed
    /// binary hypervectors on the GPU.
    ///
    /// `a` and `b` must each contain `count * dim_words` elements, where
    /// `count` is the number of vector pairs and `dim_words` is the number
    /// of `u32` words per vector.
    ///
    /// Returns a `Vec<u32>` of length `count`, where each element is the
    /// Hamming distance between the corresponding pair.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `a` - First batch of packed vectors (`count * dim_words` elements).
    /// * `b` - Second batch of packed vectors (same length as `a`).
    /// * `dim_words` - Number of `u32` words per single hypervector.
    /// * `count` - Number of vector pairs.
    pub fn hamming_distance_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        a: &[u32],
        b: &[u32],
        dim_words: u32,
        count: u32,
    ) -> Vec<u32> {
        assert_eq!(a.len(), b.len(), "a and b must have the same length");
        assert_eq!(
            a.len(),
            (dim_words as usize) * (count as usize),
            "a length must equal dim_words * count"
        );

        let input_size = (a.len() as u64) * 4;
        let output_size = (count as u64) * 4;

        // Upload inputs
        let a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_pop_a"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));

        let b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_pop_b"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));

        // Output buffer (one u32 per vector pair)
        let distances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_pop_distances"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params uniform: vec2<u32> = (dim_words, count), padded to 8 bytes
        let params: [u32; 2] = [dim_words, count];
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_pop_params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdc_pop_bind_group"),
            layout: &self.popcount_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: distances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroup_count = (count + 255) / 256;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdc_pop_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hdc_pop_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.popcount_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Read back
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_pop_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&distances_buffer, 0, &staging, 0, output_size);

        queue.submit(std::iter::once(encoder.finish()));

        Self::read_back_u32(device, &staging, count as usize)
    }

    /// Batch-encode key-value pairs into hypervectors on the GPU.
    ///
    /// Each record is encoded as `key_hv XOR value_hv`. Both `keys` and
    /// `values` must contain `num_records * dim_words` elements.
    ///
    /// Returns a `Vec<u32>` of the same length, containing the encoded
    /// hypervectors laid out contiguously.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `keys` - Packed key hypervectors.
    /// * `values` - Packed value hypervectors (same length as `keys`).
    /// * `dim_words` - Number of `u32` words per single hypervector.
    /// * `num_records` - Number of key-value pairs to encode.
    pub fn batch_encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        keys: &[u32],
        values: &[u32],
        dim_words: u32,
        num_records: u32,
    ) -> Vec<u32> {
        assert_eq!(
            keys.len(),
            values.len(),
            "keys and values must have the same length"
        );
        assert_eq!(
            keys.len(),
            (dim_words as usize) * (num_records as usize),
            "keys length must equal dim_words * num_records"
        );

        let buf_size = (keys.len() as u64) * 4;

        // Upload inputs
        let keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_enc_keys"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&keys_buffer, 0, bytemuck::cast_slice(keys));

        let values_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_enc_values"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&values_buffer, 0, bytemuck::cast_slice(values));

        // Output buffer
        let encoded_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_enc_output"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params uniform: vec2<u32> = (dim_words, num_records)
        let params: [u32; 2] = [dim_words, num_records];
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_enc_params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdc_enc_bind_group"),
            layout: &self.batch_encode_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch -- one invocation per record
        let workgroup_count = (num_records + 255) / 256;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdc_enc_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hdc_enc_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.batch_encode_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Read back
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdc_enc_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&encoded_buffer, 0, &staging, 0, buf_size);

        queue.submit(std::iter::once(encoder.finish()));

        Self::read_back_u32(device, &staging, keys.len())
    }

    /// Map a staging buffer back to the CPU and return its contents as `Vec<u32>`.
    ///
    /// Blocks until the GPU has finished writing and the buffer is mapped.
    fn read_back_u32(device: &wgpu::Device, staging: &wgpu::Buffer, num_words: usize) -> Vec<u32> {
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely());

        rx.recv()
            .expect("GPU map channel closed unexpectedly")
            .expect("GPU buffer map failed");

        let mapped = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&mapped)[..num_words].to_vec();
        drop(mapped);
        staging.unmap();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdc_gpu_config_default() {
        let config = HdcGpuConfig::default();
        assert_eq!(config.dimension, 512);
        assert_eq!(config.batch_size, 10000);
        assert_eq!(config.workgroup_size, 256);
    }

    #[test]
    fn test_hdc_gpu_config_custom() {
        let config = HdcGpuConfig {
            dimension: 1024,
            batch_size: 50000,
            workgroup_size: 128,
        };
        assert_eq!(config.dimension, 1024);
        assert_eq!(config.batch_size, 50000);
        assert_eq!(config.workgroup_size, 128);
    }

    #[test]
    fn test_shader_sources_non_empty() {
        assert!(!HDC_XOR_BIND_SHADER.is_empty());
        assert!(!HDC_POPCOUNT_SHADER.is_empty());
        assert!(!HDC_BATCH_ENCODE_SHADER.is_empty());
    }

    #[test]
    fn test_shader_contains_entry_points() {
        assert!(HDC_XOR_BIND_SHADER.contains("fn xor_bind"));
        assert!(HDC_POPCOUNT_SHADER.contains("fn popcount"));
        assert!(HDC_BATCH_ENCODE_SHADER.contains("fn batch_encode"));
    }

    #[test]
    fn test_shader_uses_workgroup_size_256() {
        assert!(HDC_XOR_BIND_SHADER.contains("@workgroup_size(256)"));
        assert!(HDC_POPCOUNT_SHADER.contains("@workgroup_size(256)"));
        assert!(HDC_BATCH_ENCODE_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_popcount_shader_uses_count_one_bits() {
        assert!(HDC_POPCOUNT_SHADER.contains("countOneBits"));
    }

    #[test]
    fn test_dim_words_calculation() {
        // 512-bit vector requires 512/32 = 16 u32 words
        let dim = 512usize;
        let dim_words = dim / 32;
        assert_eq!(dim_words, 16);

        // 1024-bit vector requires 32 words
        let dim = 1024usize;
        let dim_words = dim / 32;
        assert_eq!(dim_words, 32);
    }
}
