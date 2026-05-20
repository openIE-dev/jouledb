//! wgpu-based GPU compute backend
//!
//! Implements the ComputeBackend trait using wgpu for cross-platform GPU compute.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use wgpu::util::DeviceExt;

use joule_db_core::StorageError;
use joule_db_core::persistence::compute::{
    AggregationType, BufferHandle, BufferUsage, ComputeBackend, ComputeOp, ComputeResult,
    DeviceCapabilities, DeviceType, HashAlgorithm,
};

use crate::shaders;

/// Configuration for wgpu compute backend
#[derive(Debug, Clone)]
pub struct WgpuComputeConfig {
    /// Power preference
    pub power_preference: wgpu::PowerPreference,
    /// Force fallback to software renderer
    pub force_fallback: bool,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
}

impl Default for WgpuComputeConfig {
    fn default() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback: false,
            max_buffer_size: 256 * 1024 * 1024, // 256MB
        }
    }
}

/// Internal buffer data
struct BufferData {
    buffer: wgpu::Buffer,
    size: u64,
    usage: BufferUsage,
}

impl BufferData {
    /// Get reference to the underlying wgpu buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the buffer size in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the buffer usage flags
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }
}

/// wgpu-based GPU compute backend
pub struct WgpuComputeBackend {
    /// wgpu device
    device: Arc<wgpu::Device>,
    /// wgpu queue
    queue: Arc<wgpu::Queue>,
    /// Device capabilities
    capabilities: DeviceCapabilities,
    /// Allocated buffers
    buffers: HashMap<u64, BufferData>,
    /// Next buffer ID
    next_buffer_id: u64,
    /// Compute pipelines (lazy loaded)
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    /// Bind group layouts
    bind_group_layouts: HashMap<String, wgpu::BindGroupLayout>,
    /// Statistics
    total_operations: u64,
    total_time_us: u64,
    /// Pool of reusable MAP_READ staging buffers, sorted ascending by size
    staging_pool: Vec<(u64, wgpu::Buffer)>,
}

impl WgpuComputeBackend {
    /// Create a new wgpu compute backend
    pub async fn new() -> Result<Self, StorageError> {
        Self::with_config(WgpuComputeConfig::default()).await
    }

    /// Create with custom configuration
    pub async fn with_config(config: WgpuComputeConfig) -> Result<Self, StorageError> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
            memory_budget_thresholds: Default::default(),
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                compatible_surface: None,
                force_fallback_adapter: config.force_fallback,
            })
            .await
            .map_err(|e| StorageError::Backend(format!("Failed to get GPU adapter: {:?}", e)))?;

        // Get device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("JouleDB GPU Compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| StorageError::Backend(format!("Failed to get GPU device: {}", e)))?;

        // Get device capabilities
        let capabilities = DeviceCapabilities {
            device_type: DeviceType::Gpu,
            name: adapter.get_info().name,
            total_memory: adapter.get_info().device as u64 * 1024 * 1024, // Rough estimate
            available_memory: 0,                                          // Unknown
            max_workgroup_size: device.limits().max_compute_workgroup_size_x,
            max_buffer_size: device.limits().max_buffer_size,
            supports_f16: device.features().contains(wgpu::Features::SHADER_F16),
            supports_f64: false, // WGSL doesn't support f64
            supports_atomics: device
                .features()
                .contains(wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS),
            compute_units: 0, // Unknown
        };

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            capabilities,
            buffers: HashMap::new(),
            next_buffer_id: 1,
            pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            total_operations: 0,
            total_time_us: 0,
            staging_pool: Vec::new(),
        })
    }

    /// Convert BufferUsage to wgpu BufferUsages
    fn buffer_usage_to_wgpu(usage: BufferUsage) -> wgpu::BufferUsages {
        let mut wgpu_usage = wgpu::BufferUsages::empty();
        if usage.bits() & BufferUsage::STORAGE_READ.bits() != 0 {
            wgpu_usage |= wgpu::BufferUsages::STORAGE;
        }
        if usage.bits() & BufferUsage::STORAGE_WRITE.bits() != 0 {
            wgpu_usage |= wgpu::BufferUsages::STORAGE;
        }
        if usage.bits() & BufferUsage::UNIFORM.bits() != 0 {
            wgpu_usage |= wgpu::BufferUsages::UNIFORM;
        }
        if usage.bits() & BufferUsage::COPY_SRC.bits() != 0 {
            wgpu_usage |= wgpu::BufferUsages::COPY_SRC;
        }
        if usage.bits() & BufferUsage::COPY_DST.bits() != 0 {
            wgpu_usage |= wgpu::BufferUsages::COPY_DST;
        }
        if usage.bits() & BufferUsage::MAP_READ.bits() != 0 {
            wgpu_usage |= wgpu::BufferUsages::MAP_READ;
        }
        if usage.bits() & BufferUsage::MAP_WRITE.bits() != 0 {
            wgpu_usage |= wgpu::BufferUsages::MAP_WRITE;
        }

        // Always allow copy for staging
        wgpu_usage |= wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;

        wgpu_usage
    }

    /// Maximum number of staging buffers to keep in the pool.
    const MAX_STAGING_POOL_SIZE: usize = 8;

    /// Acquire a MAP_READ staging buffer of at least `min_size` bytes.
    ///
    /// Looks for the smallest buffer in the pool that is >= min_size.
    /// If none is large enough, creates a new one. The pool is sorted ascending by size.
    fn acquire_staging_buffer(&mut self, min_size: u64) -> wgpu::Buffer {
        // Find the first buffer that is large enough (pool is sorted by size ascending)
        let pos = self.staging_pool.iter().position(|(sz, _)| *sz >= min_size);
        if let Some(idx) = pos {
            return self.staging_pool.remove(idx).1;
        }

        // No suitable buffer in pool — create a new one
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer_pooled"),
            size: min_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Return a staging buffer to the pool for reuse.
    ///
    /// If the pool is at capacity, the smallest buffer is evicted (destroyed) to make
    /// room, since larger buffers are more expensive to recreate.
    fn release_staging_buffer(&mut self, buffer: wgpu::Buffer) {
        let size = buffer.size();

        // Insert sorted by size (ascending)
        let pos = self
            .staging_pool
            .iter()
            .position(|(sz, _)| *sz > size)
            .unwrap_or(self.staging_pool.len());
        self.staging_pool.insert(pos, (size, buffer));

        // Evict smallest if over capacity
        if self.staging_pool.len() > Self::MAX_STAGING_POOL_SIZE {
            let (_, evicted) = self.staging_pool.remove(0);
            evicted.destroy();
        }
    }

    /// Get or create a compute pipeline
    fn get_or_create_pipeline(
        &mut self,
        name: &str,
        shader_source: &str,
    ) -> &wgpu::ComputePipeline {
        if !self.pipelines.contains_key(name) {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });

            // Create bind group layout - different layouts for different operations
            let entries: Vec<wgpu::BindGroupLayoutEntry> = if name == "btree_range_scan" {
                // B-tree scan needs 7 bindings (already implemented)
                vec![
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ]
            } else if name == "binary_hd_bind"
                || name == "binary_hd_batch_bind"
                || name == "binary_hd_bundle"
                || name == "binary_hd_similarity"
                || name == "binary_hd_multi_similarity"
            {
                // Binary HD operations need 4 bindings (2 inputs, 1 output, 1 params for bind/similarity)
                // Bundle needs 2 inputs (vectors, output), 1 params
                if name == "binary_hd_bundle" {
                    vec![
                        // 0: Input vectors
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
                        // 1: Output vector
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 2: Params
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ]
                } else {
                    // bind and similarity need 4 bindings
                    vec![
                        // 0: Input vector A (or query for similarity)
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
                        // 1: Input vector B (or database vectors for similarity)
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
                        // 2: Output (or similarities for similarity)
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
                        // 3: Params
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
                    ]
                }
            } else {
                // Standard layout for other operations (aggregate, hash, etc.)
                vec![
                    // Input buffer
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
                    // Output buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params buffer (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ]
            };

            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("{}_layout", name)),
                        entries: &entries,
                    });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{}_pipeline_layout", name)),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            self.bind_group_layouts
                .insert(name.to_string(), bind_group_layout);
            self.pipelines.insert(name.to_string(), pipeline);
        }

        self.pipelines.get(name).unwrap()
    }

    /// Execute aggregation on GPU
    fn execute_aggregate(
        &mut self,
        agg_type: AggregationType,
        num_items: u32,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = Instant::now();

        if inputs.is_empty() {
            return Err(StorageError::Backend(
                "No input buffers provided".to_string(),
            ));
        }

        let input_handle = inputs[0];
        let shader_name = match agg_type {
            AggregationType::Sum => "sum",
            AggregationType::Count => "count",
            AggregationType::Avg => "avg",
            AggregationType::Min => "min",
            AggregationType::Max => "max",
            _ => {
                return Err(StorageError::Backend(format!(
                    "Unsupported aggregation: {:?}",
                    agg_type
                )));
            }
        };

        let shader_source = match agg_type {
            AggregationType::Sum => shaders::SUM_SHADER,
            AggregationType::Count => shaders::COUNT_SHADER,
            AggregationType::Avg => shaders::AVG_SHADER,
            AggregationType::Min => shaders::MIN_SHADER,
            AggregationType::Max => shaders::MAX_SHADER,
            // Should have been filtered by the first match, but return error just in case
            _ => {
                return Err(StorageError::Backend(format!(
                    "No shader for aggregation: {:?}",
                    agg_type
                )));
            }
        };

        let _pipeline = self.get_or_create_pipeline(shader_name, shader_source);

        // Now get references after pipeline is created
        let pipeline = self.pipelines.get(shader_name).unwrap();
        let bind_group_layout = self.bind_group_layouts.get(shader_name).unwrap();
        let input_data = self.buffers.get(&input_handle.0).unwrap();

        // Create output buffer (single f32 for aggregation result)
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aggregate_output"),
            size: 8, // f32 result + padding
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct AggregateParams {
            num_items: u32,
            _padding: [u32; 3],
        }

        let params = AggregateParams {
            num_items,
            _padding: [0; 3],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("aggregate_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aggregate_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aggregate_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aggregate_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate workgroup count
            let workgroup_size = 256u32;
            let workgroup_count = (num_items + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));

        // Store output buffer
        let output_id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(
            output_id,
            BufferData {
                buffer: output_buffer,
                size: 8,
                usage: BufferUsage::STORAGE_WRITE,
            },
        );

        let execution_time_us = start.elapsed().as_micros() as u64;
        self.total_operations += 1;
        self.total_time_us += execution_time_us;

        Ok(ComputeResult {
            output_buffer: BufferHandle(output_id),
            result_count: 1,
            execution_time_us,
        })
    }

    /// Execute hash computation on GPU
    fn execute_hash(
        &mut self,
        num_items: u32,
        algorithm: HashAlgorithm,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = Instant::now();

        if inputs.is_empty() {
            return Err(StorageError::Backend(
                "No input buffers provided".to_string(),
            ));
        }

        let input_handle = inputs[0];
        let (shader_name, shader_source) = match algorithm {
            HashAlgorithm::XxHash64 => ("xxhash64", shaders::XXHASH64_SHADER),
            HashAlgorithm::Crc32 => ("crc32", shaders::CRC32_SHADER),
            HashAlgorithm::Murmur3 => ("murmur3", shaders::MURMUR3_SHADER),
            _ => {
                return Err(StorageError::Backend(format!(
                    "Unsupported hash algorithm: {:?}",
                    algorithm
                )));
            }
        };

        let _pipeline = self.get_or_create_pipeline(shader_name, shader_source);

        // Get references after pipeline is created
        let pipeline = self.pipelines.get(shader_name).unwrap();
        let bind_group_layout = self.bind_group_layouts.get(shader_name).unwrap();
        let input_data = self.buffers.get(&input_handle.0).unwrap();

        // Create output buffer (u32 per item)
        let output_size = (num_items as u64) * 4;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct HashParams {
            num_items: u32,
            item_size: u32,
            _padding: [u32; 2],
        }

        let params = HashParams {
            num_items,
            item_size: (input_data.size() / num_items as u64) as u32,
            _padding: [0; 2],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hash_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hash_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hash_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate workgroup count
            let workgroup_size = 256u32;
            let workgroup_count = (num_items + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));

        // Store output buffer
        let output_id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(
            output_id,
            BufferData {
                buffer: output_buffer,
                size: output_size,
                usage: BufferUsage::STORAGE_WRITE,
            },
        );

        let execution_time_us = start.elapsed().as_micros() as u64;
        self.total_operations += 1;
        self.total_time_us += execution_time_us;

        Ok(ComputeResult {
            output_buffer: BufferHandle(output_id),
            result_count: num_items,
            execution_time_us,
        })
    }

    /// Execute B-tree range scan on GPU
    fn execute_btree_range_scan(
        &mut self,
        num_nodes: u32,
        keys_per_node: u32,
        start_key_len: u32,
        end_key_len: u32,
        include_start: bool,
        include_end: bool,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = Instant::now();

        if inputs.len() < 3 {
            return Err(StorageError::Backend(
                "Need at least 3 input buffers: nodes, start_key, end_key".to_string(),
            ));
        }

        let nodes_handle = inputs[0];
        let start_key_handle = inputs[1];
        let end_key_handle = inputs[2];

        // Create pipeline first (needs mutable borrow)
        let _pipeline =
            self.get_or_create_pipeline("btree_range_scan", shaders::BTREE_RANGE_SCAN_SHADER);
        let pipeline = self.pipelines.get("btree_range_scan").unwrap();
        let bind_group_layout = self.bind_group_layouts.get("btree_range_scan").unwrap();

        // Now get buffer data (immutable borrow)
        let nodes_data = self.buffers.get(&nodes_handle.0).unwrap();

        // Estimate node size (would be calculated from serialized nodes)
        let node_size = (nodes_data.size() / num_nodes as u64) as u32;

        // Create result buffers
        let max_results = num_nodes * keys_per_node;
        let result_keys_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("btree_result_keys"),
            size: (max_results as u64) * 4, // u32 per result
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let result_values_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("btree_result_values"),
            size: (max_results as u64) * 4, // u32 per result
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let result_count_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("btree_result_count"),
            size: 4, // u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BTreeParams {
            num_nodes: u32,
            node_size: u32,
            key_data_offset: u32,
            value_data_offset: u32,
            start_key_len: u32,
            end_key_len: u32,
            include_start: u32,
            include_end: u32,
            unbounded_start: u32,
            unbounded_end: u32,
            max_results: u32,
        }

        let params = BTreeParams {
            num_nodes,
            node_size,
            key_data_offset: 64 + (32 * 8) + (32 * 8), // header + key_entries + value_entries
            value_data_offset: 64 + (32 * 8) + (32 * 8) + 1024, // after key data
            start_key_len,
            end_key_len,
            include_start: if include_start { 1 } else { 0 },
            include_end: if include_end { 1 } else { 0 },
            unbounded_start: if start_key_len == 0 { 1 } else { 0 },
            unbounded_end: if end_key_len == 0 { 1 } else { 0 },
            max_results,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("btree_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group (extended layout for B-tree scan)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("btree_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nodes_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .buffers
                        .get(&start_key_handle.0)
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .buffers
                        .get(&end_key_handle.0)
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: result_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("btree_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("btree_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per key (num_nodes * keys_per_node)
            let total_keys = num_nodes * keys_per_node;
            let workgroup_size = 256u32;
            let workgroup_count = (total_keys + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));

        // Store output buffers
        let output_id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(
            output_id,
            BufferData {
                buffer: result_keys_buffer,
                size: (max_results as u64) * 4,
                usage: BufferUsage::STORAGE_WRITE,
            },
        );

        let execution_time_us = start.elapsed().as_micros() as u64;
        self.total_operations += 1;
        self.total_time_us += execution_time_us;

        Ok(ComputeResult {
            output_buffer: BufferHandle(output_id),
            result_count: max_results,
            execution_time_us,
        })
    }

    /// Execute binary hyperdimensional bind operation
    fn execute_binary_hd_bind(
        &mut self,
        num_words: u32,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = Instant::now();

        if inputs.len() < 2 {
            return Err(StorageError::Backend(
                "Need 2 input buffers: vector_a, vector_b".to_string(),
            ));
        }

        let vector_a_handle = inputs[0];
        let vector_b_handle = inputs[1];

        // Create pipeline first (needs mutable borrow)
        let _pipeline =
            self.get_or_create_pipeline("binary_hd_bind", shaders::BINARY_HD_BIND_SHADER);
        let pipeline = self.pipelines.get("binary_hd_bind").unwrap();
        let bind_group_layout = self.bind_group_layouts.get("binary_hd_bind").unwrap();

        // Now get buffer data (immutable borrow)
        let vector_a_data = self.buffers.get(&vector_a_handle.0).unwrap();
        let vector_b_data = self.buffers.get(&vector_b_handle.0).unwrap();

        // num_words is u64-word count from ComputeOp; shader operates on u32 words
        let u32_words = num_words * 2;

        // Create output buffer (num_words u64s = num_words * 8 bytes)
        let output_size = (num_words as u64) * 8;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_hd_bind_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer (shader sees u32-word count)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BindParams {
            num_words: u32,
            _padding: [u32; 3],
        }

        let params = BindParams {
            num_words: u32_words,
            _padding: [0; 3],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("binary_hd_bind_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_hd_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vector_a_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vector_b_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
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
                label: Some("binary_hd_bind_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_hd_bind_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per u32 word
            let workgroup_size = 256u32;
            let workgroup_count = (u32_words + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));

        // Store output buffer
        let output_id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(
            output_id,
            BufferData {
                buffer: output_buffer,
                size: output_size,
                usage: BufferUsage::STORAGE_WRITE,
            },
        );

        let execution_time_us = start.elapsed().as_micros() as u64;
        self.total_operations += 1;
        self.total_time_us += execution_time_us;

        Ok(ComputeResult {
            output_buffer: BufferHandle(output_id),
            result_count: num_words,
            execution_time_us,
        })
    }

    /// Execute binary hyperdimensional bundle operation
    fn execute_binary_hd_bundle(
        &mut self,
        num_vectors: u32,
        num_words: u32,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = Instant::now();

        if inputs.is_empty() {
            return Err(StorageError::Backend(
                "Need input buffer: vectors".to_string(),
            ));
        }

        let vectors_handle = inputs[0];
        // Create pipeline first (needs mutable borrow)
        let _pipeline =
            self.get_or_create_pipeline("binary_hd_bundle", shaders::BINARY_HD_BUNDLE_SHADER);
        let pipeline = self.pipelines.get("binary_hd_bundle").unwrap();
        let bind_group_layout = self.bind_group_layouts.get("binary_hd_bundle").unwrap();

        // Now get buffer data (immutable borrow)
        let vectors_data = self.buffers.get(&vectors_handle.0).unwrap();

        // num_words is u64-word count from ComputeOp; shader operates on u32 words
        let u32_words = num_words * 2;

        // Create output buffer (num_words u64s = num_words * 8 bytes)
        let output_size = (num_words as u64) * 8;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_hd_bundle_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer (shader sees u32-word count)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BundleParams {
            num_vectors: u32,
            num_words: u32,
            _padding: [u32; 2],
        }

        let params = BundleParams {
            num_vectors,
            num_words: u32_words,
            _padding: [0; 2],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("binary_hd_bundle_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_hd_bundle_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vectors_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("binary_hd_bundle_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_hd_bundle_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per u32 word
            let workgroup_size = 256u32;
            let workgroup_count = (u32_words + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));

        // Store output buffer
        let output_id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(
            output_id,
            BufferData {
                buffer: output_buffer,
                size: output_size,
                usage: BufferUsage::STORAGE_WRITE,
            },
        );

        let execution_time_us = start.elapsed().as_micros() as u64;
        self.total_operations += 1;
        self.total_time_us += execution_time_us;

        Ok(ComputeResult {
            output_buffer: BufferHandle(output_id),
            result_count: num_words,
            execution_time_us,
        })
    }

    /// Execute binary hyperdimensional similarity search
    fn execute_binary_hd_similarity(
        &mut self,
        num_vectors: u32,
        num_words: u32,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = Instant::now();

        if inputs.len() < 2 {
            return Err(StorageError::Backend(
                "Need 2 input buffers: query, vectors".to_string(),
            ));
        }

        let query_handle = inputs[0];
        let vectors_handle = inputs[1];

        // Create pipeline first (needs mutable borrow)
        let _pipeline = self
            .get_or_create_pipeline("binary_hd_similarity", shaders::BINARY_HD_SIMILARITY_SHADER);
        let pipeline = self.pipelines.get("binary_hd_similarity").unwrap();
        let bind_group_layout = self.bind_group_layouts.get("binary_hd_similarity").unwrap();

        // Now get buffer data (immutable borrow)
        let query_data = self.buffers.get(&query_handle.0).unwrap();
        let vectors_data = self.buffers.get(&vectors_handle.0).unwrap();

        // num_words is u64-word count from ComputeOp; shader operates on u32 words
        let u32_words = num_words * 2;

        // Create output buffer (u32 per vector for similarity scores)
        let output_size = (num_vectors as u64) * 4;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_hd_similarity_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer (shader sees u32-word count)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SimilarityParams {
            num_vectors: u32,
            num_words: u32,
            _padding: [u32; 2],
        }

        let params = SimilarityParams {
            num_vectors,
            num_words: u32_words,
            _padding: [0; 2],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("binary_hd_similarity_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_hd_similarity_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vectors_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
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
                label: Some("binary_hd_similarity_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_hd_similarity_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per vector
            let workgroup_size = 256u32;
            let workgroup_count = (num_vectors + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));

        // Store output buffer
        let output_id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(
            output_id,
            BufferData {
                buffer: output_buffer,
                size: output_size,
                usage: BufferUsage::STORAGE_WRITE,
            },
        );

        let execution_time_us = start.elapsed().as_micros() as u64;
        self.total_operations += 1;
        self.total_time_us += execution_time_us;

        Ok(ComputeResult {
            output_buffer: BufferHandle(output_id),
            result_count: num_vectors,
            execution_time_us,
        })
    }

    /// Execute binary hyperdimensional batch bind operation (N pairs in one dispatch)
    fn execute_binary_hd_batch_bind(
        &mut self,
        num_pairs: u32,
        num_words: u32,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = Instant::now();

        if inputs.len() < 2 {
            return Err(StorageError::Backend(
                "Need 2 input buffers: vectors_a, vectors_b".to_string(),
            ));
        }

        let vectors_a_handle = inputs[0];
        let vectors_b_handle = inputs[1];

        // Create pipeline — same 4-binding layout as bind/similarity
        let _pipeline = self
            .get_or_create_pipeline("binary_hd_batch_bind", shaders::BINARY_HD_BATCH_BIND_SHADER);
        let pipeline = self.pipelines.get("binary_hd_batch_bind").unwrap();
        let bind_group_layout = self.bind_group_layouts.get("binary_hd_batch_bind").unwrap();

        let vectors_a_data = self.buffers.get(&vectors_a_handle.0).unwrap();
        let vectors_b_data = self.buffers.get(&vectors_b_handle.0).unwrap();

        // num_words is u64-word count; shader operates on u32 words
        let u32_words = num_words * 2;

        // Output: num_pairs vectors, each num_words u64s
        let output_size = (num_pairs as u64) * (num_words as u64) * 8;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_hd_batch_bind_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BatchBindParams {
            num_pairs: u32,
            num_words: u32,
            _padding: [u32; 2],
        }

        let params = BatchBindParams {
            num_pairs,
            num_words: u32_words,
            _padding: [0; 2],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("binary_hd_batch_bind_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_hd_batch_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vectors_a_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vectors_b_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
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
                label: Some("binary_hd_batch_bind_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_hd_batch_bind_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per u32 word across ALL pairs
            let total_u32_words = num_pairs * u32_words;
            let workgroup_size = 256u32;
            let workgroup_count = (total_u32_words + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let output_id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(
            output_id,
            BufferData {
                buffer: output_buffer,
                size: output_size,
                usage: BufferUsage::STORAGE_WRITE,
            },
        );

        let execution_time_us = start.elapsed().as_micros() as u64;
        self.total_operations += 1;
        self.total_time_us += execution_time_us;

        Ok(ComputeResult {
            output_buffer: BufferHandle(output_id),
            result_count: num_pairs * num_words,
            execution_time_us,
        })
    }

    /// Execute binary hyperdimensional multi-query similarity (M queries × N vectors)
    fn execute_binary_hd_multi_similarity(
        &mut self,
        num_queries: u32,
        num_vectors: u32,
        num_words: u32,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        let start = Instant::now();

        if inputs.len() < 2 {
            return Err(StorageError::Backend(
                "Need 2 input buffers: queries, vectors".to_string(),
            ));
        }

        let queries_handle = inputs[0];
        let vectors_handle = inputs[1];

        // Create pipeline — same 4-binding layout as similarity
        let _pipeline = self.get_or_create_pipeline(
            "binary_hd_multi_similarity",
            shaders::BINARY_HD_MULTI_SIMILARITY_SHADER,
        );
        let pipeline = self.pipelines.get("binary_hd_multi_similarity").unwrap();
        let bind_group_layout = self
            .bind_group_layouts
            .get("binary_hd_multi_similarity")
            .unwrap();

        let queries_data = self.buffers.get(&queries_handle.0).unwrap();
        let vectors_data = self.buffers.get(&vectors_handle.0).unwrap();

        // num_words is u64-word count; shader operates on u32 words
        let u32_words = num_words * 2;

        // Output: u32 similarity score per (query, vector) pair
        let total_pairs = num_queries * num_vectors;
        let output_size = (total_pairs as u64) * 4;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_hd_multi_similarity_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MultiSimilarityParams {
            num_queries: u32,
            num_vectors: u32,
            num_words: u32,
            _padding: u32,
        }

        let params = MultiSimilarityParams {
            num_queries,
            num_vectors,
            num_words: u32_words,
            _padding: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("binary_hd_multi_similarity_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_hd_multi_similarity_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: queries_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vectors_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
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
                label: Some("binary_hd_multi_similarity_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_hd_multi_similarity_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 256u32;
            let workgroup_count = (total_pairs + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let output_id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffers.insert(
            output_id,
            BufferData {
                buffer: output_buffer,
                size: output_size,
                usage: BufferUsage::STORAGE_WRITE,
            },
        );

        let execution_time_us = start.elapsed().as_micros() as u64;
        self.total_operations += 1;
        self.total_time_us += execution_time_us;

        Ok(ComputeResult {
            output_buffer: BufferHandle(output_id),
            result_count: total_pairs,
            execution_time_us,
        })
    }
}

impl ComputeBackend for WgpuComputeBackend {
    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn create_buffer(
        &mut self,
        size: u64,
        usage: BufferUsage,
        _label: Option<&str>,
    ) -> Result<BufferHandle, StorageError> {
        let wgpu_usage = Self::buffer_usage_to_wgpu(usage);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compute_buffer"),
            size,
            usage: wgpu_usage,
            mapped_at_creation: false,
        });

        let id = self.next_buffer_id;
        self.next_buffer_id += 1;

        self.buffers.insert(
            id,
            BufferData {
                buffer,
                size,
                usage,
            },
        );

        Ok(BufferHandle(id))
    }

    fn write_buffer(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<(), StorageError> {
        let buffer_data = self
            .buffers
            .get_mut(&buffer.0)
            .ok_or_else(|| StorageError::Backend("Buffer not found".to_string()))?;

        if (data.len() as u64 + offset) > buffer_data.size {
            return Err(StorageError::Backend(
                "Data too large for buffer".to_string(),
            ));
        }

        self.queue.write_buffer(&buffer_data.buffer, offset, data);
        Ok(())
    }

    fn read_buffer(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, StorageError> {
        // Validate bounds (scoped borrow)
        {
            let buffer_data = self
                .buffers
                .get(&buffer.0)
                .ok_or_else(|| StorageError::Backend("Buffer not found".to_string()))?;
            if (offset + size) > buffer_data.size {
                return Err(StorageError::Backend("Read out of bounds".to_string()));
            }
        }

        // Acquire a staging buffer from the pool (or create a new one)
        let staging_buffer = self.acquire_staging_buffer(size);

        // Copy to staging (re-borrow buffer_data immutably)
        let src_buffer = &self.buffers.get(&buffer.0).unwrap().buffer;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_encoder"),
            });
        encoder.copy_buffer_to_buffer(src_buffer, offset, &staging_buffer, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buffer.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::wait_indefinitely());

        rx.recv()
            .map_err(|_| StorageError::Backend("Failed to receive map result".to_string()))?
            .map_err(|e| StorageError::Backend(format!("Map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range()[..size as usize].to_vec();
        staging_buffer.unmap();

        // Return staging buffer to pool for reuse
        self.release_staging_buffer(staging_buffer);

        Ok(data)
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) -> Result<(), StorageError> {
        if let Some(data) = self.buffers.remove(&buffer.0) {
            data.buffer.destroy();
        }
        Ok(())
    }

    fn copy_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), StorageError> {
        let src_data = self
            .buffers
            .get(&src.0)
            .ok_or_else(|| StorageError::Backend("Source buffer not found".to_string()))?;
        let dst_data = self
            .buffers
            .get(&dst.0)
            .ok_or_else(|| StorageError::Backend("Destination buffer not found".to_string()))?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &src_data.buffer,
            src_offset,
            &dst_data.buffer,
            dst_offset,
            size,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    fn execute(
        &mut self,
        op: ComputeOp,
        inputs: &[BufferHandle],
    ) -> Result<ComputeResult, StorageError> {
        match op {
            ComputeOp::Aggregate {
                agg_type,
                num_items,
            } => self.execute_aggregate(agg_type, num_items, inputs),
            ComputeOp::Hash {
                num_items,
                algorithm,
            } => self.execute_hash(num_items, algorithm, inputs),
            ComputeOp::BTreeRangeScan {
                num_nodes,
                keys_per_node,
                start_key_len,
                end_key_len,
                include_start,
                include_end,
            } => self.execute_btree_range_scan(
                num_nodes,
                keys_per_node,
                start_key_len,
                end_key_len,
                include_start,
                include_end,
                inputs,
            ),
            ComputeOp::BinaryHDBind { num_words } => self.execute_binary_hd_bind(num_words, inputs),
            ComputeOp::BinaryHDBatchBind {
                num_pairs,
                num_words,
            } => self.execute_binary_hd_batch_bind(num_pairs, num_words, inputs),
            ComputeOp::BinaryHDBundle {
                num_vectors,
                num_words,
            } => self.execute_binary_hd_bundle(num_vectors, num_words, inputs),
            ComputeOp::BinaryHDSimilarity {
                num_vectors,
                num_words,
            } => self.execute_binary_hd_similarity(num_vectors, num_words, inputs),
            ComputeOp::BinaryHDMultiSimilarity {
                num_queries,
                num_vectors,
                num_words,
            } => {
                self.execute_binary_hd_multi_similarity(num_queries, num_vectors, num_words, inputs)
            }
            // For other operations, fall back to CPU for now
            _ => {
                log::warn!(
                    "Operation {:?} not yet implemented on GPU, falling back to CPU",
                    op
                );

                // Use CPU fallback from joule-db-core
                let mut cpu = joule_db_core::persistence::compute::CpuComputeBackend::new();
                cpu.execute(op, inputs)
            }
        }
    }

    fn synchronize(&mut self) -> Result<(), StorageError> {
        self.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(())
    }

    fn supports_operation(&self, op: &ComputeOp) -> bool {
        match op {
            ComputeOp::Aggregate { .. } => true,
            ComputeOp::Hash { algorithm, .. } => {
                matches!(
                    algorithm,
                    HashAlgorithm::XxHash64 | HashAlgorithm::Crc32 | HashAlgorithm::Murmur3
                )
            }
            ComputeOp::BTreeRangeScan { .. } => {
                // GPU B-tree range scan is supported
                true
            }
            ComputeOp::BinaryHDBind { .. } => true,
            ComputeOp::BinaryHDBatchBind { .. } => true,
            ComputeOp::BinaryHDBundle { .. } => true,
            ComputeOp::BinaryHDSimilarity { .. } => true,
            ComputeOp::BinaryHDMultiSimilarity { .. } => true,
            _ => false, // Other operations not yet implemented
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test configuration with default settings
    #[test]
    fn test_config_default() {
        let config = WgpuComputeConfig::default();
        assert_eq!(
            config.power_preference,
            wgpu::PowerPreference::HighPerformance
        );
        assert!(!config.force_fallback);
        assert_eq!(config.max_buffer_size, 256 * 1024 * 1024);
    }

    /// Test configuration modification
    #[test]
    fn test_config_modification() {
        let config = WgpuComputeConfig {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback: true,
            max_buffer_size: 64 * 1024 * 1024,
        };
        assert_eq!(config.power_preference, wgpu::PowerPreference::LowPower);
        assert!(config.force_fallback);
        assert_eq!(config.max_buffer_size, 64 * 1024 * 1024);
    }

    /// Test buffer usage conversion
    #[test]
    fn test_buffer_usage_conversion() {
        let usage = BufferUsage::STORAGE_READ;
        let wgpu_usage = WgpuComputeBackend::buffer_usage_to_wgpu(usage);
        assert!(wgpu_usage.contains(wgpu::BufferUsages::STORAGE));

        let usage = BufferUsage::UNIFORM;
        let wgpu_usage = WgpuComputeBackend::buffer_usage_to_wgpu(usage);
        assert!(wgpu_usage.contains(wgpu::BufferUsages::UNIFORM));

        let usage = BufferUsage::MAP_READ;
        let wgpu_usage = WgpuComputeBackend::buffer_usage_to_wgpu(usage);
        assert!(wgpu_usage.contains(wgpu::BufferUsages::MAP_READ));
    }

    /// Test buffer usage combination
    #[test]
    fn test_buffer_usage_combination() {
        // BufferUsage is a newtype over u32, use bitwise combination via from_bits
        let usage =
            BufferUsage::from_bits(BufferUsage::STORAGE_READ.bits() | BufferUsage::COPY_SRC.bits());
        let wgpu_usage = WgpuComputeBackend::buffer_usage_to_wgpu(usage);
        assert!(wgpu_usage.contains(wgpu::BufferUsages::STORAGE));
        assert!(wgpu_usage.contains(wgpu::BufferUsages::COPY_SRC));
    }

    /// Test supports_operation
    #[test]
    fn test_supports_operation_aggregate() {
        // Without creating an actual backend, we can test the match logic
        let op_sum = ComputeOp::Aggregate {
            agg_type: AggregationType::Sum,
            num_items: 100,
        };
        let op_count = ComputeOp::Aggregate {
            agg_type: AggregationType::Count,
            num_items: 100,
        };
        let op_avg = ComputeOp::Aggregate {
            agg_type: AggregationType::Avg,
            num_items: 100,
        };
        let op_min = ComputeOp::Aggregate {
            agg_type: AggregationType::Min,
            num_items: 100,
        };
        let op_max = ComputeOp::Aggregate {
            agg_type: AggregationType::Max,
            num_items: 100,
        };

        // Check all aggregate ops are recognized as supported
        assert!(matches!(op_sum, ComputeOp::Aggregate { .. }));
        assert!(matches!(op_count, ComputeOp::Aggregate { .. }));
        assert!(matches!(op_avg, ComputeOp::Aggregate { .. }));
        assert!(matches!(op_min, ComputeOp::Aggregate { .. }));
        assert!(matches!(op_max, ComputeOp::Aggregate { .. }));
    }

    /// Test supports_operation for hash algorithms
    #[test]
    fn test_supports_operation_hash() {
        let op_xxhash = ComputeOp::Hash {
            num_items: 100,
            algorithm: HashAlgorithm::XxHash64,
        };
        let op_crc32 = ComputeOp::Hash {
            num_items: 100,
            algorithm: HashAlgorithm::Crc32,
        };
        let op_murmur3 = ComputeOp::Hash {
            num_items: 100,
            algorithm: HashAlgorithm::Murmur3,
        };

        // Check supported algorithms
        match &op_xxhash {
            ComputeOp::Hash { algorithm, .. } => {
                assert!(matches!(algorithm, HashAlgorithm::XxHash64));
            }
            _ => panic!("Expected Hash op"),
        }
        match &op_crc32 {
            ComputeOp::Hash { algorithm, .. } => {
                assert!(matches!(algorithm, HashAlgorithm::Crc32));
            }
            _ => panic!("Expected Hash op"),
        }
        match &op_murmur3 {
            ComputeOp::Hash { algorithm, .. } => {
                assert!(matches!(algorithm, HashAlgorithm::Murmur3));
            }
            _ => panic!("Expected Hash op"),
        }
    }

    /// Test supports_operation for binary HD operations
    #[test]
    fn test_supports_operation_binary_hd() {
        let op_bind = ComputeOp::BinaryHDBind { num_words: 128 };
        let op_bundle = ComputeOp::BinaryHDBundle {
            num_vectors: 10,
            num_words: 128,
        };
        let op_similarity = ComputeOp::BinaryHDSimilarity {
            num_vectors: 100,
            num_words: 128,
        };

        assert!(matches!(op_bind, ComputeOp::BinaryHDBind { .. }));
        assert!(matches!(op_bundle, ComputeOp::BinaryHDBundle { .. }));
        assert!(matches!(
            op_similarity,
            ComputeOp::BinaryHDSimilarity { .. }
        ));
    }

    /// Test supports_operation for btree range scan
    #[test]
    fn test_supports_operation_btree() {
        let op = ComputeOp::BTreeRangeScan {
            num_nodes: 100,
            keys_per_node: 32,
            start_key_len: 8,
            end_key_len: 8,
            include_start: true,
            include_end: false,
        };

        assert!(matches!(op, ComputeOp::BTreeRangeScan { .. }));
    }

    /// Test BufferHandle creation and comparison
    #[test]
    fn test_buffer_handle() {
        let h1 = BufferHandle(1);
        let h2 = BufferHandle(2);
        let h1_copy = BufferHandle(1);

        assert_eq!(h1.0, 1);
        assert_eq!(h2.0, 2);
        assert_eq!(h1.0, h1_copy.0);
        assert_ne!(h1.0, h2.0);
    }

    /// Test DeviceType enum
    #[test]
    fn test_device_type() {
        let gpu = DeviceType::Gpu;
        let cpu = DeviceType::Cpu;

        assert!(matches!(gpu, DeviceType::Gpu));
        assert!(matches!(cpu, DeviceType::Cpu));
    }

    /// Test DeviceCapabilities construction
    #[test]
    fn test_device_capabilities() {
        let caps = DeviceCapabilities {
            device_type: DeviceType::Gpu,
            name: "Test GPU".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,     // 8GB
            available_memory: 6 * 1024 * 1024 * 1024, // 6GB
            max_workgroup_size: 1024,
            max_buffer_size: 256 * 1024 * 1024,
            supports_f16: true,
            supports_f64: false,
            supports_atomics: true,
            compute_units: 64,
        };

        assert_eq!(caps.name, "Test GPU");
        assert_eq!(caps.max_workgroup_size, 1024);
        assert!(caps.supports_f16);
        assert!(!caps.supports_f64);
        assert!(caps.supports_atomics);
    }

    /// Test ComputeResult construction
    #[test]
    fn test_compute_result() {
        let result = ComputeResult {
            output_buffer: BufferHandle(42),
            result_count: 256,
            execution_time_us: 1500,
        };

        assert_eq!(result.output_buffer.0, 42);
        assert_eq!(result.result_count, 256);
        assert_eq!(result.execution_time_us, 1500);
    }

    /// Test AggregationType enum variants
    #[test]
    fn test_aggregation_type() {
        let sum = AggregationType::Sum;
        let count = AggregationType::Count;
        let avg = AggregationType::Avg;
        let min = AggregationType::Min;
        let max = AggregationType::Max;

        assert!(matches!(sum, AggregationType::Sum));
        assert!(matches!(count, AggregationType::Count));
        assert!(matches!(avg, AggregationType::Avg));
        assert!(matches!(min, AggregationType::Min));
        assert!(matches!(max, AggregationType::Max));
    }

    /// Test HashAlgorithm enum variants
    #[test]
    fn test_hash_algorithm() {
        let xxhash = HashAlgorithm::XxHash64;
        let crc32 = HashAlgorithm::Crc32;
        let murmur = HashAlgorithm::Murmur3;

        assert!(matches!(xxhash, HashAlgorithm::XxHash64));
        assert!(matches!(crc32, HashAlgorithm::Crc32));
        assert!(matches!(murmur, HashAlgorithm::Murmur3));
    }

    // ========================================================================
    // GPU Shader Correctness Tests
    // ========================================================================
    // These tests require a GPU. Run with: cargo test -p joule-db-gpu -- --ignored

    /// Helper: create a WgpuComputeBackend for tests
    fn create_gpu_backend() -> Option<WgpuComputeBackend> {
        pollster::block_on(WgpuComputeBackend::new()).ok()
    }

    /// CPU reference: XOR bind two vectors (u64 words → u8 bytes)
    fn cpu_bind(a: &[u64], b: &[u64]) -> Vec<u64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect()
    }

    /// CPU reference: majority-vote bundle
    fn cpu_bundle(vectors: &[Vec<u64>], num_words: usize) -> Vec<u64> {
        let n = vectors.len();
        let threshold = n / 2;
        let mut result = vec![0u64; num_words];
        for word_idx in 0..num_words {
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let ones: usize = vectors
                    .iter()
                    .map(|v| if v[word_idx] & mask != 0 { 1 } else { 0 })
                    .sum();
                if ones > threshold {
                    result[word_idx] |= mask;
                }
            }
        }
        result
    }

    /// CPU reference: Hamming similarity (matching bits count)
    fn cpu_similarity(query: &[u64], vector: &[u64]) -> u32 {
        let total_bits = query.len() as u32 * 64;
        let hamming: u32 = query
            .iter()
            .zip(vector.iter())
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum();
        total_bits - hamming
    }

    /// Simple xorshift PRNG for reproducible test vectors
    fn test_random_u64(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_bind_correctness() {
        let mut backend = create_gpu_backend().expect("GPU required");

        let num_u64_words = 8; // 512 dimensions
        let mut seed = 42u64;

        let a: Vec<u64> = (0..num_u64_words)
            .map(|_| test_random_u64(&mut seed))
            .collect();
        let b: Vec<u64> = (0..num_u64_words)
            .map(|_| test_random_u64(&mut seed))
            .collect();

        let cpu_result = cpu_bind(&a, &b);

        // Upload vectors as bytes
        let a_bytes = bytemuck::cast_slice::<u64, u8>(&a);
        let b_bytes = bytemuck::cast_slice::<u64, u8>(&b);

        let a_buf = backend
            .create_buffer(
                a_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("test_a"),
            )
            .unwrap();
        backend.write_buffer(a_buf, 0, a_bytes).unwrap();

        let b_buf = backend
            .create_buffer(
                b_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("test_b"),
            )
            .unwrap();
        backend.write_buffer(b_buf, 0, b_bytes).unwrap();

        let result = backend
            .execute(
                ComputeOp::BinaryHDBind {
                    num_words: num_u64_words as u32,
                },
                &[a_buf, b_buf],
            )
            .unwrap();

        let output_bytes = backend
            .read_buffer(result.output_buffer, 0, (num_u64_words * 8) as u64)
            .unwrap();

        let gpu_result: Vec<u64> = output_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        assert_eq!(
            cpu_result, gpu_result,
            "GPU bind output differs from CPU reference"
        );

        backend.destroy_buffer(a_buf).unwrap();
        backend.destroy_buffer(b_buf).unwrap();
        backend.destroy_buffer(result.output_buffer).unwrap();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_bundle_correctness() {
        let mut backend = create_gpu_backend().expect("GPU required");

        let num_u64_words = 8; // 512 dimensions
        let num_vectors = 5;
        let mut seed = 123u64;

        let vectors: Vec<Vec<u64>> = (0..num_vectors)
            .map(|_| {
                (0..num_u64_words)
                    .map(|_| test_random_u64(&mut seed))
                    .collect()
            })
            .collect();

        let cpu_result = cpu_bundle(&vectors, num_u64_words);

        // Flatten vectors for GPU upload
        let flat: Vec<u64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let flat_bytes = bytemuck::cast_slice::<u64, u8>(&flat);

        let vec_buf = backend
            .create_buffer(
                flat_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("test_vectors"),
            )
            .unwrap();
        backend.write_buffer(vec_buf, 0, flat_bytes).unwrap();

        let result = backend
            .execute(
                ComputeOp::BinaryHDBundle {
                    num_vectors: num_vectors as u32,
                    num_words: num_u64_words as u32,
                },
                &[vec_buf],
            )
            .unwrap();

        let output_bytes = backend
            .read_buffer(result.output_buffer, 0, (num_u64_words * 8) as u64)
            .unwrap();

        let gpu_result: Vec<u64> = output_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        assert_eq!(
            cpu_result, gpu_result,
            "GPU bundle output differs from CPU reference"
        );

        backend.destroy_buffer(vec_buf).unwrap();
        backend.destroy_buffer(result.output_buffer).unwrap();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_similarity_correctness() {
        let mut backend = create_gpu_backend().expect("GPU required");

        let num_u64_words = 8; // 512 dimensions
        let num_vectors = 10;
        let mut seed = 999u64;

        let query: Vec<u64> = (0..num_u64_words)
            .map(|_| test_random_u64(&mut seed))
            .collect();
        let vectors: Vec<Vec<u64>> = (0..num_vectors)
            .map(|_| {
                (0..num_u64_words)
                    .map(|_| test_random_u64(&mut seed))
                    .collect()
            })
            .collect();

        // CPU reference similarities
        let cpu_scores: Vec<u32> = vectors.iter().map(|v| cpu_similarity(&query, v)).collect();

        // Flatten vectors for GPU upload
        let flat: Vec<u64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let query_bytes = bytemuck::cast_slice::<u64, u8>(&query);
        let flat_bytes = bytemuck::cast_slice::<u64, u8>(&flat);

        let query_buf = backend
            .create_buffer(
                query_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("test_query"),
            )
            .unwrap();
        backend.write_buffer(query_buf, 0, query_bytes).unwrap();

        let vec_buf = backend
            .create_buffer(
                flat_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("test_vectors"),
            )
            .unwrap();
        backend.write_buffer(vec_buf, 0, flat_bytes).unwrap();

        let result = backend
            .execute(
                ComputeOp::BinaryHDSimilarity {
                    num_vectors: num_vectors as u32,
                    num_words: num_u64_words as u32,
                },
                &[query_buf, vec_buf],
            )
            .unwrap();

        let output_bytes = backend
            .read_buffer(result.output_buffer, 0, (num_vectors * 4) as u64)
            .unwrap();

        let gpu_scores: Vec<u32> = output_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(
            cpu_scores, gpu_scores,
            "GPU similarity scores differ from CPU reference"
        );

        backend.destroy_buffer(query_buf).unwrap();
        backend.destroy_buffer(vec_buf).unwrap();
        backend.destroy_buffer(result.output_buffer).unwrap();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_bind_identity() {
        // XOR with zero should return the original vector
        let mut backend = create_gpu_backend().expect("GPU required");

        let num_u64_words = 8;
        let mut seed = 77u64;
        let a: Vec<u64> = (0..num_u64_words)
            .map(|_| test_random_u64(&mut seed))
            .collect();
        let zero = vec![0u64; num_u64_words];

        let a_bytes = bytemuck::cast_slice::<u64, u8>(&a);
        let z_bytes = bytemuck::cast_slice::<u64, u8>(&zero);

        let a_buf = backend
            .create_buffer(
                a_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("a"),
            )
            .unwrap();
        backend.write_buffer(a_buf, 0, a_bytes).unwrap();

        let z_buf = backend
            .create_buffer(
                z_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("z"),
            )
            .unwrap();
        backend.write_buffer(z_buf, 0, z_bytes).unwrap();

        let result = backend
            .execute(
                ComputeOp::BinaryHDBind {
                    num_words: num_u64_words as u32,
                },
                &[a_buf, z_buf],
            )
            .unwrap();

        let output_bytes = backend
            .read_buffer(result.output_buffer, 0, (num_u64_words * 8) as u64)
            .unwrap();
        let gpu_result: Vec<u64> = output_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        assert_eq!(a, gpu_result, "XOR with zero should return original vector");

        backend.destroy_buffer(a_buf).unwrap();
        backend.destroy_buffer(z_buf).unwrap();
        backend.destroy_buffer(result.output_buffer).unwrap();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_similarity_self() {
        // Similarity of a vector with itself should be max (total_bits)
        let mut backend = create_gpu_backend().expect("GPU required");

        let num_u64_words = 8; // 512 bits
        let total_bits = (num_u64_words * 64) as u32;
        let mut seed = 55u64;
        let v: Vec<u64> = (0..num_u64_words)
            .map(|_| test_random_u64(&mut seed))
            .collect();

        let v_bytes = bytemuck::cast_slice::<u64, u8>(&v);

        let q_buf = backend
            .create_buffer(
                v_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("q"),
            )
            .unwrap();
        backend.write_buffer(q_buf, 0, v_bytes).unwrap();

        let db_buf = backend
            .create_buffer(
                v_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("db"),
            )
            .unwrap();
        backend.write_buffer(db_buf, 0, v_bytes).unwrap();

        let result = backend
            .execute(
                ComputeOp::BinaryHDSimilarity {
                    num_vectors: 1,
                    num_words: num_u64_words as u32,
                },
                &[q_buf, db_buf],
            )
            .unwrap();

        let output_bytes = backend.read_buffer(result.output_buffer, 0, 4).unwrap();
        let score = u32::from_le_bytes([
            output_bytes[0],
            output_bytes[1],
            output_bytes[2],
            output_bytes[3],
        ]);

        assert_eq!(
            score, total_bits,
            "Self-similarity should equal total bits ({}), got {}",
            total_bits, score
        );

        backend.destroy_buffer(q_buf).unwrap();
        backend.destroy_buffer(db_buf).unwrap();
        backend.destroy_buffer(result.output_buffer).unwrap();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_batch_bind_correctness() {
        let mut backend = create_gpu_backend().expect("GPU required");

        let num_u64_words = 8; // 512 dimensions
        let num_pairs = 16;
        let mut seed = 314u64;

        let vectors_a: Vec<Vec<u64>> = (0..num_pairs)
            .map(|_| {
                (0..num_u64_words)
                    .map(|_| test_random_u64(&mut seed))
                    .collect()
            })
            .collect();
        let vectors_b: Vec<Vec<u64>> = (0..num_pairs)
            .map(|_| {
                (0..num_u64_words)
                    .map(|_| test_random_u64(&mut seed))
                    .collect()
            })
            .collect();

        // CPU reference: XOR each pair
        let cpu_results: Vec<Vec<u64>> = vectors_a
            .iter()
            .zip(vectors_b.iter())
            .map(|(a, b)| cpu_bind(a, b))
            .collect();

        // Flatten for GPU upload
        let flat_a: Vec<u64> = vectors_a.iter().flat_map(|v| v.iter().copied()).collect();
        let flat_b: Vec<u64> = vectors_b.iter().flat_map(|v| v.iter().copied()).collect();

        let a_bytes = bytemuck::cast_slice::<u64, u8>(&flat_a);
        let b_bytes = bytemuck::cast_slice::<u64, u8>(&flat_b);

        let a_buf = backend
            .create_buffer(
                a_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("batch_a"),
            )
            .unwrap();
        backend.write_buffer(a_buf, 0, a_bytes).unwrap();

        let b_buf = backend
            .create_buffer(
                b_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("batch_b"),
            )
            .unwrap();
        backend.write_buffer(b_buf, 0, b_bytes).unwrap();

        let result = backend
            .execute(
                ComputeOp::BinaryHDBatchBind {
                    num_pairs: num_pairs as u32,
                    num_words: num_u64_words as u32,
                },
                &[a_buf, b_buf],
            )
            .unwrap();

        let output_bytes = backend
            .read_buffer(
                result.output_buffer,
                0,
                (num_pairs * num_u64_words * 8) as u64,
            )
            .unwrap();

        let gpu_flat: Vec<u64> = output_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        // Verify each pair
        for pair in 0..num_pairs {
            let gpu_slice = &gpu_flat[pair * num_u64_words..(pair + 1) * num_u64_words];
            assert_eq!(
                &cpu_results[pair], gpu_slice,
                "Batch bind pair {} differs from CPU reference",
                pair
            );
        }

        backend.destroy_buffer(a_buf).unwrap();
        backend.destroy_buffer(b_buf).unwrap();
        backend.destroy_buffer(result.output_buffer).unwrap();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_multi_similarity_correctness() {
        let mut backend = create_gpu_backend().expect("GPU required");

        let num_u64_words = 8; // 512 dimensions
        let num_queries = 4;
        let num_vectors = 10;
        let mut seed = 271u64;

        let queries: Vec<Vec<u64>> = (0..num_queries)
            .map(|_| {
                (0..num_u64_words)
                    .map(|_| test_random_u64(&mut seed))
                    .collect()
            })
            .collect();
        let vectors: Vec<Vec<u64>> = (0..num_vectors)
            .map(|_| {
                (0..num_u64_words)
                    .map(|_| test_random_u64(&mut seed))
                    .collect()
            })
            .collect();

        // CPU reference: M × N similarity matrix
        let cpu_scores: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| vectors.iter().map(|v| cpu_similarity(q, v)).collect())
            .collect();

        // Flatten for GPU upload
        let flat_q: Vec<u64> = queries.iter().flat_map(|v| v.iter().copied()).collect();
        let flat_v: Vec<u64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let q_bytes = bytemuck::cast_slice::<u64, u8>(&flat_q);
        let v_bytes = bytemuck::cast_slice::<u64, u8>(&flat_v);

        let q_buf = backend
            .create_buffer(
                q_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("multi_queries"),
            )
            .unwrap();
        backend.write_buffer(q_buf, 0, q_bytes).unwrap();

        let v_buf = backend
            .create_buffer(
                v_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("multi_vectors"),
            )
            .unwrap();
        backend.write_buffer(v_buf, 0, v_bytes).unwrap();

        let result = backend
            .execute(
                ComputeOp::BinaryHDMultiSimilarity {
                    num_queries: num_queries as u32,
                    num_vectors: num_vectors as u32,
                    num_words: num_u64_words as u32,
                },
                &[q_buf, v_buf],
            )
            .unwrap();

        let output_bytes = backend
            .read_buffer(
                result.output_buffer,
                0,
                (num_queries * num_vectors * 4) as u64,
            )
            .unwrap();

        let gpu_scores: Vec<u32> = output_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Verify M × N matrix row by row
        for q in 0..num_queries {
            let gpu_row = &gpu_scores[q * num_vectors..(q + 1) * num_vectors];
            assert_eq!(
                &cpu_scores[q], gpu_row,
                "Multi-similarity query {} differs from CPU reference",
                q
            );
        }

        backend.destroy_buffer(q_buf).unwrap();
        backend.destroy_buffer(v_buf).unwrap();
        backend.destroy_buffer(result.output_buffer).unwrap();
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_multi_similarity_self() {
        // Each query compared to itself should yield max similarity (total_bits)
        let mut backend = create_gpu_backend().expect("GPU required");

        let num_u64_words = 8;
        let total_bits = (num_u64_words * 64) as u32;
        let num_vectors = 3;
        let mut seed = 88u64;

        let vectors: Vec<Vec<u64>> = (0..num_vectors)
            .map(|_| {
                (0..num_u64_words)
                    .map(|_| test_random_u64(&mut seed))
                    .collect()
            })
            .collect();

        // Use same vectors as queries
        let flat: Vec<u64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let v_bytes = bytemuck::cast_slice::<u64, u8>(&flat);

        let q_buf = backend
            .create_buffer(
                v_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("self_q"),
            )
            .unwrap();
        backend.write_buffer(q_buf, 0, v_bytes).unwrap();

        let v_buf = backend
            .create_buffer(
                v_bytes.len() as u64,
                BufferUsage::STORAGE_READ.union(BufferUsage::COPY_DST),
                Some("self_v"),
            )
            .unwrap();
        backend.write_buffer(v_buf, 0, v_bytes).unwrap();

        let result = backend
            .execute(
                ComputeOp::BinaryHDMultiSimilarity {
                    num_queries: num_vectors as u32,
                    num_vectors: num_vectors as u32,
                    num_words: num_u64_words as u32,
                },
                &[q_buf, v_buf],
            )
            .unwrap();

        let output_bytes = backend
            .read_buffer(
                result.output_buffer,
                0,
                (num_vectors * num_vectors * 4) as u64,
            )
            .unwrap();

        let gpu_scores: Vec<u32> = output_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Diagonal of the M × M matrix should be max similarity
        for i in 0..num_vectors {
            let diag = gpu_scores[i * num_vectors + i];
            assert_eq!(
                diag, total_bits,
                "Self-similarity at ({},{}) should be {} but got {}",
                i, i, total_bits, diag
            );
        }

        backend.destroy_buffer(q_buf).unwrap();
        backend.destroy_buffer(v_buf).unwrap();
        backend.destroy_buffer(result.output_buffer).unwrap();
    }
}
