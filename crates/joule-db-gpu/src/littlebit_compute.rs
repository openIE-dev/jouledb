//! Hardware GPU `wgpu` (Metal / Vulkan / DX12 / WebGPU) Compute Shader Kernel for
//! Samsung `LittleBit-2` Sub-1-Bit Binary Bitplanes ⊕ Sakana `Transformer²` `SVF`.
//!
//! Executes a fused two-pass GPU command buffer with zero intermediate CPU roundtrip:
//! 1. `project_v_popcnt`: Coalesced CWI `FastLanes`-transposed `u32` bitplane `XNOR + countOneBits()`
//!    (`POPCNT`) inner product into latent rank space, scaled in-flight by Sakana `SVF` singular
//!    expert multipliers.
//! 2. `project_u`: Parallel binary bitplane expansion across output rows scaled by Dual-SVID
//!    `row_scale`.

use std::sync::{Arc, mpsc};
use std::time::Instant;

/// WGSL shader implementing both `project_v_popcnt` (Pass 1) and `project_u` (Pass 2).
pub const LITTLEBIT_XNOR_POPCNT_WGSL: &str = r#"
struct LittleBitParams {
    rows: u32,
    cols: u32,
    rank: u32,
    col_words: u32,
    rank_words: u32,
    tail_mask: u32,
    mean_act_scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> v_transposed: array<u32>;
@group(0) @binding(1) var<storage, read> x_bits: array<u32>;
@group(0) @binding(2) var<storage, read> eff_sigma: array<f32>;
@group(0) @binding(3) var<storage, read_write> latent_out: array<f32>;
@group(0) @binding(4) var<uniform> params_p1: LittleBitParams;

@compute @workgroup_size(64)
fn project_v_popcnt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params_p1.rank {
        return;
    }
    let base = k * params_p1.col_words;
    var matches: u32 = 0u;
    for (var w = 0u; w < params_p1.col_words; w = w + 1u) {
        let mask = select(0xFFFFFFFFu, params_p1.tail_mask, (w + 1u) == params_p1.col_words);
        let xnor_val = (~(x_bits[w] ^ v_transposed[base + w])) & mask;
        matches = matches + countOneBits(xnor_val);
    }
    let bipolar_dot = f32(i32(2u * matches) - i32(params_p1.cols));
    latent_out[k] = bipolar_dot * params_p1.mean_act_scale * eff_sigma[k];
}
"#;

/// WGSL shader for Pass 2 (`U_bin * latent * row_scale`).
pub const LITTLEBIT_EXPAND_U_WGSL: &str = r#"
struct LittleBitParams {
    rows: u32,
    cols: u32,
    rank: u32,
    col_words: u32,
    rank_words: u32,
    tail_mask: u32,
    mean_act_scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> u_rows: array<u32>;
@group(0) @binding(1) var<storage, read> latent_in: array<f32>;
@group(0) @binding(2) var<storage, read> row_scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params_p2: LittleBitParams;

@compute @workgroup_size(64)
fn project_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= params_p2.rows {
        return;
    }
    let row_base = r * params_p2.rank_words;
    var acc: f32 = 0.0;
    for (var k = 0u; k < params_p2.rank; k = k + 1u) {
        let word = u_rows[row_base + (k / 32u)];
        let bit = (word >> (k % 32u)) & 1u;
        let sgn = select(-1.0, 1.0, bit == 1u);
        acc = acc + sgn * latent_in[k];
    }
    output[r] = acc * row_scale[r];
}
"#;

/// Wave 17 Batched Exact Dual-SVID Pass 1 (`[batch, cols] -> [batch, rank]`) over resident GPU bitplanes.
pub const LITTLEBIT_BATCH_EXACT_P1_WGSL: &str = r#"
struct BatchLittleBitParams {
    rows: u32,
    cols: u32,
    rank: u32,
    col_words: u32,
    rank_words: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> v_transposed: array<u32>;
@group(0) @binding(1) var<storage, read> col_scale: array<f32>;
@group(0) @binding(2) var<storage, read> batch_x: array<f32>;
@group(0) @binding(3) var<storage, read> eff_sigma: array<f32>;
@group(0) @binding(4) var<storage, read_write> batch_latent: array<f32>;
@group(0) @binding(5) var<uniform> bparams: BatchLittleBitParams;

@compute @workgroup_size(64, 1, 1)
fn project_v_batch_exact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let b = gid.y;
    if k >= bparams.rank || b >= bparams.batch_size {
        return;
    }
    let v_base = k * bparams.col_words;
    let x_base = b * bparams.cols;
    var acc: f32 = 0.0;
    for (var w = 0u; w < bparams.col_words; w = w + 1u) {
        let word = v_transposed[v_base + w];
        let j0 = w * 32u;
        let lim = min(32u, bparams.cols - j0);
        for (var bit_idx = 0u; bit_idx < lim; bit_idx = bit_idx + 1u) {
            let j = j0 + bit_idx;
            let xs = batch_x[x_base + j] * col_scale[j];
            let sgn = select(-1.0, 1.0, ((word >> bit_idx) & 1u) == 1u);
            acc = acc + sgn * xs;
        }
    }
    batch_latent[b * bparams.rank + k] = acc * eff_sigma[k];
}
"#;

/// Wave 17 Batched Exact Dual-SVID Pass 2 (`[batch, rank] -> [batch, rows]`) over resident GPU bitplanes.
pub const LITTLEBIT_BATCH_EXACT_P2_WGSL: &str = r#"
struct BatchLittleBitParams {
    rows: u32,
    cols: u32,
    rank: u32,
    col_words: u32,
    rank_words: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> u_rows: array<u32>;
@group(0) @binding(1) var<storage, read> row_scale: array<f32>;
@group(0) @binding(2) var<storage, read> batch_latent: array<f32>;
@group(0) @binding(3) var<storage, read_write> batch_out: array<f32>;
@group(0) @binding(4) var<uniform> bparams: BatchLittleBitParams;

@compute @workgroup_size(64, 1, 1)
fn project_u_batch_exact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let b = gid.y;
    if r >= bparams.rows || b >= bparams.batch_size {
        return;
    }
    let u_base = r * bparams.rank_words;
    let lat_base = b * bparams.rank;
    var acc: f32 = 0.0;
    for (var w = 0u; w < bparams.rank_words; w = w + 1u) {
        let word = u_rows[u_base + w];
        let k0 = w * 32u;
        let lim = min(32u, bparams.rank - k0);
        for (var bit_idx = 0u; bit_idx < lim; bit_idx = bit_idx + 1u) {
            let sgn = select(-1.0, 1.0, ((word >> bit_idx) & 1u) == 1u);
            acc = acc + sgn * batch_latent[lat_base + k0 + bit_idx];
        }
    }
    batch_out[b * bparams.rows + r] = acc * row_scale[r];
}
"#;

/// Wave 17 Fused Causal Grouped-Query Attention (`GQA`) WGSL Compute Shader over `(query_len, num_q_heads)`.
pub const CAUSAL_GQA_WGSL: &str = r#"
struct CausalGqaParams {
    q_len: u32,
    total_kv_len: u32,
    start_pos: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_group_size: u32,
    inv_sqrt_hd: f32,
};

@group(0) @binding(0) var<storage, read> q_batch: array<f32>;
@group(0) @binding(1) var<storage, read> k_all: array<f32>;
@group(0) @binding(2) var<storage, read> v_all: array<f32>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;
@group(0) @binding(4) var<uniform> gparams: CausalGqaParams;

@compute @workgroup_size(16, 4, 1)
fn causal_gqa_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qh = gid.x;
    let qi = gid.y;
    if qh >= gparams.num_q_heads || qi >= gparams.q_len {
        return;
    }
    let kvh = min(qh / max(gparams.kv_group_size, 1u), gparams.num_kv_heads - 1u);
    let q_dim = gparams.num_q_heads * gparams.head_dim;
    let kv_dim = gparams.num_kv_heads * gparams.head_dim;
    let q_off = qi * q_dim + qh * gparams.head_dim;
    let causal_limit = min(gparams.start_pos + qi + 1u, gparams.total_kv_len);

    var max_s: f32 = -1e30;
    for (var s = 0u; s < causal_limit; s = s + 1u) {
        let k_off = s * kv_dim + kvh * gparams.head_dim;
        var dot: f32 = 0.0;
        for (var d = 0u; d < gparams.head_dim; d = d + 1u) {
            dot = dot + q_batch[q_off + d] * k_all[k_off + d];
        }
        let sc = dot * gparams.inv_sqrt_hd;
        if sc > max_s {
            max_s = sc;
        }
    }

    var denom: f32 = 0.0;
    for (var d = 0u; d < gparams.head_dim; d = d + 1u) {
        attn_out[q_off + d] = 0.0;
    }
    for (var s = 0u; s < causal_limit; s = s + 1u) {
        let kv_off = s * kv_dim + kvh * gparams.head_dim;
        var dot: f32 = 0.0;
        for (var d = 0u; d < gparams.head_dim; d = d + 1u) {
            dot = dot + q_batch[q_off + d] * k_all[kv_off + d];
        }
        let w = exp(dot * gparams.inv_sqrt_hd - max_s);
        denom = denom + w;
        for (var d = 0u; d < gparams.head_dim; d = d + 1u) {
            attn_out[q_off + d] = attn_out[q_off + d] + w * v_all[kv_off + d];
        }
    }
    let inv_d = 1.0 / max(denom, 1e-12);
    for (var d = 0u; d < gparams.head_dim; d = d + 1u) {
        attn_out[q_off + d] = attn_out[q_off + d] * inv_d;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLittleBitParams {
    rows: u32,
    cols: u32,
    rank: u32,
    col_words: u32,
    rank_words: u32,
    tail_mask: u32,
    mean_act_scale: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBatchLittleBitParams {
    rows: u32,
    cols: u32,
    rank: u32,
    col_words: u32,
    rank_words: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCausalGqaParams {
    q_len: u32,
    total_kv_len: u32,
    start_pos: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_group_size: u32,
    inv_sqrt_hd: f32,
}

/// Persistent GPU-resident `wgpu::Buffer` handles for zero-upload multi-token `LittleBit-2` passes.
#[derive(Debug, Clone)]
pub struct ResidentGpuLittleBitWeights {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub col_u32_words: usize,
    pub rank_u32_words: usize,
    pub v_buf: wgpu::Buffer,
    pub u_buf: wgpu::Buffer,
    pub col_scale_buf: wgpu::Buffer,
    pub row_scale_buf: wgpu::Buffer,
}

/// Execution telemetry returned by [`LittleBitGpuKernel::forward_xnor_popcnt`].
#[derive(Debug, Clone)]
pub struct LittleBitGpuReceipt {
    pub adapter_name: String,
    pub backend_api: String,
    pub output: Vec<f32>,
    pub elapsed_us: u64,
    pub estimated_joules_uj: f64,
}

/// Pre-transposed CWI `FastLanes` `u32` layout for coalesced GPU `XNOR + POPCNT` reads.
#[derive(Debug, Clone)]
pub struct PackedGpuLittleBitWeights {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub col_u32_words: usize,
    pub rank_u32_words: usize,
    pub v_transposed_u32: Vec<u32>,
    pub u_rows_u32: Vec<u32>,
    pub row_scale: Vec<f32>,
    pub col_scale: Vec<f32>,
}

impl PackedGpuLittleBitWeights {
    /// Convert `u64` row/col bitplanes (`LittleBitSVFTensor` layout) into coalesced `u32` GPU words.
    pub fn from_u64_bitplanes(
        rows: usize,
        cols: usize,
        rank: usize,
        rank_u64_words: usize,
        u_bits: &[u64],
        v_bits: &[u64],
        row_scale: Vec<f32>,
        col_scale: Vec<f32>,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rank == 0 {
            return Err("rows, cols, and rank must be >= 1".to_string());
        }
        if u_bits.len() != rows * rank_u64_words || v_bits.len() != cols * rank_u64_words {
            return Err("u_bits / v_bits slice length mismatch".to_string());
        }
        let col_u32_words = cols.div_ceil(32);
        let rank_u32_words = rank.div_ceil(32);

        // Transpose V_bin[j, k] -> V_T[k, j] packed in u32 words so every GPU thread k reads contiguous u32s
        let mut v_transposed_u32 = vec![0u32; rank * col_u32_words];
        for k in 0..rank {
            let kw64 = k / 64;
            let kmask64 = 1u64 << (k % 64);
            let out_base = k * col_u32_words;
            for j in 0..cols {
                if (v_bits[j * rank_u64_words + kw64] & kmask64) != 0 {
                    v_transposed_u32[out_base + (j / 32)] |= 1u32 << (j % 32);
                }
            }
        }

        // Pack U_bin[i, k] into u32 row words
        let mut u_rows_u32 = vec![0u32; rows * rank_u32_words];
        for i in 0..rows {
            let out_base = i * rank_u32_words;
            for k in 0..rank {
                let kw64 = k / 64;
                let kmask64 = 1u64 << (k % 64);
                if (u_bits[i * rank_u64_words + kw64] & kmask64) != 0 {
                    u_rows_u32[out_base + (k / 32)] |= 1u32 << (k % 32);
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            rank,
            col_u32_words,
            rank_u32_words,
            v_transposed_u32,
            u_rows_u32,
            row_scale,
            col_scale,
        })
    }
}

/// Hardware `wgpu` compute pipeline executing `LittleBit-2 ⊕ Sakana SVF` binary matvecs on GPU.
pub struct LittleBitGpuKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_name: String,
    backend_api: String,
    p1_layout: wgpu::BindGroupLayout,
    p1_pipeline: wgpu::ComputePipeline,
    p2_layout: wgpu::BindGroupLayout,
    p2_pipeline: wgpu::ComputePipeline,
    batch_p1_layout: wgpu::BindGroupLayout,
    batch_p1_pipeline: wgpu::ComputePipeline,
    batch_p2_layout: wgpu::BindGroupLayout,
    batch_p2_pipeline: wgpu::ComputePipeline,
    gqa_layout: wgpu::BindGroupLayout,
    gqa_pipeline: wgpu::ComputePipeline,
}

impl LittleBitGpuKernel {
    /// Initialize a hardware `wgpu` context (Metal / Vulkan / DX12 / WebGPU) and compile both WGSL passes.
    pub async fn new_hardware() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
            memory_budget_thresholds: Default::default(),
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no wgpu adapter available: {e:?}"))?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_api = format!("{:?}", info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("JouleDB LittleBit XNOR-POPCNT GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| format!("failed to request wgpu device: {e}"))?;

        Ok(Self::from_device(
            Arc::new(device),
            Arc::new(queue),
            adapter_name,
            backend_api,
        ))
    }

    /// Construct pipelines on an existing `wgpu::Device` & `Queue`.
    pub fn from_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter_name: String,
        backend_api: String,
    ) -> Self {
        let make_layout = |label: &str| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        };

        let p1_layout = make_layout("littlebit_p1_layout");
        let p1_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("littlebit_p1_shader"),
            source: wgpu::ShaderSource::Wgsl(LITTLEBIT_XNOR_POPCNT_WGSL.into()),
        });
        let p1_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("littlebit_p1_pipe_layout"),
            bind_group_layouts: &[Some(&p1_layout)],
            immediate_size: 0,
        });
        let p1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("littlebit_p1_pipeline"),
            layout: Some(&p1_pipe_layout),
            module: &p1_shader,
            entry_point: Some("project_v_popcnt"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let p2_layout = make_layout("littlebit_p2_layout");
        let p2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("littlebit_p2_shader"),
            source: wgpu::ShaderSource::Wgsl(LITTLEBIT_EXPAND_U_WGSL.into()),
        });
        let p2_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("littlebit_p2_pipe_layout"),
            bind_group_layouts: &[Some(&p2_layout)],
            immediate_size: 0,
        });
        let p2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("littlebit_p2_pipeline"),
            layout: Some(&p2_pipe_layout),
            module: &p2_shader,
            entry_point: Some("project_u"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let batch_p1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("littlebit_batch_p1_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let batch_p1_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("littlebit_batch_p1_shader"),
            source: wgpu::ShaderSource::Wgsl(LITTLEBIT_BATCH_EXACT_P1_WGSL.into()),
        });
        let batch_p1_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("littlebit_batch_p1_pipe_layout"),
            bind_group_layouts: &[Some(&batch_p1_layout)],
            immediate_size: 0,
        });
        let batch_p1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("littlebit_batch_p1_pipeline"),
            layout: Some(&batch_p1_pipe_layout),
            module: &batch_p1_shader,
            entry_point: Some("project_v_batch_exact"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let batch_p2_layout = make_layout("littlebit_batch_p2_layout");
        let batch_p2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("littlebit_batch_p2_shader"),
            source: wgpu::ShaderSource::Wgsl(LITTLEBIT_BATCH_EXACT_P2_WGSL.into()),
        });
        let batch_p2_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("littlebit_batch_p2_pipe_layout"),
            bind_group_layouts: &[Some(&batch_p2_layout)],
            immediate_size: 0,
        });
        let batch_p2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("littlebit_batch_p2_pipeline"),
            layout: Some(&batch_p2_pipe_layout),
            module: &batch_p2_shader,
            entry_point: Some("project_u_batch_exact"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let gqa_layout = make_layout("causal_gqa_layout");
        let gqa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("causal_gqa_shader"),
            source: wgpu::ShaderSource::Wgsl(CAUSAL_GQA_WGSL.into()),
        });
        let gqa_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("causal_gqa_pipe_layout"),
            bind_group_layouts: &[Some(&gqa_layout)],
            immediate_size: 0,
        });
        let gqa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("causal_gqa_pipeline"),
            layout: Some(&gqa_pipe_layout),
            module: &gqa_shader,
            entry_point: Some("causal_gqa_kernel"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            device,
            queue,
            adapter_name,
            backend_api,
            p1_layout,
            p1_pipeline,
            p2_layout,
            p2_pipeline,
            batch_p1_layout,
            batch_p1_pipeline,
            batch_p2_layout,
            batch_p2_pipeline,
            gqa_layout,
            gqa_pipeline,
        }
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn backend_api(&self) -> &str {
        &self.backend_api
    }

    /// Execute the two-pass `XNOR + countOneBits()` (`POPCNT`) binary latent forward pass on GPU!
    pub fn forward_xnor_popcnt(
        &self,
        weights: &PackedGpuLittleBitWeights,
        input: &[f32],
        eff_sigma: &[f32],
    ) -> Result<LittleBitGpuReceipt, String> {
        if input.len() != weights.cols {
            return Err(format!(
                "input len {} != cols {}",
                input.len(),
                weights.cols
            ));
        }
        if eff_sigma.len() != weights.rank {
            return Err(format!(
                "eff_sigma len {} != rank {}",
                eff_sigma.len(),
                weights.rank
            ));
        }

        let start = Instant::now();

        // Binarize pre-scaled input x[j] * col_scale[j] into u32 words
        let mut x_bits = vec![0u32; weights.col_u32_words];
        let mut act_l1_sum = 0.0f32;
        for (j, &xj) in input.iter().enumerate() {
            let prod = xj * weights.col_scale[j];
            act_l1_sum += prod.abs();
            if prod >= 0.0 {
                x_bits[j / 32] |= 1u32 << (j % 32);
            }
        }
        let mean_act_scale = act_l1_sum / (weights.cols.max(1) as f32);
        let tail_bits = weights.cols % 32;
        let tail_mask = if tail_bits == 0 {
            0xFFFF_FFFFu32
        } else {
            (1u32 << tail_bits) - 1
        };

        let params = GpuLittleBitParams {
            rows: weights.rows as u32,
            cols: weights.cols as u32,
            rank: weights.rank as u32,
            col_words: weights.col_u32_words as u32,
            rank_words: weights.rank_u32_words as u32,
            tail_mask,
            mean_act_scale,
            _pad: 0,
        };

        let create_ro_buf = |label: &str, bytes: &[u8]| {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes.len() as u64).max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytes);
            buf
        };

        let v_buf = create_ro_buf(
            "lb_v_transposed",
            bytemuck::cast_slice(&weights.v_transposed_u32),
        );
        let x_buf = create_ro_buf("lb_x_bits", bytemuck::cast_slice(&x_bits));
        let sig_buf = create_ro_buf("lb_eff_sigma", bytemuck::cast_slice(eff_sigma));
        let u_buf = create_ro_buf("lb_u_rows", bytemuck::cast_slice(&weights.u_rows_u32));
        let rscale_buf = create_ro_buf("lb_row_scale", bytemuck::cast_slice(&weights.row_scale));

        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lb_params"),
            size: std::mem::size_of::<GpuLittleBitParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let latent_bytes = ((weights.rank * 4) as u64).max(4);
        let latent_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lb_latent"),
            size: latent_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let out_bytes = ((weights.rows * 4) as u64).max(4);
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lb_output"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lb_staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg_p1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lb_bg_p1"),
            layout: &self.p1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sig_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: latent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let bg_p2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lb_bg_p2"),
            layout: &self.p2_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: latent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rscale_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lb_two_pass_encoder"),
            });
        {
            let mut pass1 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lb_pass1_v_popcnt"),
                timestamp_writes: None,
            });
            pass1.set_pipeline(&self.p1_pipeline);
            pass1.set_bind_group(0, &bg_p1, &[]);
            pass1.dispatch_workgroups((weights.rank as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass2 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lb_pass2_u_expand"),
                timestamp_writes: None,
            });
            pass2.set_pipeline(&self.p2_pipeline);
            pass2.set_bind_group(0, &bg_p2, &[]);
            pass2.dispatch_workgroups((weights.rows as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        rx.recv()
            .map_err(|e| format!("GPU staging map channel closed: {e}"))?
            .map_err(|e| format!("GPU staging map error: {e:?}"))?;

        let mapped = slice.get_mapped_range();
        let output: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped)[..weights.rows].to_vec();
        drop(mapped);
        staging.unmap();

        let elapsed_us = start.elapsed().as_micros() as u64;
        // ~0.18 pJ per binary XNOR-POPCNT op + dispatch floor
        let bit_ops = ((weights.cols + weights.rows) * weights.rank) as f64;
        let estimated_joules_uj = 1.2 + bit_ops * 1.8e-7;

        Ok(LittleBitGpuReceipt {
            adapter_name: self.adapter_name.clone(),
            backend_api: self.backend_api.clone(),
            output,
            elapsed_us,
            estimated_joules_uj,
        })
    }

    /// Upload `PackedGpuLittleBitWeights` once into persistent GPU storage buffers so subsequent
    /// batched prefill/decode calls perform **zero** weight uploads!
    pub fn upload_resident_weights(
        &self,
        weights: &PackedGpuLittleBitWeights,
    ) -> ResidentGpuLittleBitWeights {
        let upload = |label: &str, bytes: &[u8]| {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes.len() as u64).max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytes);
            buf
        };
        ResidentGpuLittleBitWeights {
            rows: weights.rows,
            cols: weights.cols,
            rank: weights.rank,
            col_u32_words: weights.col_u32_words,
            rank_u32_words: weights.rank_u32_words,
            v_buf: upload("lb_res_v", bytemuck::cast_slice(&weights.v_transposed_u32)),
            u_buf: upload("lb_res_u", bytemuck::cast_slice(&weights.u_rows_u32)),
            col_scale_buf: upload("lb_res_cscale", bytemuck::cast_slice(&weights.col_scale)),
            row_scale_buf: upload("lb_res_rscale", bytemuck::cast_slice(&weights.row_scale)),
        }
    }

    /// Execute exact batched Dual-SVID `f32 x 1-bit` multi-token projection (`[batch_size, cols] -> [batch_size, rows]`)
    /// in a single two-pass GPU command buffer over resident GPU bitplane buffers!
    pub fn forward_batched_resident(
        &self,
        resident: &ResidentGpuLittleBitWeights,
        batch_inputs: &[f32],
        batch_size: usize,
        eff_sigma: &[f32],
    ) -> Result<LittleBitGpuReceipt, String> {
        if batch_size == 0 {
            return Ok(LittleBitGpuReceipt {
                adapter_name: self.adapter_name.clone(),
                backend_api: self.backend_api.clone(),
                output: Vec::new(),
                elapsed_us: 0,
                estimated_joules_uj: 0.0,
            });
        }
        if batch_inputs.len() != batch_size * resident.cols {
            return Err(format!(
                "batch_inputs len {} != batch_size ({batch_size}) * cols ({})",
                batch_inputs.len(),
                resident.cols
            ));
        }
        if eff_sigma.len() != resident.rank {
            return Err(format!(
                "eff_sigma len {} != rank {}",
                eff_sigma.len(),
                resident.rank
            ));
        }

        let start = Instant::now();
        let bparams = GpuBatchLittleBitParams {
            rows: resident.rows as u32,
            cols: resident.cols as u32,
            rank: resident.rank as u32,
            col_words: resident.col_u32_words as u32,
            rank_words: resident.rank_u32_words as u32,
            batch_size: batch_size as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let create_ro = |label: &str, bytes: &[u8]| {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes.len() as u64).max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytes);
            buf
        };
        let x_buf = create_ro("lb_batch_x", bytemuck::cast_slice(batch_inputs));
        let sig_buf = create_ro("lb_batch_sig", bytemuck::cast_slice(eff_sigma));
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lb_batch_params"),
            size: std::mem::size_of::<GpuBatchLittleBitParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&bparams));

        let latent_bytes = ((batch_size * resident.rank * 4) as u64).max(4);
        let latent_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lb_batch_latent"),
            size: latent_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let total_out_elems = batch_size * resident.rows;
        let out_bytes = ((total_out_elems * 4) as u64).max(4);
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lb_batch_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lb_batch_staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lb_batch_bg1"),
            layout: &self.batch_p1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resident.v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resident.col_scale_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sig_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: latent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lb_batch_bg2"),
            layout: &self.batch_p2_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resident.u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resident.row_scale_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: latent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lb_batch_encoder"),
            });
        {
            let mut p1 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lb_batch_p1"),
                timestamp_writes: None,
            });
            p1.set_pipeline(&self.batch_p1_pipeline);
            p1.set_bind_group(0, &bg1, &[]);
            p1.dispatch_workgroups((resident.rank as u32).div_ceil(64), batch_size as u32, 1);
        }
        {
            let mut p2 = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lb_batch_p2"),
                timestamp_writes: None,
            });
            p2.set_pipeline(&self.batch_p2_pipeline);
            p2.set_bind_group(0, &bg2, &[]);
            p2.dispatch_workgroups((resident.rows as u32).div_ceil(64), batch_size as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|e| format!("GPU batch staging channel closed: {e}"))?
            .map_err(|e| format!("GPU batch staging map error: {e:?}"))?;

        let mapped = slice.get_mapped_range();
        let output: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped)[..total_out_elems].to_vec();
        drop(mapped);
        staging.unmap();

        let elapsed_us = start.elapsed().as_micros() as u64;
        let ops = (batch_size * (resident.cols + resident.rows) * resident.rank) as f64;
        let estimated_joules_uj = 1.4 + ops * 3.5e-7;
        Ok(LittleBitGpuReceipt {
            adapter_name: self.adapter_name.clone(),
            backend_api: self.backend_api.clone(),
            output,
            elapsed_us,
            estimated_joules_uj,
        })
    }

    /// Execute fused causal Grouped-Query Attention (`GQA`) across `(q_len, num_q_heads)` on GPU!
    #[allow(clippy::too_many_arguments)]
    pub fn forward_causal_gqa(
        &self,
        q_batch: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        q_len: usize,
        total_kv_len: usize,
        start_pos: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<LittleBitGpuReceipt, String> {
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        if q_len == 0 || num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err("invalid GQA dimensions".to_string());
        }
        if q_batch.len() != q_len * q_dim
            || k_all.len() != total_kv_len * kv_dim
            || v_all.len() != total_kv_len * kv_dim
        {
            return Err("q/k/v slice length mismatch in forward_causal_gqa".to_string());
        }

        let start = Instant::now();
        let gparams = GpuCausalGqaParams {
            q_len: q_len as u32,
            total_kv_len: total_kv_len as u32,
            start_pos: start_pos as u32,
            num_q_heads: num_q_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            kv_group_size: (num_q_heads / num_kv_heads.max(1)).max(1) as u32,
            inv_sqrt_hd: 1.0 / (head_dim as f32).sqrt(),
        };

        let create_ro = |label: &str, bytes: &[u8]| {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes.len() as u64).max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytes);
            buf
        };
        let q_buf = create_ro("gqa_q", bytemuck::cast_slice(q_batch));
        let k_buf = create_ro("gqa_k", bytemuck::cast_slice(k_all));
        let v_buf = create_ro("gqa_v", bytemuck::cast_slice(v_all));
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gqa_params"),
            size: std::mem::size_of::<GpuCausalGqaParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&gparams));

        let out_elems = q_len * q_dim;
        let out_bytes = ((out_elems * 4) as u64).max(4);
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gqa_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gqa_staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gqa_bg"),
            layout: &self.gqa_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gqa_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gqa_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gqa_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                (num_q_heads as u32).div_ceil(16),
                (q_len as u32).div_ceil(4),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|e| format!("GPU GQA staging channel closed: {e}"))?
            .map_err(|e| format!("GPU GQA staging map error: {e:?}"))?;

        let mapped = slice.get_mapped_range();
        let output: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped)[..out_elems].to_vec();
        drop(mapped);
        staging.unmap();

        let elapsed_us = start.elapsed().as_micros() as u64;
        let attn_ops = (q_len * total_kv_len * num_q_heads * head_dim * 2) as f64;
        let estimated_joules_uj = 1.0 + attn_ops * 4.0e-7;
        Ok(LittleBitGpuReceipt {
            adapter_name: self.adapter_name.clone(),
            backend_api: self.backend_api.clone(),
            output,
            elapsed_us,
            estimated_joules_uj,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_metal_wgpu_littlebit_xnor_popcnt_matches_cpu_reference() {
        let kernel = match pollster::block_on(LittleBitGpuKernel::new_hardware()) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping hardware GPU test (no adapter): {e}");
                return;
            }
        };
        println!(
            "Running LittleBit XNOR+POPCNT WGSL shader on GPU: {} ({})",
            kernel.adapter_name(),
            kernel.backend_api()
        );

        let rows = 160usize;
        let cols = 100usize; // Intentionally non-multiple of 32 & 64 to exercise tail_mask!
        let rank = 12usize;
        let rank_u64_words = rank.div_ceil(64);

        let mut u_bits = vec![0u64; rows * rank_u64_words];
        let mut v_bits = vec![0u64; cols * rank_u64_words];
        for i in 0..rows {
            for k in 0..rank {
                if (i + k * 3) % 2 == 0 {
                    u_bits[i * rank_u64_words + (k / 64)] |= 1u64 << (k % 64);
                }
            }
        }
        for j in 0..cols {
            for k in 0..rank {
                if (j * 5 + k) % 3 != 0 {
                    v_bits[j * rank_u64_words + (k / 64)] |= 1u64 << (k % 64);
                }
            }
        }
        let row_scale: Vec<f32> = (0..rows).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let col_scale: Vec<f32> = (0..cols).map(|j| 0.8 + (j as f32) * 0.005).collect();
        let eff_sigma: Vec<f32> = (0..rank).map(|k| 1.2 / ((k + 1) as f32)).collect();
        let input: Vec<f32> = (0..cols).map(|j| ((j as f32) * 0.17).sin()).collect();

        let packed = PackedGpuLittleBitWeights::from_u64_bitplanes(
            rows,
            cols,
            rank,
            rank_u64_words,
            &u_bits,
            &v_bits,
            row_scale.clone(),
            col_scale.clone(),
        )
        .expect("pack");

        let gpu_res = kernel
            .forward_xnor_popcnt(&packed, &input, &eff_sigma)
            .expect("gpu forward");
        assert_eq!(gpu_res.output.len(), rows);

        // Compute exact CPU reference for `XNOR + POPCNT`
        let mut act_sum = 0.0f32;
        for j in 0..cols {
            act_sum += (input[j] * col_scale[j]).abs();
        }
        let mean_act = act_sum / (cols as f32);
        let mut cpu_latent = vec![0.0f32; rank];
        for k in 0..rank {
            let mut matches = 0i32;
            for j in 0..cols {
                let x_pos = (input[j] * col_scale[j]) >= 0.0;
                let v_pos = ((v_bits[j * rank_u64_words + (k / 64)] >> (k % 64)) & 1) == 1;
                if x_pos == v_pos {
                    matches += 1;
                }
            }
            let bipolar = (2 * matches - (cols as i32)) as f32;
            cpu_latent[k] = bipolar * mean_act * eff_sigma[k];
        }
        for i in 0..rows {
            let mut acc = 0.0f32;
            for k in 0..rank {
                let u_pos = ((u_bits[i * rank_u64_words + (k / 64)] >> (k % 64)) & 1) == 1;
                acc += if u_pos { cpu_latent[k] } else { -cpu_latent[k] };
            }
            let cpu_val = acc * row_scale[i];
            let diff = (gpu_res.output[i] - cpu_val).abs();
            assert!(
                diff < 1e-3,
                "Row {i} GPU ({}) != CPU ({}) diff={diff}",
                gpu_res.output[i],
                cpu_val
            );
        }
    }

    #[test]
    fn test_hardware_batched_resident_and_causal_gqa_parity() {
        let kernel = match pollster::block_on(LittleBitGpuKernel::new_hardware()) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping hardware GPU test (no adapter): {e}");
                return;
            }
        };

        let rows = 96usize;
        let cols = 72usize;
        let rank = 20usize;
        let rank_u64 = rank.div_ceil(64);
        let batch_sz = 6usize;

        let mut u_bits = vec![0u64; rows * rank_u64];
        let mut v_bits = vec![0u64; cols * rank_u64];
        for i in 0..rows {
            for k in 0..rank {
                if (i * 7 + k * 3) % 2 == 0 {
                    u_bits[i * rank_u64 + (k / 64)] |= 1u64 << (k % 64);
                }
            }
        }
        for j in 0..cols {
            for k in 0..rank {
                if (j * 5 + k * 11) % 3 != 0 {
                    v_bits[j * rank_u64 + (k / 64)] |= 1u64 << (k % 64);
                }
            }
        }
        let row_scale: Vec<f32> = (0..rows).map(|i| 0.25 + (i as f32) * 0.008).collect();
        let col_scale: Vec<f32> = (0..cols).map(|j| 0.4 + (j as f32) * 0.006).collect();
        let eff_sigma: Vec<f32> = (0..rank).map(|k| 0.9 / ((k + 1) as f32)).collect();
        let batch_x: Vec<f32> = (0..batch_sz * cols)
            .map(|idx| ((idx as f32) * 0.13).cos())
            .collect();

        let packed = PackedGpuLittleBitWeights::from_u64_bitplanes(
            rows,
            cols,
            rank,
            rank_u64,
            &u_bits,
            &v_bits,
            row_scale.clone(),
            col_scale.clone(),
        )
        .expect("pack");
        let resident = kernel.upload_resident_weights(&packed);
        let gpu_batch = kernel
            .forward_batched_resident(&resident, &batch_x, batch_sz, &eff_sigma)
            .expect("gpu batch");
        assert_eq!(gpu_batch.output.len(), batch_sz * rows);

        for b in 0..batch_sz {
            let xb = &batch_x[b * cols..(b + 1) * cols];
            let mut lat = vec![0.0f32; rank];
            for k in 0..rank {
                let mut acc = 0.0f32;
                for j in 0..cols {
                    let vp = ((v_bits[j * rank_u64 + (k / 64)] >> (k % 64)) & 1) == 1;
                    let xs = xb[j] * col_scale[j];
                    acc += if vp { xs } else { -xs };
                }
                lat[k] = acc * eff_sigma[k];
            }
            for i in 0..rows {
                let mut acc = 0.0f32;
                for k in 0..rank {
                    let up = ((u_bits[i * rank_u64 + (k / 64)] >> (k % 64)) & 1) == 1;
                    acc += if up { lat[k] } else { -lat[k] };
                }
                let cpu_y = acc * row_scale[i];
                let gpu_y = gpu_batch.output[b * rows + i];
                assert!(
                    (gpu_y - cpu_y).abs() < 2e-3,
                    "batch {b} row {i} gpu={gpu_y} cpu={cpu_y}"
                );
            }
        }

        // Verify fused Causal GQA WGSL kernel (`4` Q heads, `2` KV heads, `head_dim = 8`, `q_len = 3`)
        let q_len = 3usize;
        let total_kv = 5usize;
        let start_pos = 2usize;
        let nq = 4usize;
        let nkv = 2usize;
        let hd = 8usize;
        let q_batch: Vec<f32> = (0..q_len * nq * hd)
            .map(|i| ((i as f32) * 0.19).sin())
            .collect();
        let k_all: Vec<f32> = (0..total_kv * nkv * hd)
            .map(|i| ((i as f32) * 0.23).cos())
            .collect();
        let v_all: Vec<f32> = (0..total_kv * nkv * hd)
            .map(|i| ((i as f32) * 0.11).sin())
            .collect();
        let gqa_res = kernel
            .forward_causal_gqa(
                &q_batch, &k_all, &v_all, q_len, total_kv, start_pos, nq, nkv, hd,
            )
            .expect("gqa");
        assert_eq!(gqa_res.output.len(), q_len * nq * hd);
        let inv_sqrt = 1.0 / (hd as f32).sqrt();
        for qi in 0..q_len {
            let clim = (start_pos + qi + 1).min(total_kv);
            for qh in 0..nq {
                let kvh = qh / 2;
                let q_sl = &q_batch[(qi * nq + qh) * hd..(qi * nq + qh + 1) * hd];
                let mut scores = Vec::new();
                let mut mx = f32::NEG_INFINITY;
                for s in 0..clim {
                    let k_sl = &k_all[(s * nkv + kvh) * hd..(s * nkv + kvh + 1) * hd];
                    let dot: f32 = q_sl.iter().zip(k_sl.iter()).map(|(a, b)| a * b).sum();
                    let sc = dot * inv_sqrt;
                    if sc > mx {
                        mx = sc;
                    }
                    scores.push(sc);
                }
                let exps: Vec<f32> = scores.iter().map(|&sc| (sc - mx).exp()).collect();
                let denom: f32 = exps.iter().sum::<f32>().max(1e-12);
                for d in 0..hd {
                    let mut expected = 0.0f32;
                    for s in 0..clim {
                        expected += (exps[s] / denom) * v_all[(s * nkv + kvh) * hd + d];
                    }
                    let got = gqa_res.output[(qi * nq + qh) * hd + d];
                    assert!(
                        (got - expected).abs() < 1e-3,
                        "gqa mismatch qi={qi} qh={qh} d={d}: got={got} exp={expected}"
                    );
                }
            }
        }
    }
}
