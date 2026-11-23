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

        Self {
            device,
            queue,
            adapter_name,
            backend_api,
            p1_layout,
            p1_pipeline,
            p2_layout,
            p2_pipeline,
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

        let v_buf = create_ro_buf("lb_v_transposed", bytemuck::cast_slice(&weights.v_transposed_u32));
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
}
