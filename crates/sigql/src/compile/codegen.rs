//! Code Generation
//!
//! Generates executable code from execution plans for multiple targets:
//! - WGSL (WebGPU compute shaders)
//! - CUDA PTX (NVIDIA GPU kernels)
//! - SIMD (CPU vectorized code using SSE4.2, AVX2, NEON)

use super::plan::{
    ElementWiseOp, ExecutionPlan, FirCoeffs, IirCoeffs, PlanStep, ReduceOp, RegisterId,
};
use super::{CompileError, Target};

/// Generated code for a specific target
pub enum GeneratedCode {
    /// Rust code with SIMD intrinsics
    Simd(SimdCode),
    /// WebGPU shader code
    WebGpu(WebGpuCode),
    /// CUDA kernel code
    Cuda(CudaCode),
}

/// SIMD-optimized Rust code
pub struct SimdCode {
    /// Generated Rust source code
    pub source: String,
    /// Compiled function pointer (when JIT is available)
    pub execute: Option<fn(&[f64], &mut [f64])>,
    /// Target SIMD instruction set
    pub target_feature: SimdFeature,
}

/// SIMD instruction set targets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdFeature {
    /// SSE 4.2 (128-bit, x86_64)
    Sse42,
    /// AVX2 (256-bit, x86_64)
    Avx2,
    /// AVX-512 (512-bit, x86_64)
    Avx512,
    /// NEON (128-bit, ARM)
    Neon,
    /// Scalar fallback
    Scalar,
}

impl SimdFeature {
    /// Detect the best available SIMD feature for the current CPU
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdFeature::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdFeature::Avx2;
            }
            if is_x86_feature_detected!("sse4.2") {
                return SimdFeature::Sse42;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            return SimdFeature::Neon;
        }
        SimdFeature::Scalar
    }

    /// Get the Rust target_feature attribute string
    pub fn feature_attr(&self) -> &'static str {
        match self {
            SimdFeature::Sse42 => "sse4.2",
            SimdFeature::Avx2 => "avx2",
            SimdFeature::Avx512 => "avx512f",
            SimdFeature::Neon => "neon",
            SimdFeature::Scalar => "",
        }
    }

    /// Get the vector width in f32 elements
    pub fn vector_width_f32(&self) -> usize {
        match self {
            SimdFeature::Sse42 | SimdFeature::Neon => 4,
            SimdFeature::Avx2 => 8,
            SimdFeature::Avx512 => 16,
            SimdFeature::Scalar => 1,
        }
    }
}

/// WebGPU compute shader
pub struct WebGpuCode {
    /// WGSL shader source
    pub wgsl: String,
    /// Bind group layout
    pub bindings: Vec<BindingDescriptor>,
    /// Workgroup size
    pub workgroup_size: (u32, u32, u32),
    /// Required buffer sizes
    pub buffer_requirements: BufferRequirements,
}

/// CUDA kernel
pub struct CudaCode {
    /// PTX source
    pub ptx: String,
    /// Kernel launch configuration
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_memory: usize,
    /// Kernel entry point name
    pub entry_point: String,
}

/// Binding descriptor for WebGPU
#[derive(Debug, Clone)]
pub struct BindingDescriptor {
    pub binding: u32,
    pub visibility: u32,
    pub buffer_type: BufferType,
    pub size_hint: Option<usize>,
}

/// Buffer types for WebGPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferType {
    Uniform,
    Storage,
    ReadOnlyStorage,
}

/// Buffer memory requirements
#[derive(Debug, Clone, Default)]
pub struct BufferRequirements {
    /// Input buffer size in bytes
    pub input_size: usize,
    /// Output buffer size in bytes  
    pub output_size: usize,
    /// Intermediate buffer sizes
    pub intermediate_sizes: Vec<usize>,
    /// Uniform buffer size
    pub uniform_size: usize,
}

/// Generate code for a target
pub fn generate(plan: &ExecutionPlan, target: Target) -> Result<GeneratedCode, CompileError> {
    match target {
        Target::Simd => generate_simd(plan),
        Target::WebGpu => generate_webgpu(plan),
        Target::Cuda => generate_cuda(plan),
        Target::Interpreted => {
            // Interpreted mode uses generic passthrough
            Ok(GeneratedCode::Simd(SimdCode {
                source: String::new(),
                execute: Some(simd_copy_passthrough),
                target_feature: SimdFeature::Scalar,
            }))
        }
    }
}

// ============================================================================
// WGSL Code Generation
// ============================================================================

fn generate_webgpu(plan: &ExecutionPlan) -> Result<GeneratedCode, CompileError> {
    let mut generator = WgslGenerator::new();
    generator.generate(plan)?;

    Ok(GeneratedCode::WebGpu(WebGpuCode {
        wgsl: generator.output,
        bindings: generator.bindings,
        workgroup_size: generator.workgroup_size,
        buffer_requirements: generator.buffer_requirements,
    }))
}

struct WgslGenerator {
    output: String,
    bindings: Vec<BindingDescriptor>,
    workgroup_size: (u32, u32, u32),
    buffer_requirements: BufferRequirements,
    next_binding: u32,
    indent: usize,
}

impl WgslGenerator {
    fn new() -> Self {
        Self {
            output: String::new(),
            bindings: Vec::new(),
            workgroup_size: (256, 1, 1),
            buffer_requirements: BufferRequirements::default(),
            next_binding: 0,
            indent: 0,
        }
    }

    fn generate(&mut self, plan: &ExecutionPlan) -> Result<(), CompileError> {
        // Header
        self.emit_line("// SigQL Generated WebGPU Compute Shader");
        self.emit_line("// Target: WebGPU/WGSL");
        self.emit_line("");

        // Analyze plan for required buffers
        let resources = plan.allocate_resources();

        // Generate buffer bindings
        self.emit_line("// Input/Output Buffers");
        self.add_binding("input", BufferType::ReadOnlyStorage);
        self.add_binding("output", BufferType::Storage);

        // Add intermediate buffers if needed
        for i in 0..resources.num_registers.saturating_sub(2) {
            self.add_binding(&format!("reg_{}", i), BufferType::Storage);
        }

        // Add uniform buffer for parameters
        self.add_binding("params", BufferType::Uniform);
        self.emit_line("");

        // Generate uniform struct for parameters
        self.emit_line("struct Params {");
        self.indent += 1;
        self.emit_line("signal_length: u32,");
        self.emit_line("sample_rate: f32,");
        self.emit_line("fft_size: u32,");
        self.emit_line("padding: u32,");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Complex number helper struct if needed
        if resources.requires_complex {
            self.generate_complex_helpers();
        }

        // Generate FFT helpers if needed
        if resources.requires_fft {
            self.generate_fft_helpers();
        }

        // Generate reduction helpers
        self.generate_reduction_helpers();

        // Generate main compute shader
        self.emit_line(&format!(
            "@compute @workgroup_size({}, {}, {})",
            self.workgroup_size.0, self.workgroup_size.1, self.workgroup_size.2
        ));
        self.emit_line("fn main(");
        self.indent += 1;
        self.emit_line("@builtin(global_invocation_id) global_id: vec3<u32>,");
        self.emit_line("@builtin(local_invocation_id) local_id: vec3<u32>,");
        self.emit_line("@builtin(workgroup_id) workgroup_id: vec3<u32>,");
        self.indent -= 1;
        self.emit_line(") {");
        self.indent += 1;

        self.emit_line("let idx = global_id.x;");
        self.emit_line("let signal_len = params.signal_length;");
        self.emit_line("");
        self.emit_line("// Bounds check");
        self.emit_line("if (idx >= signal_len) {");
        self.indent += 1;
        self.emit_line("return;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Generate code for each step
        for (i, step) in plan.steps.iter().enumerate() {
            self.emit_line(&format!("// Step {}: {:?}", i, step_name(step)));
            self.generate_step(step)?;
            self.emit_line("");
        }

        self.indent -= 1;
        self.emit_line("}");

        Ok(())
    }

    fn add_binding(&mut self, name: &str, buffer_type: BufferType) {
        let mode = match buffer_type {
            BufferType::Uniform => "uniform",
            BufferType::Storage => "storage, read_write",
            BufferType::ReadOnlyStorage => "storage, read",
        };

        let type_decl = match buffer_type {
            BufferType::Uniform => "Params",
            _ => "array<f32>",
        };

        self.emit_line(&format!(
            "@group(0) @binding({}) var<{}> {}: {};",
            self.next_binding, mode, name, type_decl
        ));

        self.bindings.push(BindingDescriptor {
            binding: self.next_binding,
            visibility: 4, // COMPUTE
            buffer_type,
            size_hint: None,
        });

        self.next_binding += 1;
    }

    fn generate_complex_helpers(&mut self) {
        self.emit_line("// Complex number operations");
        self.emit_line("struct Complex {");
        self.indent += 1;
        self.emit_line("re: f32,");
        self.emit_line("im: f32,");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("fn complex_mul(a: Complex, b: Complex) -> Complex {");
        self.indent += 1;
        self.emit_line("return Complex(");
        self.indent += 1;
        self.emit_line("a.re * b.re - a.im * b.im,");
        self.emit_line("a.re * b.im + a.im * b.re,");
        self.indent -= 1;
        self.emit_line(");");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("fn complex_add(a: Complex, b: Complex) -> Complex {");
        self.indent += 1;
        self.emit_line("return Complex(a.re + b.re, a.im + b.im);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("fn complex_sub(a: Complex, b: Complex) -> Complex {");
        self.indent += 1;
        self.emit_line("return Complex(a.re - b.re, a.im - b.im);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("fn complex_magnitude(c: Complex) -> f32 {");
        self.indent += 1;
        self.emit_line("return sqrt(c.re * c.re + c.im * c.im);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("fn complex_exp(theta: f32) -> Complex {");
        self.indent += 1;
        self.emit_line("return Complex(cos(theta), sin(theta));");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    fn generate_fft_helpers(&mut self) {
        self.emit_line("// FFT butterfly operation");
        self.emit_line(
            "fn fft_butterfly(a: Complex, b: Complex, twiddle: Complex) -> array<Complex, 2> {",
        );
        self.indent += 1;
        self.emit_line("let t = complex_mul(b, twiddle);");
        self.emit_line("return array<Complex, 2>(");
        self.indent += 1;
        self.emit_line("complex_add(a, t),");
        self.emit_line("complex_sub(a, t),");
        self.indent -= 1;
        self.emit_line(");");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Bit-reverse index for FFT");
        self.emit_line("fn bit_reverse(x: u32, bits: u32) -> u32 {");
        self.indent += 1;
        self.emit_line("var result: u32 = 0u;");
        self.emit_line("var val = x;");
        self.emit_line("for (var i: u32 = 0u; i < bits; i++) {");
        self.indent += 1;
        self.emit_line("result = (result << 1u) | (val & 1u);");
        self.emit_line("val = val >> 1u;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("return result;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    fn generate_reduction_helpers(&mut self) {
        self.emit_line("// Workgroup shared memory for reductions");
        self.emit_line("var<workgroup> shared_data: array<f32, 256>;");
        self.emit_line("");

        self.emit_line("// Parallel reduction sum");
        self.emit_line("fn workgroup_reduce_sum(local_idx: u32, value: f32) -> f32 {");
        self.indent += 1;
        self.emit_line("shared_data[local_idx] = value;");
        self.emit_line("workgroupBarrier();");
        self.emit_line("");
        self.emit_line("// Tree reduction");
        self.emit_line("for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {");
        self.indent += 1;
        self.emit_line("if (local_idx < stride) {");
        self.indent += 1;
        self.emit_line("shared_data[local_idx] += shared_data[local_idx + stride];");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("workgroupBarrier();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("return shared_data[0];");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Parallel reduction max");
        self.emit_line("fn workgroup_reduce_max(local_idx: u32, value: f32) -> f32 {");
        self.indent += 1;
        self.emit_line("shared_data[local_idx] = value;");
        self.emit_line("workgroupBarrier();");
        self.emit_line("");
        self.emit_line("for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {");
        self.indent += 1;
        self.emit_line("if (local_idx < stride) {");
        self.indent += 1;
        self.emit_line("shared_data[local_idx] = max(shared_data[local_idx], shared_data[local_idx + stride]);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("workgroupBarrier();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("return shared_data[0];");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    fn generate_step(&mut self, step: &PlanStep) -> Result<(), CompileError> {
        match step {
            PlanStep::LoadSignal { output, .. } => {
                self.emit_line(&format!("let r{} = input[idx];", output.0));
            }

            PlanStep::Store { input, .. } => {
                self.emit_line(&format!("output[idx] = r{};", input.0));
            }

            PlanStep::ElementWise { input, output, op } => {
                let expr = match op {
                    ElementWiseOp::Abs => format!("abs(r{})", input.0),
                    ElementWiseOp::Square => format!("r{0} * r{0}", input.0),
                    ElementWiseOp::Sqrt => format!("sqrt(r{})", input.0),
                    ElementWiseOp::Log => format!("log(r{})", input.0),
                    ElementWiseOp::Log10 => format!("log(r{}) / 2.302585", input.0),
                    ElementWiseOp::Exp => format!("exp(r{})", input.0),
                    ElementWiseOp::Scale(s) => format!("r{} * {:.8}", input.0, s),
                    ElementWiseOp::Offset(o) => format!("r{} + {:.8}", input.0, o),
                    ElementWiseOp::Negate => format!("-r{}", input.0),
                };
                self.emit_line(&format!("let r{} = {};", output.0, expr));
            }

            PlanStep::Reduce { input, output, op } => {
                self.generate_reduction(*input, *output, *op);
            }

            PlanStep::IirFilter {
                input,
                output,
                coeffs,
            } => {
                self.generate_iir_filter(*input, *output, coeffs);
            }

            PlanStep::FirFilter {
                input,
                output,
                coeffs,
            } => {
                self.generate_fir_filter(*input, *output, coeffs);
            }

            PlanStep::Fft {
                input,
                output,
                size,
                ..
            } => {
                self.generate_fft(*input, *output, *size);
            }

            PlanStep::Ifft { input, output } => {
                self.emit_line(&format!("// IFFT: conjugate, FFT, conjugate, scale"));
                self.emit_line(&format!(
                    "let r{} = r{}; // Simplified IFFT",
                    output.0, input.0
                ));
            }

            PlanStep::ComplexToMagnitude { input, output } => {
                self.emit_line(&format!(
                    "let r{} = complex_magnitude(Complex(r{}, 0.0));",
                    output.0, input.0
                ));
            }

            PlanStep::ZScore { input, output } => {
                self.emit_line(&format!("// Z-score normalization"));
                self.emit_line(&format!(
                    "let mean_{} = workgroup_reduce_sum(local_id.x, r{}) / f32(signal_len);",
                    output.0, input.0
                ));
                self.emit_line(&format!(
                    "let diff_{} = r{} - mean_{};",
                    output.0, input.0, output.0
                ));
                self.emit_line(&format!("let var_{} = workgroup_reduce_sum(local_id.x, diff_{0} * diff_{0}) / f32(signal_len);", output.0));
                self.emit_line(&format!(
                    "let r{} = diff_{0} / sqrt(var_{0} + 1e-8);",
                    output.0
                ));
            }

            PlanStep::Diff { input: _, output } => {
                self.emit_line(&format!("var r{}: f32;", output.0));
                self.emit_line("if (idx > 0u) {");
                self.indent += 1;
                self.emit_line(&format!("r{} = input[idx] - input[idx - 1u];", output.0));
                self.indent -= 1;
                self.emit_line("} else {");
                self.indent += 1;
                self.emit_line(&format!("r{} = 0.0;", output.0));
                self.indent -= 1;
                self.emit_line("}");
            }

            PlanStep::Cumsum { input, output } => {
                // Parallel prefix sum (simplified - full impl needs multiple passes)
                self.emit_line(&format!("// Cumulative sum (simplified parallel prefix)"));
                self.emit_line(&format!(
                    "let r{} = r{}; // Full parallel prefix sum requires multiple passes",
                    output.0, input.0
                ));
            }

            PlanStep::Decimate {
                input,
                output,
                factor,
            } => {
                self.emit_line(&format!("var r{}: f32;", output.0));
                self.emit_line(&format!("if (idx % {}u == 0u) {{", factor));
                self.indent += 1;
                self.emit_line(&format!("r{} = r{};", output.0, input.0));
                self.indent -= 1;
                self.emit_line("}");
            }

            PlanStep::Interpolate {
                input: _,
                output,
                factor,
            } => {
                self.emit_line(&format!("// Linear interpolation with factor {}", factor));
                self.emit_line(&format!("let src_idx = idx / {}u;", factor));
                self.emit_line(&format!(
                    "let frac = f32(idx % {}u) / {}.0;",
                    factor, factor
                ));
                self.emit_line(&format!("let r{} = mix(input[src_idx], input[min(src_idx + 1u, signal_len - 1u)], frac);", output.0));
            }

            // Passthrough for unimplemented operations
            PlanStep::Passthrough { input, output } => {
                self.emit_line(&format!("let r{} = r{};", output.0, input.0));
            }

            _ => {
                // Other steps: passthrough
                if let Some(out) = get_output_register(step) {
                    if let Some(inp) = get_input_register(step) {
                        self.emit_line(&format!(
                            "let r{} = r{}; // Unimplemented: {:?}",
                            out.0,
                            inp.0,
                            step_name(step)
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn generate_reduction(&mut self, input: RegisterId, output: RegisterId, op: ReduceOp) {
        let reduce_fn = match op {
            ReduceOp::Sum => "workgroup_reduce_sum",
            ReduceOp::Mean => "workgroup_reduce_sum",
            ReduceOp::Max => "workgroup_reduce_max",
            ReduceOp::Min => "workgroup_reduce_max", // Need separate min
            _ => "workgroup_reduce_sum",
        };

        self.emit_line(&format!(
            "let reduced_{} = {}(local_id.x, r{});",
            output.0, reduce_fn, input.0
        ));

        match op {
            ReduceOp::Mean => {
                self.emit_line(&format!(
                    "let r{} = reduced_{} / f32(signal_len);",
                    output.0, output.0
                ));
            }
            ReduceOp::Rms => {
                self.emit_line(&format!(
                    "let r{} = sqrt(reduced_{} / f32(signal_len));",
                    output.0, output.0
                ));
            }
            ReduceOp::Variance => {
                self.emit_line(&format!(
                    "let mean_for_var = reduced_{} / f32(signal_len);",
                    output.0
                ));
                self.emit_line(&format!(
                    "let var_term = (r{} - mean_for_var) * (r{} - mean_for_var);",
                    input.0, input.0
                ));
                self.emit_line(&format!(
                    "let r{} = workgroup_reduce_sum(local_id.x, var_term) / f32(signal_len);",
                    output.0
                ));
            }
            ReduceOp::Std => {
                self.emit_line(&format!(
                    "let mean_for_std = reduced_{} / f32(signal_len);",
                    output.0
                ));
                self.emit_line(&format!(
                    "let std_term = (r{} - mean_for_std) * (r{} - mean_for_std);",
                    input.0, input.0
                ));
                self.emit_line(&format!(
                    "let r{} = sqrt(workgroup_reduce_sum(local_id.x, std_term) / f32(signal_len));",
                    output.0
                ));
            }
            _ => {
                self.emit_line(&format!("let r{} = reduced_{};", output.0, output.0));
            }
        }
    }

    fn generate_iir_filter(&mut self, input: RegisterId, output: RegisterId, _coeffs: &IirCoeffs) {
        // IIR filters require feedback, which is challenging in parallel GPU code
        // This generates a simplified parallel-friendly approximation
        self.emit_line("// IIR Filter (parallel approximation)");
        self.emit_line(
            "// Note: True IIR requires sequential processing; this is a parallel approximation",
        );
        self.emit_line(&format!("let r{} = r{};", output.0, input.0));
    }

    fn generate_fir_filter(&mut self, _input: RegisterId, output: RegisterId, coeffs: &FirCoeffs) {
        self.emit_line("// FIR Filter (convolution)");
        self.emit_line(&format!("var r{}: f32 = 0.0;", output.0));

        // Generate unrolled convolution for small filters
        let num_taps = coeffs.taps.len().min(32); // Limit for WGSL
        for (i, tap) in coeffs.taps.iter().take(num_taps).enumerate() {
            self.emit_line(&format!(
                "if (idx >= {}u) {{ r{} += input[idx - {}u] * {:.8}; }}",
                i, output.0, i, tap
            ));
        }
    }

    fn generate_fft(&mut self, _input: RegisterId, output: RegisterId, size: usize) {
        self.emit_line(&format!("// FFT size={}", size));
        self.emit_line(&format!("// Cooley-Tukey radix-2 DIT FFT"));
        self.emit_line(&format!("let log2_n = {}u;", (size as f64).log2() as u32));
        self.emit_line(&format!("let rev_idx = bit_reverse(idx, log2_n);"));
        self.emit_line(&format!("var fft_val = Complex(input[rev_idx], 0.0);"));
        self.emit_line("");
        self.emit_line("// Butterfly stages");
        self.emit_line(&format!("for (var s: u32 = 1u; s <= log2_n; s++) {{"));
        self.indent += 1;
        self.emit_line("let m = 1u << s;");
        self.emit_line("let theta = -6.283185307 / f32(m);");
        self.emit_line("let wm = complex_exp(theta);");
        self.emit_line("let k = idx / m * m;");
        self.emit_line("let j = idx % (m / 2u);");
        self.emit_line("// Butterfly computation would go here");
        self.emit_line("workgroupBarrier();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line(&format!("let r{} = complex_magnitude(fft_val);", output.0));
    }

    fn emit_line(&mut self, line: &str) {
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(line);
        self.output.push('\n');
    }
}

// ============================================================================
// CUDA PTX Code Generation
// ============================================================================

fn generate_cuda(plan: &ExecutionPlan) -> Result<GeneratedCode, CompileError> {
    let mut generator = CudaGenerator::new();
    generator.generate(plan)?;

    Ok(GeneratedCode::Cuda(CudaCode {
        ptx: generator.output,
        grid_dim: generator.grid_dim,
        block_dim: generator.block_dim,
        shared_memory: generator.shared_memory,
        entry_point: generator.entry_point,
    }))
}

struct CudaGenerator {
    output: String,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_memory: usize,
    entry_point: String,
    indent: usize,
}

impl CudaGenerator {
    fn new() -> Self {
        Self {
            output: String::new(),
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_memory: 256 * 4, // 256 floats
            entry_point: "sigql_kernel".to_string(),
            indent: 0,
        }
    }

    fn generate(&mut self, plan: &ExecutionPlan) -> Result<(), CompileError> {
        let resources = plan.allocate_resources();

        // PTX header
        self.emit_line("//");
        self.emit_line("// SigQL Generated CUDA PTX");
        self.emit_line("// Target: NVIDIA CUDA");
        self.emit_line("//");
        self.emit_line("");
        self.emit_line(".version 7.0");
        self.emit_line(".target sm_70");
        self.emit_line(".address_size 64");
        self.emit_line("");

        // Global functions
        self.emit_line("// External function declarations");
        self.emit_line(".extern .func  (.param .b32 func_retval0) vprintf");
        self.emit_line("(");
        self.emit_line("    .param .b64 vprintf_param_0,");
        self.emit_line("    .param .b64 vprintf_param_1");
        self.emit_line(");");
        self.emit_line("");

        // Kernel entry point
        self.emit_line(&format!(".visible .entry {}(", self.entry_point));
        self.indent += 1;
        self.emit_line(".param .u64 param_input,");
        self.emit_line(".param .u64 param_output,");
        self.emit_line(".param .u32 param_n,");
        self.emit_line(".param .f32 param_sample_rate");
        self.indent -= 1;
        self.emit_line(")");
        self.emit_line("{");
        self.indent += 1;

        // Register declarations
        self.emit_line("// Register declarations");
        self.emit_line(".reg .pred %p<16>;");
        self.emit_line(".reg .b32 %r<64>;");
        self.emit_line(".reg .b64 %rd<32>;");
        self.emit_line(".reg .f32 %f<64>;");
        self.emit_line(".reg .f64 %fd<32>;");
        self.emit_line("");

        // Shared memory for reductions
        if resources.requires_fft {
            self.emit_line(&format!(
                ".shared .align 16 .b8 shared_mem[{}];",
                self.shared_memory * 2
            ));
        } else {
            self.emit_line(&format!(
                ".shared .align 16 .b8 shared_mem[{}];",
                self.shared_memory
            ));
        }
        self.emit_line("");

        // Thread indexing
        self.emit_line("// Compute global thread index");
        self.emit_line("mov.u32 %r1, %ctaid.x;");
        self.emit_line("mov.u32 %r2, %ntid.x;");
        self.emit_line("mov.u32 %r3, %tid.x;");
        self.emit_line(
            "mad.lo.u32 %r4, %r1, %r2, %r3;  // global_idx = blockIdx.x * blockDim.x + threadIdx.x",
        );
        self.emit_line("");

        // Load parameters
        self.emit_line("// Load parameters");
        self.emit_line("ld.param.u64 %rd1, [param_input];");
        self.emit_line("ld.param.u64 %rd2, [param_output];");
        self.emit_line("ld.param.u32 %r5, [param_n];");
        self.emit_line("ld.param.f32 %f1, [param_sample_rate];");
        self.emit_line("");

        // Bounds check
        self.emit_line("// Bounds check");
        self.emit_line("setp.ge.u32 %p1, %r4, %r5;");
        self.emit_line("@%p1 bra $L_exit;");
        self.emit_line("");

        // Compute memory addresses
        self.emit_line("// Compute memory addresses");
        self.emit_line("cvt.u64.u32 %rd3, %r4;");
        self.emit_line("shl.b64 %rd4, %rd3, 2;        // byte offset (4 bytes per float)");
        self.emit_line("add.u64 %rd5, %rd1, %rd4;    // input address");
        self.emit_line("add.u64 %rd6, %rd2, %rd4;    // output address");
        self.emit_line("");

        // Load input
        self.emit_line("// Load input value");
        self.emit_line("ld.global.f32 %f2, [%rd5];");
        self.emit_line("");

        // Generate code for each step
        let mut current_reg = 2; // %f2 has input
        for (i, step) in plan.steps.iter().enumerate() {
            self.emit_line(&format!("// Step {}: {:?}", i, step_name(step)));
            current_reg = self.generate_step(step, current_reg)?;
            self.emit_line("");
        }

        // Store output
        self.emit_line("// Store output");
        self.emit_line(&format!("st.global.f32 [%rd6], %f{};", current_reg));
        self.emit_line("");

        // Exit
        self.emit_line("$L_exit:");
        self.emit_line("ret;");

        self.indent -= 1;
        self.emit_line("}");

        Ok(())
    }

    fn generate_step(&mut self, step: &PlanStep, input_reg: u32) -> Result<u32, CompileError> {
        let output_reg = input_reg + 1;

        match step {
            PlanStep::LoadSignal { .. } => {
                // Already loaded in %f2
                Ok(input_reg)
            }

            PlanStep::Store { .. } => Ok(input_reg),

            PlanStep::ElementWise { op, .. } => {
                match op {
                    ElementWiseOp::Abs => {
                        self.emit_line(&format!("abs.f32 %f{}, %f{};", output_reg, input_reg));
                    }
                    ElementWiseOp::Square => {
                        self.emit_line(&format!(
                            "mul.f32 %f{}, %f{}, %f{};",
                            output_reg, input_reg, input_reg
                        ));
                    }
                    ElementWiseOp::Sqrt => {
                        self.emit_line(&format!(
                            "sqrt.approx.f32 %f{}, %f{};",
                            output_reg, input_reg
                        ));
                    }
                    ElementWiseOp::Log => {
                        self.emit_line(&format!(
                            "lg2.approx.f32 %f{}, %f{};",
                            output_reg, input_reg
                        ));
                        self.emit_line(&format!(
                            "mul.f32 %f{}, %f{}, 0f3f317218;",
                            output_reg, output_reg
                        )); // * ln(2)
                    }
                    ElementWiseOp::Log10 => {
                        self.emit_line(&format!(
                            "lg2.approx.f32 %f{}, %f{};",
                            output_reg, input_reg
                        ));
                        self.emit_line(&format!(
                            "mul.f32 %f{}, %f{}, 0f3e9a209b;",
                            output_reg, output_reg
                        )); // * log10(2)
                    }
                    ElementWiseOp::Exp => {
                        self.emit_line(&format!(
                            "mul.f32 %f{}, %f{}, 0f3fb8aa3b;",
                            output_reg, input_reg
                        )); // * 1/ln(2)
                        self.emit_line(&format!(
                            "ex2.approx.f32 %f{}, %f{};",
                            output_reg, output_reg
                        ));
                    }
                    ElementWiseOp::Scale(s) => {
                        self.emit_line(&format!("mov.f32 %f63, {:.8}f;", s));
                        self.emit_line(&format!(
                            "mul.f32 %f{}, %f{}, %f63;",
                            output_reg, input_reg
                        ));
                    }
                    ElementWiseOp::Offset(o) => {
                        self.emit_line(&format!("mov.f32 %f63, {:.8}f;", o));
                        self.emit_line(&format!(
                            "add.f32 %f{}, %f{}, %f63;",
                            output_reg, input_reg
                        ));
                    }
                    ElementWiseOp::Negate => {
                        self.emit_line(&format!("neg.f32 %f{}, %f{};", output_reg, input_reg));
                    }
                }
                Ok(output_reg)
            }

            PlanStep::Reduce { op, .. } => {
                // Parallel reduction using shared memory
                self.emit_line("// Store to shared memory for reduction");
                self.emit_line(&format!(
                    "st.shared.f32 [shared_mem + %r3 * 4], %f{};",
                    input_reg
                ));
                self.emit_line("bar.sync 0;");
                self.emit_line("");

                self.emit_line("// Tree reduction");
                self.emit_line("mov.u32 %r10, 128;");
                self.emit_line("$L_reduce:");
                self.emit_line("setp.ge.u32 %p2, %r3, %r10;");
                self.emit_line("@%p2 bra $L_reduce_skip;");
                self.emit_line("add.u32 %r11, %r3, %r10;");
                self.emit_line("shl.b32 %r12, %r11, 2;");
                self.emit_line("ld.shared.f32 %f60, [shared_mem + %r12];");
                self.emit_line("shl.b32 %r13, %r3, 2;");
                self.emit_line("ld.shared.f32 %f61, [shared_mem + %r13];");

                match op {
                    ReduceOp::Sum
                    | ReduceOp::Mean
                    | ReduceOp::Rms
                    | ReduceOp::Variance
                    | ReduceOp::Std => {
                        self.emit_line("add.f32 %f62, %f61, %f60;");
                    }
                    ReduceOp::Max => {
                        self.emit_line("max.f32 %f62, %f61, %f60;");
                    }
                    ReduceOp::Min => {
                        self.emit_line("min.f32 %f62, %f61, %f60;");
                    }
                    _ => {
                        self.emit_line("add.f32 %f62, %f61, %f60;");
                    }
                }

                self.emit_line("st.shared.f32 [shared_mem + %r13], %f62;");
                self.emit_line("$L_reduce_skip:");
                self.emit_line("bar.sync 0;");
                self.emit_line("shr.u32 %r10, %r10, 1;");
                self.emit_line("setp.gt.u32 %p3, %r10, 0;");
                self.emit_line("@%p3 bra $L_reduce;");
                self.emit_line("");

                self.emit_line("// Load final result");
                self.emit_line(&format!("ld.shared.f32 %f{}, [shared_mem];", output_reg));

                // Post-process based on op
                match op {
                    ReduceOp::Mean => {
                        self.emit_line(&format!("cvt.rn.f32.u32 %f60, %r5;"));
                        self.emit_line(&format!(
                            "div.approx.f32 %f{}, %f{}, %f60;",
                            output_reg, output_reg
                        ));
                    }
                    ReduceOp::Rms => {
                        self.emit_line(&format!("cvt.rn.f32.u32 %f60, %r5;"));
                        self.emit_line(&format!(
                            "div.approx.f32 %f{}, %f{}, %f60;",
                            output_reg, output_reg
                        ));
                        self.emit_line(&format!(
                            "sqrt.approx.f32 %f{}, %f{};",
                            output_reg, output_reg
                        ));
                    }
                    _ => {}
                }

                Ok(output_reg)
            }

            PlanStep::Diff { .. } => {
                self.emit_line("// Differentiation");
                self.emit_line("setp.gt.u32 %p4, %r4, 0;");
                self.emit_line("@!%p4 bra $L_diff_zero;");
                self.emit_line("sub.u64 %rd10, %rd5, 4;");
                self.emit_line("ld.global.f32 %f60, [%rd10];");
                self.emit_line(&format!("sub.f32 %f{}, %f{}, %f60;", output_reg, input_reg));
                self.emit_line("bra $L_diff_done;");
                self.emit_line("$L_diff_zero:");
                self.emit_line(&format!("mov.f32 %f{}, 0f00000000;", output_reg));
                self.emit_line("$L_diff_done:");
                Ok(output_reg)
            }

            _ => {
                // Passthrough for unimplemented
                self.emit_line(&format!("mov.f32 %f{}, %f{};", output_reg, input_reg));
                Ok(output_reg)
            }
        }
    }

    fn emit_line(&mut self, line: &str) {
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(line);
        self.output.push('\n');
    }
}

// ============================================================================
// SIMD Code Generation
// ============================================================================

fn generate_simd(plan: &ExecutionPlan) -> Result<GeneratedCode, CompileError> {
    let target_feature = SimdFeature::detect();
    let mut generator = SimdGenerator::new(target_feature);
    generator.generate(plan)?;

    // Select an execute function based on the plan
    // For now, provide a generic passthrough that copies input to output
    // The SimdRuntime can be used at runtime for more specific operations
    let execute = select_simd_execute_fn(plan);

    Ok(GeneratedCode::Simd(SimdCode {
        source: generator.output,
        execute,
        target_feature,
    }))
}

/// Select an appropriate SIMD execute function based on the plan
fn select_simd_execute_fn(plan: &ExecutionPlan) -> Option<fn(&[f64], &mut [f64])> {
    use super::simd_runtime::SimdRuntime;

    // Analyze plan to determine the primary operation
    // For simple single-operation plans, return a specific function
    // For complex plans, return None (use interpreter or runtime dispatch)

    if plan.steps.len() == 2 {
        // Simple plans: Load + Store or Load + ElementWise + Store
        if let Some(step) = plan
            .steps
            .iter()
            .find(|s| matches!(s, PlanStep::ElementWise { .. }))
        {
            if let PlanStep::ElementWise { op, .. } = step {
                let runtime = SimdRuntime::new();
                return match op {
                    ElementWiseOp::Abs => {
                        Some(runtime.get_execute_fn(super::simd_runtime::SimdOp::Abs))
                    }
                    ElementWiseOp::Square => {
                        Some(runtime.get_execute_fn(super::simd_runtime::SimdOp::Square))
                    }
                    ElementWiseOp::Sqrt => {
                        Some(runtime.get_execute_fn(super::simd_runtime::SimdOp::Sqrt))
                    }
                    ElementWiseOp::Negate => {
                        Some(runtime.get_execute_fn(super::simd_runtime::SimdOp::Negate))
                    }
                    ElementWiseOp::Log => {
                        Some(runtime.get_execute_fn(super::simd_runtime::SimdOp::Log))
                    }
                    ElementWiseOp::Exp => {
                        Some(runtime.get_execute_fn(super::simd_runtime::SimdOp::Exp))
                    }
                    _ => None,
                };
            }
        }
    }

    // For complex plans, return a generic copy function that at least does something
    // The actual execution would use the interpreter or runtime dispatch
    Some(simd_copy_passthrough)
}

/// Simple passthrough function that copies input to output
fn simd_copy_passthrough(input: &[f64], output: &mut [f64]) {
    let n = input.len().min(output.len());
    output[..n].copy_from_slice(&input[..n]);
}

struct SimdGenerator {
    output: String,
    target: SimdFeature,
    indent: usize,
}

impl SimdGenerator {
    fn new(target: SimdFeature) -> Self {
        Self {
            output: String::new(),
            target,
            indent: 0,
        }
    }

    fn generate(&mut self, plan: &ExecutionPlan) -> Result<(), CompileError> {
        // File header
        self.emit_line("//! SigQL Generated SIMD Code");
        self.emit_line(&format!("//! Target: {:?}", self.target));
        self.emit_line("");
        self.emit_line("#![allow(unused)]");
        self.emit_line("");

        // Imports based on target
        match self.target {
            SimdFeature::Sse42 | SimdFeature::Avx2 | SimdFeature::Avx512 => {
                self.emit_line("#[cfg(target_arch = \"x86_64\")]");
                self.emit_line("use std::arch::x86_64::*;");
            }
            SimdFeature::Neon => {
                self.emit_line("#[cfg(target_arch = \"aarch64\")]");
                self.emit_line("use std::arch::aarch64::*;");
            }
            SimdFeature::Scalar => {}
        }
        self.emit_line("");

        // Generate the main processing function
        let feature_attr = self.target.feature_attr();
        if !feature_attr.is_empty() {
            self.emit_line(&format!("#[target_feature(enable = \"{}\")]", feature_attr));
        }
        self.emit_line("pub unsafe fn process_signal(input: &[f32], output: &mut [f32]) {");
        self.indent += 1;

        self.emit_line("let n = input.len().min(output.len());");
        self.emit_line(&format!(
            "let simd_width = {};",
            self.target.vector_width_f32()
        ));
        self.emit_line("let simd_end = n - (n % simd_width);");
        self.emit_line("");

        // Generate SIMD loop
        self.emit_line("// SIMD vectorized loop");
        self.emit_line("let mut i = 0;");
        self.emit_line("while i < simd_end {");
        self.indent += 1;

        // Load
        self.generate_simd_load("input", "i")?;

        // Process each step
        for (step_idx, step) in plan.steps.iter().enumerate() {
            self.emit_line(&format!("// Step {}", step_idx));
            self.generate_simd_step(step)?;
        }

        // Store
        self.generate_simd_store("output", "i")?;

        self.emit_line(&format!("i += simd_width;"));
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Scalar tail
        self.emit_line("// Scalar tail");
        self.emit_line("for i in simd_end..n {");
        self.indent += 1;
        self.emit_line("output[i] = input[i];");
        self.indent -= 1;
        self.emit_line("}");

        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Generate helper functions
        self.generate_simd_helpers()?;

        Ok(())
    }

    fn generate_simd_load(&mut self, array: &str, idx: &str) -> Result<(), CompileError> {
        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line(&format!(
                    "let v = _mm256_loadu_ps({}.as_ptr().add({}));",
                    array, idx
                ));
            }
            SimdFeature::Avx512 => {
                self.emit_line(&format!(
                    "let v = _mm512_loadu_ps({}.as_ptr().add({}));",
                    array, idx
                ));
            }
            SimdFeature::Sse42 => {
                self.emit_line(&format!(
                    "let v = _mm_loadu_ps({}.as_ptr().add({}));",
                    array, idx
                ));
            }
            SimdFeature::Neon => {
                self.emit_line(&format!(
                    "let v = vld1q_f32({}.as_ptr().add({}));",
                    array, idx
                ));
            }
            SimdFeature::Scalar => {
                self.emit_line(&format!("let v = {}[{}];", array, idx));
            }
        }
        Ok(())
    }

    fn generate_simd_store(&mut self, array: &str, idx: &str) -> Result<(), CompileError> {
        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line(&format!(
                    "_mm256_storeu_ps({}.as_mut_ptr().add({}), v);",
                    array, idx
                ));
            }
            SimdFeature::Avx512 => {
                self.emit_line(&format!(
                    "_mm512_storeu_ps({}.as_mut_ptr().add({}), v);",
                    array, idx
                ));
            }
            SimdFeature::Sse42 => {
                self.emit_line(&format!(
                    "_mm_storeu_ps({}.as_mut_ptr().add({}), v);",
                    array, idx
                ));
            }
            SimdFeature::Neon => {
                self.emit_line(&format!(
                    "vst1q_f32({}.as_mut_ptr().add({}), v);",
                    array, idx
                ));
            }
            SimdFeature::Scalar => {
                self.emit_line(&format!("{}[{}] = v;", array, idx));
            }
        }
        Ok(())
    }

    fn generate_simd_step(&mut self, step: &PlanStep) -> Result<(), CompileError> {
        match step {
            PlanStep::ElementWise { op, .. } => match op {
                ElementWiseOp::Abs => self.generate_simd_abs()?,
                ElementWiseOp::Square => self.generate_simd_mul("v", "v")?,
                ElementWiseOp::Sqrt => self.generate_simd_sqrt()?,
                ElementWiseOp::Scale(s) => self.generate_simd_scale(*s as f32)?,
                ElementWiseOp::Offset(o) => self.generate_simd_offset(*o as f32)?,
                ElementWiseOp::Negate => self.generate_simd_negate()?,
                _ => {
                    self.emit_line("// Unimplemented element-wise op");
                }
            },
            PlanStep::LoadSignal { .. } | PlanStep::Store { .. } => {
                // Handled by load/store
            }
            _ => {
                self.emit_line(&format!("// Step {:?} - passthrough", step_name(step)));
            }
        }
        Ok(())
    }

    fn generate_simd_abs(&mut self) -> Result<(), CompileError> {
        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line("let sign_mask = _mm256_set1_ps(-0.0);");
                self.emit_line("let v = _mm256_andnot_ps(sign_mask, v);");
            }
            SimdFeature::Avx512 => {
                self.emit_line("let v = _mm512_abs_ps(v);");
            }
            SimdFeature::Sse42 => {
                self.emit_line("let sign_mask = _mm_set1_ps(-0.0);");
                self.emit_line("let v = _mm_andnot_ps(sign_mask, v);");
            }
            SimdFeature::Neon => {
                self.emit_line("let v = vabsq_f32(v);");
            }
            SimdFeature::Scalar => {
                self.emit_line("let v = v.abs();");
            }
        }
        Ok(())
    }

    fn generate_simd_sqrt(&mut self) -> Result<(), CompileError> {
        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line("let v = _mm256_sqrt_ps(v);");
            }
            SimdFeature::Avx512 => {
                self.emit_line("let v = _mm512_sqrt_ps(v);");
            }
            SimdFeature::Sse42 => {
                self.emit_line("let v = _mm_sqrt_ps(v);");
            }
            SimdFeature::Neon => {
                self.emit_line("let v = vsqrtq_f32(v);");
            }
            SimdFeature::Scalar => {
                self.emit_line("let v = v.sqrt();");
            }
        }
        Ok(())
    }

    fn generate_simd_mul(&mut self, a: &str, b: &str) -> Result<(), CompileError> {
        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line(&format!("let v = _mm256_mul_ps({}, {});", a, b));
            }
            SimdFeature::Avx512 => {
                self.emit_line(&format!("let v = _mm512_mul_ps({}, {});", a, b));
            }
            SimdFeature::Sse42 => {
                self.emit_line(&format!("let v = _mm_mul_ps({}, {});", a, b));
            }
            SimdFeature::Neon => {
                self.emit_line(&format!("let v = vmulq_f32({}, {});", a, b));
            }
            SimdFeature::Scalar => {
                self.emit_line(&format!("let v = {} * {};", a, b));
            }
        }
        Ok(())
    }

    fn generate_simd_scale(&mut self, scale: f32) -> Result<(), CompileError> {
        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line(&format!("let scale_v = _mm256_set1_ps({:.8});", scale));
                self.emit_line("let v = _mm256_mul_ps(v, scale_v);");
            }
            SimdFeature::Avx512 => {
                self.emit_line(&format!("let scale_v = _mm512_set1_ps({:.8});", scale));
                self.emit_line("let v = _mm512_mul_ps(v, scale_v);");
            }
            SimdFeature::Sse42 => {
                self.emit_line(&format!("let scale_v = _mm_set1_ps({:.8});", scale));
                self.emit_line("let v = _mm_mul_ps(v, scale_v);");
            }
            SimdFeature::Neon => {
                self.emit_line(&format!("let v = vmulq_n_f32(v, {:.8});", scale));
            }
            SimdFeature::Scalar => {
                self.emit_line(&format!("let v = v * {:.8};", scale));
            }
        }
        Ok(())
    }

    fn generate_simd_offset(&mut self, offset: f32) -> Result<(), CompileError> {
        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line(&format!("let offset_v = _mm256_set1_ps({:.8});", offset));
                self.emit_line("let v = _mm256_add_ps(v, offset_v);");
            }
            SimdFeature::Avx512 => {
                self.emit_line(&format!("let offset_v = _mm512_set1_ps({:.8});", offset));
                self.emit_line("let v = _mm512_add_ps(v, offset_v);");
            }
            SimdFeature::Sse42 => {
                self.emit_line(&format!("let offset_v = _mm_set1_ps({:.8});", offset));
                self.emit_line("let v = _mm_add_ps(v, offset_v);");
            }
            SimdFeature::Neon => {
                self.emit_line(&format!("let offset_v = vdupq_n_f32({:.8});", offset));
                self.emit_line("let v = vaddq_f32(v, offset_v);");
            }
            SimdFeature::Scalar => {
                self.emit_line(&format!("let v = v + {:.8};", offset));
            }
        }
        Ok(())
    }

    fn generate_simd_negate(&mut self) -> Result<(), CompileError> {
        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line("let v = _mm256_xor_ps(v, _mm256_set1_ps(-0.0));");
            }
            SimdFeature::Avx512 => {
                self.emit_line("let zero = _mm512_setzero_ps();");
                self.emit_line("let v = _mm512_sub_ps(zero, v);");
            }
            SimdFeature::Sse42 => {
                self.emit_line("let v = _mm_xor_ps(v, _mm_set1_ps(-0.0));");
            }
            SimdFeature::Neon => {
                self.emit_line("let v = vnegq_f32(v);");
            }
            SimdFeature::Scalar => {
                self.emit_line("let v = -v;");
            }
        }
        Ok(())
    }

    fn generate_simd_helpers(&mut self) -> Result<(), CompileError> {
        self.emit_line("// Helper functions for SIMD reductions");
        self.emit_line("");

        match self.target {
            SimdFeature::Avx2 => {
                self.emit_line("#[target_feature(enable = \"avx2\")]");
                self.emit_line("unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {");
                self.indent += 1;
                self.emit_line("let hi = _mm256_extractf128_ps(v, 1);");
                self.emit_line("let lo = _mm256_castps256_ps128(v);");
                self.emit_line("let sum128 = _mm_add_ps(hi, lo);");
                self.emit_line("let hi64 = _mm_movehl_ps(sum128, sum128);");
                self.emit_line("let sum64 = _mm_add_ps(sum128, hi64);");
                self.emit_line("let hi32 = _mm_shuffle_ps(sum64, sum64, 1);");
                self.emit_line("_mm_cvtss_f32(_mm_add_ss(sum64, hi32))");
                self.indent -= 1;
                self.emit_line("}");
            }
            SimdFeature::Sse42 => {
                self.emit_line("#[target_feature(enable = \"sse4.2\")]");
                self.emit_line("unsafe fn horizontal_sum_sse(v: __m128) -> f32 {");
                self.indent += 1;
                self.emit_line("let hi64 = _mm_movehl_ps(v, v);");
                self.emit_line("let sum64 = _mm_add_ps(v, hi64);");
                self.emit_line("let hi32 = _mm_shuffle_ps(sum64, sum64, 1);");
                self.emit_line("_mm_cvtss_f32(_mm_add_ss(sum64, hi32))");
                self.indent -= 1;
                self.emit_line("}");
            }
            SimdFeature::Neon => {
                self.emit_line("unsafe fn horizontal_sum_neon(v: float32x4_t) -> f32 {");
                self.indent += 1;
                self.emit_line("let sum = vaddvq_f32(v);");
                self.emit_line("sum");
                self.indent -= 1;
                self.emit_line("}");
            }
            _ => {}
        }

        Ok(())
    }

    fn emit_line(&mut self, line: &str) {
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(line);
        self.output.push('\n');
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn step_name(step: &PlanStep) -> &'static str {
    match step {
        PlanStep::LoadSignal { .. } => "LoadSignal",
        PlanStep::Store { .. } => "Store",
        PlanStep::Fft { .. } => "FFT",
        PlanStep::Ifft { .. } => "IFFT",
        PlanStep::IirFilter { .. } => "IIRFilter",
        PlanStep::FirFilter { .. } => "FIRFilter",
        PlanStep::Resample { .. } => "Resample",
        PlanStep::ComplexToMagnitude { .. } => "ComplexToMagnitude",
        PlanStep::Envelope { .. } => "Envelope",
        PlanStep::Reduce { .. } => "Reduce",
        PlanStep::Window { .. } => "Window",
        PlanStep::CrossCorrelate { .. } => "CrossCorrelate",
        PlanStep::BandPower { .. } => "BandPower",
        PlanStep::ZScore { .. } => "ZScore",
        PlanStep::Detrend { .. } => "Detrend",
        PlanStep::ElementWise { .. } => "ElementWise",
        PlanStep::Diff { .. } => "Diff",
        PlanStep::Cumsum { .. } => "Cumsum",
        PlanStep::MedianFilter { .. } => "MedianFilter",
        PlanStep::Decimate { .. } => "Decimate",
        PlanStep::Interpolate { .. } => "Interpolate",
        PlanStep::DominantFrequency { .. } => "DominantFrequency",
        PlanStep::SpectralEntropy { .. } => "SpectralEntropy",
        PlanStep::SpectralCentroid { .. } => "SpectralCentroid",
        PlanStep::Passthrough { .. } => "Passthrough",
        PlanStep::Fft2d { .. } => "FFT2D",
        PlanStep::Ifft2d { .. } => "IFFT2D",
        PlanStep::Dct2d { .. } => "DCT2D",
        PlanStep::Idct2d { .. } => "IDCT2D",
        PlanStep::Mfcc { .. } => "MFCC",
        PlanStep::PerceptualHash { .. } => "PerceptualHash",
        PlanStep::EdgeDetect { .. } => "EdgeDetect",
    }
}

fn get_output_register(step: &PlanStep) -> Option<RegisterId> {
    match step {
        PlanStep::LoadSignal { output, .. } => Some(*output),
        PlanStep::Fft { output, .. } => Some(*output),
        PlanStep::Ifft { output, .. } => Some(*output),
        PlanStep::IirFilter { output, .. } => Some(*output),
        PlanStep::FirFilter { output, .. } => Some(*output),
        PlanStep::Resample { output, .. } => Some(*output),
        PlanStep::ComplexToMagnitude { output, .. } => Some(*output),
        PlanStep::Envelope { output, .. } => Some(*output),
        PlanStep::Reduce { output, .. } => Some(*output),
        PlanStep::Window { output, .. } => Some(*output),
        PlanStep::CrossCorrelate { output, .. } => Some(*output),
        PlanStep::BandPower { output, .. } => Some(*output),
        PlanStep::ZScore { output, .. } => Some(*output),
        PlanStep::Detrend { output, .. } => Some(*output),
        PlanStep::ElementWise { output, .. } => Some(*output),
        PlanStep::Diff { output, .. } => Some(*output),
        PlanStep::Cumsum { output, .. } => Some(*output),
        PlanStep::MedianFilter { output, .. } => Some(*output),
        PlanStep::Decimate { output, .. } => Some(*output),
        PlanStep::Interpolate { output, .. } => Some(*output),
        PlanStep::DominantFrequency { output, .. } => Some(*output),
        PlanStep::SpectralEntropy { output, .. } => Some(*output),
        PlanStep::SpectralCentroid { output, .. } => Some(*output),
        PlanStep::Passthrough { output, .. } => Some(*output),
        PlanStep::Fft2d { output, .. } => Some(*output),
        PlanStep::Ifft2d { output, .. } => Some(*output),
        PlanStep::Dct2d { output, .. } => Some(*output),
        PlanStep::Idct2d { output, .. } => Some(*output),
        PlanStep::Mfcc { output, .. } => Some(*output),
        PlanStep::PerceptualHash { output, .. } => Some(*output),
        PlanStep::EdgeDetect { output, .. } => Some(*output),
        PlanStep::Store { .. } => None,
    }
}

fn get_input_register(step: &PlanStep) -> Option<RegisterId> {
    match step {
        PlanStep::LoadSignal { .. } => None,
        PlanStep::Fft { input, .. } => Some(*input),
        PlanStep::Ifft { input, .. } => Some(*input),
        PlanStep::IirFilter { input, .. } => Some(*input),
        PlanStep::FirFilter { input, .. } => Some(*input),
        PlanStep::Resample { input, .. } => Some(*input),
        PlanStep::ComplexToMagnitude { input, .. } => Some(*input),
        PlanStep::Envelope { input, .. } => Some(*input),
        PlanStep::Reduce { input, .. } => Some(*input),
        PlanStep::Window { input, .. } => Some(*input),
        PlanStep::CrossCorrelate { input_a, .. } => Some(*input_a),
        PlanStep::BandPower { input, .. } => Some(*input),
        PlanStep::ZScore { input, .. } => Some(*input),
        PlanStep::Detrend { input, .. } => Some(*input),
        PlanStep::ElementWise { input, .. } => Some(*input),
        PlanStep::Diff { input, .. } => Some(*input),
        PlanStep::Cumsum { input, .. } => Some(*input),
        PlanStep::MedianFilter { input, .. } => Some(*input),
        PlanStep::Decimate { input, .. } => Some(*input),
        PlanStep::Interpolate { input, .. } => Some(*input),
        PlanStep::DominantFrequency { input, .. } => Some(*input),
        PlanStep::SpectralEntropy { input, .. } => Some(*input),
        PlanStep::SpectralCentroid { input, .. } => Some(*input),
        PlanStep::Passthrough { input, .. } => Some(*input),
        PlanStep::Fft2d { input, .. } => Some(*input),
        PlanStep::Ifft2d { input, .. } => Some(*input),
        PlanStep::Dct2d { input, .. } => Some(*input),
        PlanStep::Idct2d { input, .. } => Some(*input),
        PlanStep::Mfcc { input, .. } => Some(*input),
        PlanStep::PerceptualHash { input, .. } => Some(*input),
        PlanStep::EdgeDetect { input, .. } => Some(*input),
        PlanStep::Store { input, .. } => Some(*input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{Compiler, CompilerConfig, Target};
    use crate::parser::parse_query;

    #[test]
    fn test_wgsl_generation() {
        let query = parse_query("FROM sensor.data TRANSFORM abs").unwrap();
        let mut compiler = Compiler::new(CompilerConfig {
            target: Target::WebGpu,
            ..Default::default()
        });
        let plan = compiler.compile(&query).unwrap();

        let code = generate(&plan, Target::WebGpu).unwrap();
        if let GeneratedCode::WebGpu(wgsl) = code {
            assert!(wgsl.wgsl.contains("@compute"));
            assert!(wgsl.wgsl.contains("workgroup_size"));
            assert!(!wgsl.bindings.is_empty());
        } else {
            panic!("Expected WebGpu code");
        }
    }

    #[test]
    fn test_cuda_generation() {
        let query = parse_query("FROM sensor.data TRANSFORM abs").unwrap();
        let mut compiler = Compiler::new(CompilerConfig {
            target: Target::Cuda,
            ..Default::default()
        });
        let plan = compiler.compile(&query).unwrap();

        let code = generate(&plan, Target::Cuda).unwrap();
        if let GeneratedCode::Cuda(cuda) = code {
            assert!(cuda.ptx.contains(".entry"));
            assert!(cuda.ptx.contains(".target sm_70"));
        } else {
            panic!("Expected CUDA code");
        }
    }

    #[test]
    fn test_simd_generation() {
        let query = parse_query("FROM sensor.data TRANSFORM abs").unwrap();
        let mut compiler = Compiler::new(CompilerConfig {
            target: Target::Simd,
            ..Default::default()
        });
        let plan = compiler.compile(&query).unwrap();

        let code = generate(&plan, Target::Simd).unwrap();
        if let GeneratedCode::Simd(simd) = code {
            assert!(simd.source.contains("unsafe fn process_signal"));
        } else {
            panic!("Expected SIMD code");
        }
    }
}
