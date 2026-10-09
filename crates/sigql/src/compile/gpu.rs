//! WGSL compute kernels for SigQL plan steps.
//!
//! The fused shader from [`super::codegen`] cannot express steps that read
//! other samples' results (IIR recurrences, prefix sums, whole-signal
//! statistics) without cross-workgroup synchronisation, so its output was
//! never a faithful copy of the CPU runtime. This module instead lowers each
//! supported [`PlanStep`] to one self-contained compute shader with the same
//! semantics as [`crate::runtime::Runtime`]. A host (the JouleDB server's
//! wgpu executor) chains the kernels register by register; any step with no
//! kernel here is reported so the host can fall back to the CPU runtime with
//! that reason.
//!
//! Every kernel has the same interface:
//!
//! ```text
//! @binding(0) input:  array<f32>   (read)        the step's input register
//! @binding(1) output: array<f32>   (read_write)  the step's output register
//! @binding(2) params: Params       (uniform)     n, aux_len, size, scale
//! @binding(3) aux:    array<f32>   (read)        taps / window, >= 1 element
//! ```

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use super::plan::{ElementWiseOp, PlanStep, ReduceOp};

/// Threads per workgroup. 64 fits every adapter's default limits.
pub const WORKGROUP: u32 = 64;

/// Number of `f32` values a [`GpuKernel::Stats`] dispatch writes.
pub const STATS_LEN: usize = 7;

/// A plan step lowered to one compute shader.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuKernel {
    /// `output[i] = f(input[i])`, `expr` is WGSL over `x`.
    ElementWise { expr: String },
    /// `output[i] = input[i + 1] - input[i]`; output has `n - 1` samples.
    Diff,
    /// Inclusive prefix sum.
    Cumsum,
    /// `(x - mean) / sample_std`, all zeros when the std is below 1e-10.
    ZScore,
    /// Cascade of Direct Form II Transposed biquads, `[b0, b1, b2, a0, a1, a2]`.
    Iir { sections: Vec<[f64; 6]> },
    /// Causal convolution with the taps in `aux`.
    Fir { taps: Vec<f64> },
    /// One-sided magnitude spectrum, `size / 2 + 1` bins, window in `aux`,
    /// each bin scaled by `1 / (coherent_gain * sqrt(size))`.
    Spectrum {
        size: u32,
        window: Vec<f64>,
        coherent_gain: f64,
    },
    /// Whole-signal statistics feeding [`reduce_from_stats`].
    Stats { op: ReduceOp },
}

/// Output length of a kernel for an input of `n` samples.
impl GpuKernel {
    /// Lower one plan step. `Err` names the step the GPU cannot run.
    pub fn from_step(step: &PlanStep) -> Result<Option<Self>, String> {
        let kernel = match step {
            // Bookkeeping steps the host handles itself.
            PlanStep::LoadSignal { .. } | PlanStep::Store { .. } | PlanStep::Passthrough { .. } => {
                return Ok(None);
            }
            PlanStep::ElementWise { op, .. } => Self::ElementWise {
                expr: match op {
                    ElementWiseOp::Abs => "abs(x)".into(),
                    ElementWiseOp::Square => "x * x".into(),
                    ElementWiseOp::Sqrt => "sqrt(max(x, 0.0))".into(),
                    ElementWiseOp::Log => "log(max(x, 1e-10))".into(),
                    ElementWiseOp::Log10 => "log(max(x, 1e-10)) * 0.4342944819".into(),
                    ElementWiseOp::Exp => "exp(x)".into(),
                    ElementWiseOp::Scale(s) => format!("x * {}", wgsl_f32(*s)),
                    ElementWiseOp::Offset(o) => format!("x + {}", wgsl_f32(*o)),
                    ElementWiseOp::Negate => "-x".into(),
                },
            },
            PlanStep::Diff { .. } => Self::Diff,
            PlanStep::Cumsum { .. } => Self::Cumsum,
            PlanStep::ZScore { .. } => Self::ZScore,
            PlanStep::IirFilter { coeffs, .. } => Self::Iir {
                sections: coeffs.sections.clone(),
            },
            PlanStep::FirFilter { coeffs, .. } => Self::Fir {
                taps: coeffs.taps.clone(),
            },
            PlanStep::Fft { size, window, .. } => {
                if *size == 0 || !size.is_power_of_two() || *size > (1 << 16) {
                    return Err(format!(
                        "FFT size {size} is outside the GPU kernel's 1..=65536 power-of-two range"
                    ));
                }
                Self::Spectrum {
                    size: *size as u32,
                    window: window.coeffs.clone(),
                    coherent_gain: window.coherent_gain,
                }
            }
            PlanStep::Reduce { op, .. } => match op {
                ReduceOp::Sum
                | ReduceOp::Mean
                | ReduceOp::Rms
                | ReduceOp::Max
                | ReduceOp::Min
                | ReduceOp::Variance
                | ReduceOp::Std
                | ReduceOp::ZeroCrossings
                | ReduceOp::PeakToPeak => Self::Stats { op: *op },
                other => return Err(format!("reduce {other:?} has no GPU kernel")),
            },
            other => {
                return Err(format!(
                    "{} has no GPU kernel",
                    super::codegen::step_label(other)
                ))
            }
        };
        Ok(Some(kernel))
    }

    /// Short name for receipts and logs.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ElementWise { .. } => "elementwise",
            Self::Diff => "diff",
            Self::Cumsum => "cumsum",
            Self::ZScore => "zscore",
            Self::Iir { .. } => "iir",
            Self::Fir { .. } => "fir",
            Self::Spectrum { .. } => "spectrum",
            Self::Stats { .. } => "stats",
        }
    }

    /// Number of `f32` values written for an `n`-sample input.
    pub fn output_len(&self, n: usize) -> usize {
        match self {
            Self::Diff => n.saturating_sub(1),
            Self::Spectrum { size, .. } => *size as usize / 2 + 1,
            Self::Stats { .. } => STATS_LEN,
            _ => n,
        }
    }

    /// Minimum input length the CPU runtime accepts for this step.
    pub fn min_input_len(&self) -> usize {
        match self {
            Self::ZScore => 2,
            Self::Stats { op: ReduceOp::Std } => 2,
            _ => 1,
        }
    }

    /// Workgroups to dispatch. Kernels that need the whole signal run in one
    /// workgroup and stride over it; per-sample kernels cover it with many.
    pub fn workgroups(&self, n: usize) -> u32 {
        let per_sample = |len: usize| (len.max(1) as u32).div_ceil(WORKGROUP);
        match self {
            Self::ElementWise { .. } | Self::Fir { .. } => per_sample(n),
            Self::Diff => per_sample(n.saturating_sub(1)),
            Self::Spectrum { .. } => per_sample(self.output_len(n)),
            Self::Cumsum | Self::ZScore | Self::Iir { .. } | Self::Stats { .. } => 1,
        }
    }

    /// Contents of the `aux` binding (never empty: WebGPU rejects 0-byte bindings).
    pub fn aux(&self) -> Vec<f32> {
        let v: Vec<f32> = match self {
            Self::Fir { taps } => taps.iter().map(|&t| t as f32).collect(),
            Self::Spectrum { window, .. } => window.iter().map(|&w| w as f32).collect(),
            _ => Vec::new(),
        };
        if v.is_empty() {
            alloc_vec_one()
        } else {
            v
        }
    }

    /// The `params` uniform: `[n, aux_len, size, scale_bits]`.
    pub fn params(&self, n: usize) -> [u32; 4] {
        let aux_len = match self {
            Self::Fir { taps } => taps.len(),
            Self::Spectrum { window, .. } => window.len(),
            _ => 0,
        } as u32;
        match self {
            Self::Spectrum {
                size,
                coherent_gain,
                ..
            } => {
                let scale = 1.0 / (coherent_gain * libm_sqrt(*size as f64));
                [n as u32, aux_len, *size, (scale as f32).to_bits()]
            }
            _ => [n as u32, aux_len, 0, 0],
        }
    }

    /// The compute shader for this kernel, entry point `main`.
    pub fn wgsl(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("// SigQL GPU kernel: {}\n", self.name()));
        s.push_str(HEADER);
        match self {
            Self::ElementWise { expr } => {
                s.push_str(&per_sample_main(&format!(
                    "    if (i < params.n) {{\n        let x = input[i];\n        output[i] = {expr};\n    }}\n"
                )));
            }
            Self::Diff => s.push_str(&per_sample_main(
                "    if (i + 1u < params.n) {\n        output[i] = input[i + 1u] - input[i];\n    }\n",
            )),
            Self::Fir { .. } => s.push_str(&per_sample_main(
                "    if (i < params.n) {\n        var acc: f32 = 0.0;\n        let taps = min(params.aux_len, i + 1u);\n        for (var j = 0u; j < taps; j++) {\n            acc += aux[j] * input[i - j];\n        }\n        output[i] = acc;\n    }\n",
            )),
            Self::Spectrum { .. } => s.push_str(&per_sample_main(SPECTRUM_BODY)),
            Self::Cumsum => {
                s.push_str(&reduce_helpers());
                s.push_str(CUMSUM_MAIN);
            }
            Self::ZScore => {
                s.push_str(&reduce_helpers());
                s.push_str(ZSCORE_MAIN);
            }
            Self::Stats { .. } => {
                s.push_str(&reduce_helpers());
                s.push_str(STATS_MAIN);
            }
            Self::Iir { sections } => s.push_str(&iir_main(sections)),
        }
        s
    }
}

/// Turn the [`STATS_LEN`] values a `Stats` kernel wrote into the scalar the
/// CPU runtime reports for `op` on an `n`-sample signal, using the same
/// formulas (value and 95% interval).
#[cfg(feature = "std")]
pub fn reduce_from_stats(
    op: ReduceOp,
    stats: &[f64],
    n: usize,
) -> Option<crate::types::UncertainValue<f64>> {
    use crate::types::UncertainValue;
    if n == 0 || stats.len() < STATS_LEN {
        return None;
    }
    let nf = n as f64;
    let [sum, sum_sq, max, min, crossings, dev2, sq_dev2] = [
        stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6],
    ];
    let mean = sum / nf;
    let sample_var = dev2 / (n - 1).max(1) as f64;
    let generic_se = sample_var.sqrt() / nf.sqrt();
    let (value, se) = match op {
        ReduceOp::Sum => (sum, generic_se),
        ReduceOp::Mean => (mean, (sample_var / nf).sqrt()),
        ReduceOp::Max => (max, generic_se),
        ReduceOp::Min => (min, generic_se),
        ReduceOp::PeakToPeak => (max - min, generic_se),
        ReduceOp::ZeroCrossings => (crossings, generic_se),
        ReduceOp::Variance => (dev2 / nf, generic_se),
        ReduceOp::Std => {
            if n < 2 {
                return None;
            }
            let std = (dev2 / (n - 1) as f64).sqrt();
            (std, std / (2.0 * (n - 1) as f64).sqrt())
        }
        ReduceOp::Rms => {
            let rms = (sum_sq / nf).sqrt();
            let var_sq = sq_dev2 / (n - 1).max(1) as f64;
            (rms, (var_sq / nf).sqrt() / (2.0 * rms).max(1e-10))
        }
        _ => return None,
    };
    Some(UncertainValue::from_mean_se(value, se, n))
}

fn alloc_vec_one() -> Vec<f32> {
    let mut v = Vec::new();
    v.push(0.0);
    v
}

fn libm_sqrt(x: f64) -> f64 {
    // Newton iteration so this also builds without std.
    if x <= 0.0 {
        return 0.0;
    }
    let mut r = if x > 1.0 { x / 2.0 } else { 1.0 };
    for _ in 0..64 {
        r = 0.5 * (r + x / r);
    }
    r
}

/// A finite f64 as a WGSL f32 literal (always has a decimal point or exponent).
fn wgsl_f32(v: f64) -> String {
    let v = v as f32;
    if v.is_finite() {
        format!("{v:e}")
    } else {
        "0.0".into()
    }
}

const HEADER: &str = "
struct Params {
    n: u32,
    aux_len: u32,
    size: u32,
    scale_bits: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> aux: array<f32>;

";

fn per_sample_main(body: &str) -> String {
    format!(
        "@compute @workgroup_size({WORKGROUP})\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let i = gid.x;\n{body}}}\n"
    )
}

fn reduce_helpers() -> String {
    format!(
        "var<workgroup> scratch: array<f32, {WORKGROUP}>;

fn wg_sum(lid: u32, v: f32) -> f32 {{
    workgroupBarrier();
    scratch[lid] = v;
    workgroupBarrier();
    for (var stride = {half}u; stride > 0u; stride = stride >> 1u) {{
        if (lid < stride) {{ scratch[lid] = scratch[lid] + scratch[lid + stride]; }}
        workgroupBarrier();
    }}
    return scratch[0];
}}

fn wg_max(lid: u32, v: f32) -> f32 {{
    workgroupBarrier();
    scratch[lid] = v;
    workgroupBarrier();
    for (var stride = {half}u; stride > 0u; stride = stride >> 1u) {{
        if (lid < stride) {{ scratch[lid] = max(scratch[lid], scratch[lid + stride]); }}
        workgroupBarrier();
    }}
    return scratch[0];
}}

fn wg_min(lid: u32, v: f32) -> f32 {{
    workgroupBarrier();
    scratch[lid] = v;
    workgroupBarrier();
    for (var stride = {half}u; stride > 0u; stride = stride >> 1u) {{
        if (lid < stride) {{ scratch[lid] = min(scratch[lid], scratch[lid + stride]); }}
        workgroupBarrier();
    }}
    return scratch[0];
}}

",
        half = WORKGROUP / 2
    )
}

const SPECTRUM_BODY: &str = "    let bins = params.size / 2u + 1u;
    if (i < bins) {
        let size = params.size;
        let len = min(params.n, size);
        var re: f32 = 0.0;
        var im: f32 = 0.0;
        // Twiddle index kept modulo `size` in integers so the angle stays exact.
        let k = i % size;
        for (var t = 0u; t < len; t++) {
            let w = select(1.0, aux[min(t, params.aux_len - 1u)], t < params.aux_len);
            let x = input[t] * w;
            let phase = (k * t) % size;
            let theta = -6.2831853071795864 * f32(phase) / f32(size);
            re += x * cos(theta);
            im += x * sin(theta);
        }
        output[i] = sqrt(re * re + im * im) * bitcast<f32>(params.scale_bits);
    }
";

const CUMSUM_MAIN: &str = "var<workgroup> prefix: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n = params.n;
    let chunk = (n + 63u) / 64u;
    let start = min(lid * chunk, n);
    let end = min(start + chunk, n);
    var local_sum: f32 = 0.0;
    for (var i = start; i < end; i++) {
        local_sum += input[i];
    }
    prefix[lid] = local_sum;
    workgroupBarrier();
    if (lid == 0u) {
        var running: f32 = 0.0;
        for (var j = 0u; j < 64u; j++) {
            let v = prefix[j];
            prefix[j] = running;
            running += v;
        }
    }
    workgroupBarrier();
    var acc = prefix[lid];
    for (var i = start; i < end; i++) {
        acc += input[i];
        output[i] = acc;
    }
}
";

const ZSCORE_MAIN: &str = "@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n = params.n;
    var s: f32 = 0.0;
    for (var i = lid; i < n; i += 64u) { s += input[i]; }
    let mean = wg_sum(lid, s) / f32(n);
    var d: f32 = 0.0;
    for (var i = lid; i < n; i += 64u) { let e = input[i] - mean; d += e * e; }
    let sd = sqrt(wg_sum(lid, d) / f32(max(n, 2u) - 1u));
    for (var i = lid; i < n; i += 64u) {
        output[i] = select(0.0, (input[i] - mean) / sd, sd >= 1e-10);
    }
}
";

const STATS_MAIN: &str = "@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n = params.n;
    var s: f32 = 0.0;
    var sq: f32 = 0.0;
    var hi: f32 = -3.4028234e38;
    var lo: f32 = 3.4028234e38;
    var zc: f32 = 0.0;
    for (var i = lid; i < n; i += 64u) {
        let x = input[i];
        s += x;
        sq += x * x;
        hi = max(hi, x);
        lo = min(lo, x);
        if (i > 0u && ((input[i - 1u] >= 0.0) != (x >= 0.0))) { zc += 1.0; }
    }
    let total = wg_sum(lid, s);
    let total_sq = wg_sum(lid, sq);
    let top = wg_max(lid, hi);
    let bottom = wg_min(lid, lo);
    let crossings = wg_sum(lid, zc);
    let mean = total / f32(n);
    let mean_sq = total_sq / f32(n);
    var d: f32 = 0.0;
    var q: f32 = 0.0;
    for (var i = lid; i < n; i += 64u) {
        let x = input[i];
        d += (x - mean) * (x - mean);
        q += (x * x - mean_sq) * (x * x - mean_sq);
    }
    let dev2 = wg_sum(lid, d);
    let sq_dev2 = wg_sum(lid, q);
    if (lid == 0u) {
        output[0] = total;
        output[1] = total_sq;
        output[2] = top;
        output[3] = bottom;
        output[4] = crossings;
        output[5] = dev2;
        output[6] = sq_dev2;
    }
}
";

fn iir_main(sections: &[[f64; 6]]) -> String {
    // The recurrence is sequential, so one invocation walks the signal and
    // carries every section's state; cascading per sample equals running
    // each section over the whole signal in turn.
    let mut s = String::from("@compute @workgroup_size(1)\nfn main() {\n    let n = params.n;\n");
    for k in 0..sections.len() {
        s.push_str(&format!(
            "    var z1_{k}: f32 = 0.0;\n    var z2_{k}: f32 = 0.0;\n"
        ));
    }
    s.push_str("    for (var i = 0u; i < n; i++) {\n        var y = input[i];\n");
    for (k, sec) in sections.iter().enumerate() {
        let [b0, b1, b2, _a0, a1, a2] = *sec;
        s.push_str(&format!(
            "        let x_{k} = y;\n        y = {b0} * x_{k} + z1_{k};\n        z1_{k} = {b1} * x_{k} - {a1} * y + z2_{k};\n        z2_{k} = {b2} * x_{k} - {a2} * y;\n",
            b0 = wgsl_f32(b0),
            b1 = wgsl_f32(b1),
            b2 = wgsl_f32(b2),
            a1 = wgsl_f32(a1),
            a2 = wgsl_f32(a2),
        ));
    }
    s.push_str("        output[i] = y;\n    }\n}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::plan::{FirCoeffs, IirCoeffs, RegisterId, WindowCoeffs};

    fn all_kernels() -> Vec<GpuKernel> {
        let r = RegisterId(0);
        let mut steps = vec![
            PlanStep::Diff {
                input: r,
                output: r,
            },
            PlanStep::Cumsum {
                input: r,
                output: r,
            },
            PlanStep::ZScore {
                input: r,
                output: r,
            },
            PlanStep::IirFilter {
                input: r,
                output: r,
                coeffs: IirCoeffs {
                    sections: vec![
                        [0.2, 0.4, 0.2, 1.0, -0.3, 0.1],
                        [1.0, -2.0, 1.0, 1.0, -1.8, 0.81],
                    ],
                },
            },
            PlanStep::IirFilter {
                input: r,
                output: r,
                coeffs: IirCoeffs { sections: vec![] },
            },
            PlanStep::FirFilter {
                input: r,
                output: r,
                coeffs: FirCoeffs {
                    taps: vec![0.25, 0.5, 0.25],
                },
            },
            PlanStep::Fft {
                input: r,
                output: r,
                size: 256,
                window: WindowCoeffs {
                    coeffs: vec![1.0; 256],
                    coherent_gain: 1.0,
                    enbw: 1.0,
                },
            },
        ];
        for op in [
            ElementWiseOp::Abs,
            ElementWiseOp::Square,
            ElementWiseOp::Sqrt,
            ElementWiseOp::Log,
            ElementWiseOp::Log10,
            ElementWiseOp::Exp,
            ElementWiseOp::Scale(-2.5),
            ElementWiseOp::Offset(1e-12),
            ElementWiseOp::Negate,
        ] {
            steps.push(PlanStep::ElementWise {
                input: r,
                output: r,
                op,
            });
        }
        for op in [
            ReduceOp::Sum,
            ReduceOp::Mean,
            ReduceOp::Rms,
            ReduceOp::Max,
            ReduceOp::Min,
            ReduceOp::Variance,
            ReduceOp::Std,
            ReduceOp::ZeroCrossings,
            ReduceOp::PeakToPeak,
        ] {
            steps.push(PlanStep::Reduce {
                input: r,
                output: r,
                op,
            });
        }
        steps
            .iter()
            .map(|s| GpuKernel::from_step(s).unwrap().unwrap())
            .collect()
    }

    /// Every kernel the GPU path can emit must parse and validate as WGSL
    /// (naga is the compiler wgpu uses), so a shader bug fails the build
    /// instead of silently pushing queries onto the CPU.
    #[test]
    fn every_kernel_is_valid_wgsl() {
        for kernel in all_kernels() {
            let src = kernel.wgsl();
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                panic!(
                    "{} does not parse:\n{}\n{src}",
                    kernel.name(),
                    e.emit_to_string(&src)
                )
            });
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{} does not validate: {e:?}\n{src}", kernel.name()));
        }
    }

    #[test]
    fn unsupported_steps_name_themselves() {
        let r = RegisterId(0);
        let err = GpuKernel::from_step(&PlanStep::Envelope {
            input: r,
            output: r,
        })
        .unwrap_err();
        assert!(err.contains("Envelope"), "{err}");
        let err = GpuKernel::from_step(&PlanStep::Reduce {
            input: r,
            output: r,
            op: ReduceOp::Kurtosis,
        })
        .unwrap_err();
        assert!(err.contains("Kurtosis"), "{err}");
    }

    /// The host-side formulas must reproduce the CPU runtime's scalars when
    /// fed exact statistics.
    #[test]
    fn stats_formulas_match_cpu_runtime() {
        let samples: Vec<f64> = (0..257)
            .map(|i| ((i as f64) * 0.37).sin() * 3.0 + 0.5)
            .collect();
        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let mean_sq = samples.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let stats = [
            samples.iter().sum::<f64>(),
            samples.iter().map(|x| x * x).sum::<f64>(),
            samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            samples.iter().cloned().fold(f64::INFINITY, f64::min),
            samples
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count() as f64,
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>(),
            samples
                .iter()
                .map(|x| (x * x - mean_sq).powi(2))
                .sum::<f64>(),
        ];
        let signal = crate::types::DynSignal::new("s", samples.clone(), 1000, 0);
        for op in [
            ReduceOp::Sum,
            ReduceOp::Mean,
            ReduceOp::Rms,
            ReduceOp::Max,
            ReduceOp::Min,
            ReduceOp::Variance,
            ReduceOp::Std,
            ReduceOp::ZeroCrossings,
            ReduceOp::PeakToPeak,
        ] {
            let plan = crate::compile::ExecutionPlan {
                steps: vec![
                    PlanStep::LoadSignal {
                        source: "s".into(),
                        output: RegisterId(0),
                    },
                    PlanStep::Reduce {
                        input: RegisterId(0),
                        output: RegisterId(1),
                        op,
                    },
                    PlanStep::Store {
                        input: RegisterId(1),
                        name: "v".into(),
                    },
                ],
                target: crate::compile::Target::Simd,
            };
            let mut rt = crate::runtime::Runtime::default();
            rt.register_signal("s", signal.clone());
            let cpu = match rt.execute(&plan).unwrap().outputs.remove("v").unwrap() {
                crate::runtime::OutputValue::Scalar(v) => v,
                other => panic!("{other:?}"),
            };
            let gpu = reduce_from_stats(op, &stats, n).unwrap();
            for (a, b, what) in [
                (gpu.value, cpu.value, "value"),
                (gpu.lower_bound, cpu.lower_bound, "lower"),
                (gpu.upper_bound, cpu.upper_bound, "upper"),
            ] {
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                    "{op:?} {what}: {a} vs {b}"
                );
            }
            assert_eq!(gpu.n_samples, cpu.n_samples);
        }
    }
}
