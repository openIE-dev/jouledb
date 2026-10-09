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
use alloc::{format, string::String, vec, vec::Vec};

use super::plan::{ElementWiseOp, PlanStep, ReduceOp};
use crate::types::FrequencyBand;

/// Threads per workgroup. 64 fits every adapter's default limits.
pub const WORKGROUP: u32 = 64;

/// Number of `f32` values a [`GpuKernel::Stats`] dispatch writes:
/// `[sum, sum_sq, max, min, crossings, dev2, sq_dev2, dev3, dev4, xy]`.
pub const STATS_LEN: usize = 10;

/// Number of `u32` words in the `params` uniform.
pub const PARAMS_LEN: usize = 8;

/// Largest power-of-two DFT size the direct DFT kernels accept (`k * t`
/// must fit in a `u32`).
const MAX_DFT_SIZE: usize = 1 << 16;

/// What the host knows about a step's inputs when it binds the kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuInput {
    /// Samples (or spectrum bins) in the primary input register.
    pub len: usize,
    /// Sample rate of the primary input.
    pub rate: u32,
    /// The primary input is a magnitude spectrum, not a time signal.
    pub is_spectrum: bool,
    /// Samples in the second input (cross-correlation only).
    pub second_len: Option<usize>,
    /// The CPU runtime's `default_sample_rate` (band power uses it for bin width).
    pub default_rate: u32,
}

/// Band-power bin range, fixed once the input is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandBins {
    /// Input is a magnitude spectrum (sum its bins) rather than a signal
    /// (evaluate the DFT bins in the band).
    pub from_spectrum: bool,
    /// Transform size the bins belong to.
    pub size: u32,
    /// First bin, inclusive.
    pub low: u32,
    /// Last bin, inclusive. `high < low` means an empty band (power 0).
    pub high: u32,
    /// Bin width in Hz.
    pub df: f64,
}

/// How a biquad cascade is split for the parallel IIR kernel.
///
/// The cascade is linear in its state `s` (two DF2T delays per section):
/// with zero input, `L` samples map `s` to `M s` with `M = A^L`, so chunk
/// `c` maps an entry state to `M s + local[c]`, an affine map with the same
/// matrix for every full chunk. The true state leaving chunk `c` is the
/// inclusive prefix `X[c] = sum_{j<=c} M^(c-j) local[j]`, computed as a
/// two-level tree:
/// 1. `chunk_state`: every chunk runs from a zero state, in parallel, and
///    records where its state ends (`local[c]`);
/// 2. `scan_block`: each workgroup of [`IIR_SCAN_BLOCK`] chunks scans its
///    block in workgroup memory (Hillis-Steele, `log2(block)` levels,
///    `X[c] += M^s X[c - s]`), leaving per-block prefixes and block totals;
/// 3. `scan_0 .. scan_{levels-1}`: Hillis-Steele over the block totals in
///    global memory with the block map `M^block`, `ceil(log2 blocks)` levels;
/// 4. `main`: chunk `c` enters with `X[c - 1]` = its block prefix plus
///    `M^(offset + 1)` times the previous blocks' total (zero for chunk 0),
///    and reruns its samples in parallel.
/// Every matrix power is computed on the host in f64. Outputs come from the
/// same recurrence as a sequential pass; only the entry states pass through
/// the powers.
#[derive(Debug, Clone, PartialEq)]
pub struct IirChunks {
    /// Number of chunks.
    pub chunks: u32,
    /// Samples per chunk (the last one may be shorter).
    pub chunk_len: u32,
    /// Levels of the block-total scan, `ceil(log2 blocks)` (0 for one block).
    pub levels: u32,
    /// Row-major `2K x 2K` matrices (`K` sections), concatenated: `M^j` for
    /// `j in 1..=IIR_SCAN_BLOCK`, then `M^(IIR_SCAN_BLOCK * 2^k)` for
    /// `k in 0..levels`.
    pub powers: Vec<f64>,
}

/// Chunks per workgroup-local scan block (the workgroup size).
pub const IIR_SCAN_BLOCK: usize = WORKGROUP as usize;

/// Most chunks the IIR scan uses (so at most 10 block-total levels).
pub const MAX_IIR_CHUNKS: usize = 1 << 16;

/// Entry points of the block-total scan levels, by level.
pub const IIR_SCAN_ENTRIES: [&str; 10] = [
    "scan_0", "scan_1", "scan_2", "scan_3", "scan_4", "scan_5", "scan_6", "scan_7", "scan_8",
    "scan_9",
];

fn mat_mul(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut out = vec![0.0; d * d];
    for r in 0..d {
        for c in 0..d {
            out[r * d + c] = (0..d).map(|q| a[r * d + q] * b[q * d + c]).sum();
        }
    }
    out
}

impl IirChunks {
    /// Default split of `n` samples: chunks of about [`IIR_CHUNK_LEN`]
    /// samples, at least `sqrt(16 n)` chunks for short signals, at most
    /// [`MAX_IIR_CHUNKS`].
    pub fn plan(sections: &[[f64; 6]], n: usize) -> Self {
        let n = n.max(1);
        let by_len = n.div_ceil(IIR_CHUNK_LEN);
        let by_sqrt = ceil_to_usize(libm_sqrt(16.0 * n as f64));
        Self::with_chunks(sections, n, by_len.max(by_sqrt))
    }

    /// Split `n` samples into about `target` chunks (benchmarks sweep this).
    pub fn with_chunks(sections: &[[f64; 6]], n: usize, target: usize) -> Self {
        let n = n.max(1);
        let target = target.clamp(1, MAX_IIR_CHUNKS).min(n);
        let chunk_len = n.div_ceil(target);
        let chunks = n.div_ceil(chunk_len);
        let blocks = chunks.div_ceil(IIR_SCAN_BLOCK);
        let levels = usize::BITS - (blocks - 1).leading_zeros();
        let d = 2 * sections.len();
        // M = A^L: simulate L zero-input samples from each basis state.
        let mut m = vec![0.0; d * d];
        for j in 0..d {
            let mut state = vec![0.0f64; d];
            state[j] = 1.0;
            for _ in 0..chunk_len {
                cascade_step(sections, 0.0, &mut state);
            }
            for (r, v) in state.iter().enumerate() {
                m[r * d + j] = *v;
            }
        }
        let mut powers = Vec::with_capacity((IIR_SCAN_BLOCK + levels as usize) * d * d);
        let mut p = m.clone();
        for _ in 0..IIR_SCAN_BLOCK {
            powers.extend_from_slice(&p);
            p = mat_mul(&p, &m, d);
        }
        // M^block, squared per level.
        let mut q = powers[(IIR_SCAN_BLOCK - 1) * d * d..].to_vec();
        for _ in 0..levels {
            powers.extend_from_slice(&q);
            q = mat_mul(&q, &q, d);
        }
        Self {
            chunks: chunks as u32,
            chunk_len: chunk_len as u32,
            levels,
            powers,
        }
    }

    /// Number of scan blocks.
    pub fn blocks(&self) -> u32 {
        self.chunks.div_ceil(IIR_SCAN_BLOCK as u32)
    }

    /// `M = A^chunk_len` (row-major), the per-chunk state map.
    pub fn carry(&self, sections: usize) -> &[f64] {
        &self.powers[..(4 * sections * sections).min(self.powers.len())]
    }
}

/// Default samples per IIR chunk; see [`IirChunks::plan`].
pub const IIR_CHUNK_LEN: usize = 64;

/// One sample through the cascade (DF2T per section), as the kernels do.
fn cascade_step(sections: &[[f64; 6]], x: f64, state: &mut [f64]) -> f64 {
    let mut y = x;
    for (k, &[b0, b1, b2, _a0, a1, a2]) in sections.iter().enumerate() {
        let xk = y;
        y = b0 * xk + state[2 * k];
        state[2 * k] = b1 * xk - a1 * y + state[2 * k + 1];
        state[2 * k + 1] = b2 * xk - a2 * y;
    }
    y
}

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
    /// Cascade of Direct Form II Transposed biquads, `[b0, b1, b2, a0, a1, a2]`,
    /// run as a chunked parallel scan (see [`IirChunks`]); `chunks` is filled
    /// in by [`GpuKernel::bind`].
    Iir {
        sections: Vec<[f64; 6]>,
        chunks: Option<IirChunks>,
    },
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
    /// Remove the mean (`order == 0`) or the least-squares line (otherwise),
    /// as `dsp::statistics::detrend` does.
    Detrend { order: u8 },
    /// `|analytic signal|` through a zero-padded power-of-two FFT, as
    /// `dsp::envelope::HilbertTransform::envelope` does.
    Envelope,
    /// 8th-order Butterworth anti-alias filter at `0.45 * rate / factor`,
    /// then every `factor`-th sample, as `dsp::resample::decimate` does.
    /// `sections` are filled in by [`GpuKernel::bind`] from the input rate.
    Decimate {
        factor: usize,
        sections: Vec<[f64; 6]>,
        chunks: Option<IirChunks>,
    },
    /// Power in a band, `sum |X_k|^2 * df`, from a spectrum or a signal.
    BandPower {
        band: FrequencyBand,
        bins: Option<BandBins>,
    },
    /// Linear cross-correlation `c[lag] = sum_t a[t + lag] * b[t]` for
    /// `lag` in `-max_lag..=max_lag`; `b` is bound in place of `aux`.
    CrossCorrelate { max_lag: usize, len_b: usize },
}

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
                chunks: None,
            },
            PlanStep::FirFilter { coeffs, .. } => Self::Fir {
                taps: coeffs.taps.clone(),
            },
            PlanStep::Fft { size, window, .. } => {
                if *size == 0 || !size.is_power_of_two() || *size > MAX_DFT_SIZE {
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
                | ReduceOp::PeakToPeak
                | ReduceOp::Skewness
                | ReduceOp::Kurtosis
                | ReduceOp::Slope => Self::Stats { op: *op },
            },
            PlanStep::Detrend { order, .. } => Self::Detrend { order: *order },
            PlanStep::Envelope { .. } => Self::Envelope,
            PlanStep::Decimate { factor, .. } => Self::Decimate {
                factor: *factor,
                sections: Vec::new(),
                chunks: None,
            },
            PlanStep::BandPower { band, .. } => Self::BandPower {
                band: *band,
                bins: None,
            },
            PlanStep::CrossCorrelate {
                max_lag_samples, ..
            } => Self::CrossCorrelate {
                max_lag: *max_lag_samples,
                len_b: 0,
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

    /// Fix everything that depends on the actual inputs (rates, lengths,
    /// input kind) and check the CPU runtime's preconditions. `Err` is a
    /// reason to run the step on the CPU instead.
    pub fn bind(&self, input: &GpuInput) -> Result<Self, String> {
        let name = self.name();
        if input.is_spectrum && !matches!(self, Self::BandPower { .. }) {
            return Err(format!("{name} needs a time-domain signal input"));
        }
        if input.len < self.min_input_len() {
            return Err(format!(
                "{name} needs at least {} samples, got {}",
                self.min_input_len(),
                input.len
            ));
        }
        Ok(match self {
            Self::Decimate { factor, .. } => {
                if *factor == 0 {
                    return Err("decimate factor 0 is invalid".into());
                }
                let sections = if *factor <= 1 {
                    Vec::new()
                } else {
                    use crate::dsp::filter::CascadedBiquad;
                    use crate::types::{Hertz, SampleRate};
                    let cutoff = Hertz::new(input.rate as f64 / (2.0 * *factor as f64) * 0.9);
                    CascadedBiquad::butterworth_lowpass(cutoff, SampleRate::new(input.rate), 8)
                        .map_err(|e| format!("decimate anti-alias filter: {e}"))?
                        .sections()
                        .into_iter()
                        .map(|s| [s.b0, s.b1, s.b2, 1.0, s.a1, s.a2])
                        .collect()
                };
                let chunks = (!sections.is_empty()).then(|| IirChunks::plan(&sections, input.len));
                Self::Decimate {
                    factor: *factor,
                    sections,
                    chunks,
                }
            }
            Self::Iir { sections, .. } => Self::Iir {
                sections: sections.clone(),
                chunks: (!sections.is_empty()).then(|| IirChunks::plan(sections, input.len)),
            },
            Self::BandPower { band, .. } => {
                let (size, n_bins) = if input.is_spectrum {
                    if input.len < 2 {
                        return Err("band power needs a spectrum of at least 2 bins".into());
                    }
                    ((input.len - 1) * 2, input.len)
                } else {
                    let size = input.len.next_power_of_two();
                    if size > MAX_DFT_SIZE {
                        return Err(format!(
                            "band power over {} samples needs a {size}-point DFT, above the GPU kernel's 65536",
                            input.len
                        ));
                    }
                    (size, size / 2 + 1)
                };
                let df = input.default_rate as f64 / size as f64;
                let low = ceil_to_usize(band.low.0 / df).min(n_bins - 1);
                let high = floor_to_usize(band.high.0 / df).min(n_bins - 1);
                let (low, high) = if low >= high { (1, 0) } else { (low, high) };
                Self::BandPower {
                    band: *band,
                    bins: Some(BandBins {
                        from_spectrum: input.is_spectrum,
                        size: size as u32,
                        low: low as u32,
                        high: high as u32,
                        df,
                    }),
                }
            }
            Self::CrossCorrelate { max_lag, .. } => {
                let len_b = input
                    .second_len
                    .ok_or("cross-correlation needs a second input signal")?;
                if len_b == 0 {
                    return Err("cross-correlation needs a non-empty second signal".into());
                }
                let max_lag = if *max_lag == usize::MAX {
                    input.len.max(len_b) - 1
                } else {
                    *max_lag
                };
                if max_lag > (u32::MAX / 4) as usize {
                    return Err(format!("cross-correlation max lag {max_lag} is too large"));
                }
                Self::CrossCorrelate { max_lag, len_b }
            }
            other => other.clone(),
        })
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
            Self::Detrend { .. } => "detrend",
            Self::Envelope => "envelope",
            Self::Decimate { .. } => "decimate",
            Self::BandPower { .. } => "band_power",
            Self::CrossCorrelate { .. } => "cross_correlate",
        }
    }

    /// Number of `f32` values the step produces for an `n`-sample input.
    pub fn output_len(&self, n: usize) -> usize {
        match self {
            Self::Diff => n.saturating_sub(1),
            Self::Spectrum { size, .. } => *size as usize / 2 + 1,
            Self::Stats { .. } => STATS_LEN,
            Self::BandPower { .. } => 1,
            Self::Decimate { factor, .. } if *factor > 1 => n.div_ceil(*factor),
            Self::CrossCorrelate { max_lag, .. } => 2 * max_lag + 1,
            _ => n,
        }
    }

    /// Size of the output buffer in `f32`s: the output, then any workspace
    /// the kernel needs after it.
    pub fn buffer_len(&self, n: usize) -> usize {
        match self {
            // Complex FFT workspace of the padded length.
            Self::Envelope => n + 2 * n.max(1).next_power_of_two(),
            // Per-chunk local end states and true entry states.
            Self::Iir {
                sections,
                chunks: Some(c),
            }
            | Self::Decimate {
                sections,
                chunks: Some(c),
                ..
            } => self.output_len(n) + 2 * (c.chunks + c.blocks()) as usize * 2 * sections.len(),
            _ => self.output_len(n),
        }
    }

    /// Re-split a bound chunked IIR / decimate kernel into about `target`
    /// chunks. Adapters with few lanes (a CPU rasterizer such as lavapipe)
    /// run faster with fewer, longer chunks; other kernels are unchanged.
    pub fn with_iir_chunk_target(self, n: usize, target: usize) -> Self {
        match self {
            Self::Iir { sections, chunks: Some(_) } => {
                let chunks = Some(IirChunks::with_chunks(&sections, n, target));
                Self::Iir { sections, chunks }
            }
            Self::Decimate { factor, sections, chunks: Some(_) } => {
                let chunks = Some(IirChunks::with_chunks(&sections, n, target));
                Self::Decimate { factor, sections, chunks }
            }
            other => other,
        }
    }

    /// Dispatches, in order, as `(entry point, workgroups)`. Each sees the
    /// previous one's writes (WebGPU orders dispatches on shared buffers).
    pub fn passes(&self, n: usize) -> Vec<(&'static str, u32)> {
        match self {
            Self::Iir { chunks: Some(c), .. } | Self::Decimate { chunks: Some(c), .. } => {
                let groups = c.chunks.div_ceil(WORKGROUP);
                let mut passes = vec![("chunk_state", groups), ("scan_block", groups)];
                let block_groups = c.blocks().div_ceil(WORKGROUP);
                passes.extend(
                    IIR_SCAN_ENTRIES[..c.levels as usize]
                        .iter()
                        .map(|&entry| (entry, block_groups)),
                );
                passes.push(("main", groups));
                passes
            }
            _ => vec![("main", self.workgroups(n))],
        }
    }

    /// Sample rate of the output signal for an input at `rate`.
    pub fn output_rate(&self, rate: u32) -> u32 {
        match self {
            Self::Decimate { factor, .. } if *factor > 0 => rate / *factor as u32,
            _ => rate,
        }
    }

    /// The kernel produces a scalar (see [`GpuKernel::finish_scalar`]).
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Stats { .. } | Self::BandPower { .. })
    }

    /// The kernel binds a second signal register at binding 3.
    pub fn takes_second_input(&self) -> bool {
        matches!(self, Self::CrossCorrelate { .. })
    }

    /// Minimum input length the CPU runtime accepts for this step.
    pub fn min_input_len(&self) -> usize {
        match self {
            Self::ZScore => 2,
            Self::Stats { op: ReduceOp::Std } => 2,
            Self::Stats { op: ReduceOp::Kurtosis } => 4,
            Self::Detrend { order } if *order > 0 => 2,
            _ => 1,
        }
    }

    /// Workgroups to dispatch. Kernels that need the whole signal run in one
    /// workgroup and stride over it; per-sample kernels cover it with many.
    pub fn workgroups(&self, n: usize) -> u32 {
        let per_sample = |len: usize| (len.max(1) as u32).div_ceil(WORKGROUP);
        match self {
            Self::ElementWise { .. } | Self::Fir { .. } => per_sample(n),
            // Without sections the filter is a copy (strided for decimate).
            Self::Iir { sections, .. } | Self::Decimate { sections, .. } if sections.is_empty() => {
                per_sample(n)
            }
            Self::Diff => per_sample(n.saturating_sub(1)),
            Self::Spectrum { .. } | Self::CrossCorrelate { .. } => per_sample(self.output_len(n)),
            Self::Cumsum
            | Self::ZScore
            | Self::Iir { .. }
            | Self::Stats { .. }
            | Self::Detrend { .. }
            | Self::Envelope
            | Self::Decimate { .. }
            | Self::BandPower { .. } => 1,
        }
    }

    /// Contents of the `aux` binding (never empty: WebGPU rejects 0-byte bindings).
    /// Kernels that [take a second input](GpuKernel::takes_second_input) bind
    /// that register there instead.
    pub fn aux(&self) -> Vec<f32> {
        let v: Vec<f32> = match self {
            Self::Fir { taps } => taps.iter().map(|&t| t as f32).collect(),
            Self::Spectrum { window, .. } => window.iter().map(|&w| w as f32).collect(),
            Self::Iir { chunks: Some(c), .. } | Self::Decimate { chunks: Some(c), .. } => {
                c.powers.iter().map(|&m| m as f32).collect()
            }
            _ => Vec::new(),
        };
        if v.is_empty() {
            alloc_vec_one()
        } else {
            v
        }
    }

    /// The `params` uniform: `[n, aux_len, size, scale_bits, p4, p5, p6, p7]`.
    pub fn params(&self, n: usize) -> [u32; PARAMS_LEN] {
        let mut p = [n as u32, 0, 0, 0, 0, 0, 0, 0];
        match self {
            Self::Fir { taps } => p[1] = taps.len() as u32,
            Self::Spectrum {
                size,
                window,
                coherent_gain,
            } => {
                let scale = 1.0 / (coherent_gain * libm_sqrt(*size as f64));
                p[1] = window.len() as u32;
                p[2] = *size;
                p[3] = (scale as f32).to_bits();
            }
            Self::Detrend { order } => p[2] = u32::from(*order > 0),
            Self::Envelope => {
                let size = n.max(1).next_power_of_two();
                p[2] = size as u32;
                p[1] = size.trailing_zeros();
            }
            Self::Iir { chunks, .. } | Self::Decimate { chunks, .. } => {
                if let Self::Decimate { factor, .. } = self {
                    p[2] = (*factor).max(1) as u32;
                } else {
                    p[2] = 1;
                }
                if let Some(c) = chunks {
                    p[4] = c.chunks;
                    p[5] = c.chunk_len;
                    p[6] = self.output_len(n) as u32;
                }
            }
            Self::BandPower { bins: Some(b), .. } => {
                p[1] = u32::from(b.from_spectrum);
                p[2] = b.size;
                p[3] = (b.df as f32).to_bits();
                p[4] = b.low;
                p[5] = b.high;
                let scale = 1.0 / libm_sqrt(b.size as f64);
                p[6] = (scale as f32).to_bits();
            }
            Self::CrossCorrelate { max_lag, len_b } => {
                p[1] = *len_b as u32;
                p[2] = *max_lag as u32;
            }
            _ => {}
        }
        p
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
            Self::CrossCorrelate { .. } => s.push_str(&per_sample_main(XCORR_BODY)),
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
            Self::Detrend { .. } => {
                s.push_str(&reduce_helpers());
                s.push_str(DETREND_MAIN);
            }
            Self::BandPower { .. } => {
                s.push_str(&reduce_helpers());
                s.push_str(BAND_POWER_MAIN);
            }
            Self::Envelope => s.push_str(ENVELOPE_MAIN),
            Self::Iir { sections, chunks } | Self::Decimate { sections, chunks, .. } => {
                s.push_str(&iir_parallel(
                    sections,
                    IirCarry::Scan(chunks.as_ref().map_or(0, |c| c.levels)),
                ))
            }
        }
        s
    }

    /// Turn a scalar kernel's output into the value the CPU runtime reports
    /// for an `n`-sample input.
    #[cfg(feature = "std")]
    pub fn finish_scalar(
        &self,
        data: &[f64],
        n: usize,
    ) -> Option<crate::types::UncertainValue<f64>> {
        match self {
            Self::Stats { op } => reduce_from_stats(*op, data, n),
            Self::BandPower { .. } => Some(crate::types::UncertainValue::from_ci(
                *data.first()?,
                0.0,
                0.95,
                1,
            )),
            _ => None,
        }
    }
}

fn ceil_to_usize(x: f64) -> usize {
    // `f64::ceil` needs std; saturating casts match `x.ceil() as usize`.
    let t = x as usize;
    if (t as f64) < x {
        t + 1
    } else {
        t
    }
}

fn floor_to_usize(x: f64) -> usize {
    x as usize
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
    let [sum, sum_sq, max, min, crossings, dev2, sq_dev2, dev3, dev4, xy] = [
        stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7], stats[8],
        stats[9],
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
        ReduceOp::Skewness => {
            // Runtime::reduce: population moments, 0 for a constant signal.
            let std = (dev2 / nf).sqrt();
            let skew = if std == 0.0 {
                0.0
            } else {
                dev3 / nf / (std * std * std)
            };
            (skew, generic_se)
        }
        ReduceOp::Kurtosis => {
            // dsp::statistics::compute_kurtosis: excess kurtosis, se sqrt(24/n).
            if n < 4 {
                return None;
            }
            let m2 = dev2 / nf;
            (dev4 / nf / (m2 * m2) - 3.0, (24.0 / nf).sqrt())
        }
        ReduceOp::Slope => {
            // Least-squares slope against the sample index; the index
            // spread sum (i - (n-1)/2)^2 is n(n^2-1)/12 exactly.
            let den = nf * (nf * nf - 1.0) / 12.0;
            (if den == 0.0 { 0.0 } else { xy / den }, generic_se)
        }
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
    p4: u32,
    p5: u32,
    p6: u32,
    p7: u32,
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
    let x_mean = f32(n - 1u) * 0.5;
    var d: f32 = 0.0;
    var q: f32 = 0.0;
    var d3: f32 = 0.0;
    var d4: f32 = 0.0;
    var xy: f32 = 0.0;
    for (var i = lid; i < n; i += 64u) {
        let x = input[i];
        let e = x - mean;
        let e2 = e * e;
        d += e2;
        q += (x * x - mean_sq) * (x * x - mean_sq);
        d3 += e2 * e;
        d4 += e2 * e2;
        xy += (f32(i) - x_mean) * e;
    }
    let dev2 = wg_sum(lid, d);
    let sq_dev2 = wg_sum(lid, q);
    let dev3 = wg_sum(lid, d3);
    let dev4 = wg_sum(lid, d4);
    let slope_num = wg_sum(lid, xy);
    if (lid == 0u) {
        output[7] = dev3;
        output[8] = dev4;
        output[9] = slope_num;
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

const XCORR_BODY: &str = "    let max_lag = params.size;
    if (i <= 2u * max_lag) {
        // lag = i - max_lag; c[lag] = sum_t a[t + lag] * b[t] over valid t.
        let lag = i32(i) - i32(max_lag);
        let na = i32(params.n);
        let nb = i32(params.aux_len);
        let t0 = max(0, -lag);
        let t1 = min(nb, na - lag);
        var acc: f32 = 0.0;
        for (var t = t0; t < t1; t++) {
            acc += input[u32(t + lag)] * aux[u32(t)];
        }
        output[i] = acc;
    }
";

const DETREND_MAIN: &str = "@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n = params.n;
    let nf = f32(n);
    var s: f32 = 0.0;
    for (var i = lid; i < n; i += 64u) { s += input[i]; }
    let mean = wg_sum(lid, s) / nf;
    let x_mean = (nf - 1.0) * 0.5;
    var xy: f32 = 0.0;
    for (var i = lid; i < n; i += 64u) { xy += (f32(i) - x_mean) * (input[i] - mean); }
    let num = wg_sum(lid, xy);
    // sum (i - x_mean)^2 = n (n^2 - 1) / 12
    let slope = select(0.0, num / (nf * (nf * nf - 1.0) / 12.0), params.size != 0u);
    let intercept = mean - slope * x_mean;
    for (var i = lid; i < n; i += 64u) {
        output[i] = input[i] - (intercept + slope * f32(i));
    }
}
";

const BAND_POWER_MAIN: &str = "@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let size = params.size;
    let low = params.p4;
    let high = params.p5;
    let scale = bitcast<f32>(params.p6);
    var acc: f32 = 0.0;
    for (var k = low + lid; k <= high; k += 64u) {
        if (params.aux_len == 1u) {
            // Input is already a magnitude spectrum.
            acc += input[k] * input[k];
        } else {
            // Rectangular-window DFT bin of the zero-padded signal.
            let len = min(params.n, size);
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var t = 0u; t < len; t++) {
                let theta = -6.2831853071795864 * f32((k * t) % size) / f32(size);
                re += input[t] * cos(theta);
                im += input[t] * sin(theta);
            }
            let mag = sqrt(re * re + im * im) * scale;
            acc += mag * mag;
        }
    }
    let total = wg_sum(lid, acc);
    if (lid == 0u) {
        output[0] = total * bitcast<f32>(params.scale_bits);
    }
}
";

/// Radix-2 FFT envelope in one workgroup. The complex workspace lives in the
/// output buffer after the `n` output samples; `storageBarrier` orders the
/// butterfly stages.
const ENVELOPE_MAIN: &str = "fn bit_rev(i: u32, bits: u32) -> u32 {
    if (bits == 0u) { return 0u; }
    return reverseBits(i) >> (32u - bits);
}

fn ws_get(base: u32, j: u32) -> vec2<f32> {
    return vec2<f32>(output[base + 2u * j], output[base + 2u * j + 1u]);
}

fn ws_set(base: u32, j: u32, v: vec2<f32>) {
    output[base + 2u * j] = v.x;
    output[base + 2u * j + 1u] = v.y;
}

fn fft_stages(lid: u32, base: u32, size: u32, sign: f32) {
    for (var len = 2u; len <= size; len = len << 1u) {
        let half = len >> 1u;
        for (var b = lid; b < size / 2u; b += 64u) {
            let k = b % half;
            let i = (b / half) * len + k;
            let j = i + half;
            let theta = sign * 6.2831853071795864 * f32(k) / f32(len);
            let w = vec2<f32>(cos(theta), sin(theta));
            let u = ws_get(base, i);
            let v = ws_get(base, j);
            let t = vec2<f32>(w.x * v.x - w.y * v.y, w.x * v.y + w.y * v.x);
            ws_set(base, i, u + t);
            ws_set(base, j, u - t);
        }
        storageBarrier();
    }
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n = params.n;
    let size = params.size;
    let bits = params.aux_len;
    let base = n;
    // Load in bit-reversed order, zero-padded to `size`.
    for (var i = lid; i < size; i += 64u) {
        let x = select(0.0, input[min(i, n - 1u)], i < n);
        ws_set(base, bit_rev(i, bits), vec2<f32>(x, 0.0));
    }
    storageBarrier();
    fft_stages(lid, base, size, -1.0);
    // Analytic signal: keep DC and Nyquist, double positive, zero negative.
    let half = size / 2u;
    for (var k = lid; k < size; k += 64u) {
        if (k > half) {
            ws_set(base, k, vec2<f32>(0.0, 0.0));
        } else if (k > 0u && k < half) {
            ws_set(base, k, 2.0 * ws_get(base, k));
        }
    }
    storageBarrier();
    for (var i = lid; i < size; i += 64u) {
        let j = bit_rev(i, bits);
        if (i < j) {
            let a = ws_get(base, i);
            ws_set(base, i, ws_get(base, j));
            ws_set(base, j, a);
        }
    }
    storageBarrier();
    fft_stages(lid, base, size, 1.0);
    let inv = 1.0 / f32(size);
    for (var t = lid; t < n; t += 64u) {
        output[t] = length(ws_get(base, t)) * inv;
    }
}
";

/// The single-invocation IIR shader this module used before the chunked
/// scan (same bindings, entry `main`, dispatch 1 workgroup). Kept as the
/// baseline for benchmarks; the executor never uses it.
pub fn sequential_iir_wgsl(sections: &[[f64; 6]]) -> String {
    let mut s = String::from("// SigQL GPU kernel: iir (sequential baseline)\n");
    s.push_str(HEADER);
    s.push_str(&iir_main(sections, false));
    s
}

/// How the chunked IIR shader finds each chunk's entry state.
#[derive(Debug, Clone, Copy)]
enum IirCarry {
    /// One invocation walks the chunks (the Phase C shader; benchmarks only).
    Serial,
    /// Prefix scan over this many levels (see [`IirChunks`]).
    Scan(u32),
}

/// The chunked IIR shader with the single-invocation `carry` pass it used
/// before the prefix scan (entries `chunk_state`, `carry`, `main`; dispatch
/// `carry` as one workgroup). Same bindings, params and aux as the scan
/// kernel (it reads only the level-0 matrix). Kept for benchmarks.
pub fn serial_carry_iir_wgsl(sections: &[[f64; 6]]) -> String {
    let mut s = String::from("// SigQL GPU kernel: iir (serial carry baseline)\n");
    s.push_str(HEADER);
    s.push_str(&iir_parallel(sections, IirCarry::Serial));
    s
}

/// Chunked parallel IIR / decimate (see [`IirChunks`]). `params.size` is the
/// output stride (1 for a plain filter); `p4` chunks, `p5` chunk length,
/// `p6` workspace offset (two `chunks x D` regions); `aux` holds the matrix
/// powers.
fn iir_parallel(sections: &[[f64; 6]], carry: IirCarry) -> String {
    let store = "        if (i % params.size == 0u) { output[i / params.size] = y; }\n";
    if sections.is_empty() {
        return per_sample_main(&format!(
            "    if (i < params.n) {{\n        let y = input[i];\n{store}    }}\n"
        ));
    }
    let d = 2 * sections.len();
    let mut s = format!("const D: u32 = {d}u;\n\nfn filt(x: f32, s: ptr<function, array<f32, {d}>>) -> f32 {{\n    var y = x;\n");
    for (k, sec) in sections.iter().enumerate() {
        let [b0, b1, b2, _a0, a1, a2] = *sec;
        let (z1, z2) = (2 * k, 2 * k + 1);
        s.push_str(&format!(
            "    let x{k} = y;\n    y = {b0} * x{k} + (*s)[{z1}];\n    (*s)[{z1}] = {b1} * x{k} - {a1} * y + (*s)[{z2}];\n    (*s)[{z2}] = {b2} * x{k} - {a2} * y;\n",
            b0 = wgsl_f32(b0),
            b1 = wgsl_f32(b1),
            b2 = wgsl_f32(b2),
            a1 = wgsl_f32(a1),
            a2 = wgsl_f32(a2),
        ));
    }
    s.push_str("    return y;\n}\n\n");
    s.push_str(&format!(
        "@compute @workgroup_size({WORKGROUP})
fn chunk_state(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let c = gid.x;
    if (c >= params.p4) {{ return; }}
    var s: array<f32, {d}>;
    let start = c * params.p5;
    let end = min(params.n, start + params.p5);
    for (var i = start; i < end; i++) {{ _ = filt(input[i], &s); }}
    let base = params.p6 + c * D;
    for (var j = 0u; j < D; j++) {{ output[base + j] = s[j]; }}
}}

"
    ));
    // Where `main` reads chunk c's entry state from.
    let entry_state: String = match carry {
        IirCarry::Serial => {
            s.push_str(&format!(
                "@compute @workgroup_size(1)
fn carry() {{
    let local = params.p6;
    let init = params.p6 + params.p4 * D;
    var st: array<f32, {d}>;
    for (var c = 0u; c < params.p4; c++) {{
        for (var j = 0u; j < D; j++) {{ output[init + c * D + j] = st[j]; }}
        var next: array<f32, {d}>;
        for (var r = 0u; r < D; r++) {{
            var acc = output[local + c * D + r];
            for (var q = 0u; q < D; q++) {{ acc += aux[r * D + q] * st[q]; }}
            next[r] = acc;
        }}
        st = next;
    }}
}}

"
            ));
            "    let init = params.p6 + params.p4 * D + c * D;
    for (var j = 0u; j < D; j++) { s[j] = output[init + j]; }
"
            .into()
        }
        IirCarry::Scan(levels) => {
            let block = IIR_SCAN_BLOCK;
            let dd = d * d;
            // Regions: R0 local states, R1 block prefixes (chunks x D each),
            // T0/T1 block totals (blocks x D each).
            s.push_str(&format!(
                "const BLOCK: u32 = {block}u;
fn r1() -> u32 {{ return params.p6 + params.p4 * D; }}
fn t_base(k: u32) -> u32 {{
    let blocks = (params.p4 + BLOCK - 1u) / BLOCK;
    return params.p6 + 2u * params.p4 * D + k * blocks * D;
}}

var<workgroup> tile: array<f32, {tile}>;

@compute @workgroup_size({WORKGROUP})
fn scan_block(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let c = gid.x;
    let t = lid.x;
    let live = c < params.p4;
    for (var j = 0u; j < D; j++) {{
        var v = 0.0;
        if (live) {{ v = output[params.p6 + c * D + j]; }}
        tile[t * D + j] = v;
    }}
    workgroupBarrier();
    for (var shift = 1u; shift < BLOCK; shift = shift * 2u) {{
        var v: array<f32, {d}>;
        for (var r = 0u; r < D; r++) {{
            var acc = tile[t * D + r];
            if (t >= shift) {{
                // M^shift is entry shift - 1 of the power table.
                let pow = (shift - 1u) * {dd}u;
                for (var q = 0u; q < D; q++) {{ acc += aux[pow + r * D + q] * tile[(t - shift) * D + q]; }}
            }}
            v[r] = acc;
        }}
        workgroupBarrier();
        for (var r = 0u; r < D; r++) {{ tile[t * D + r] = v[r]; }}
        workgroupBarrier();
    }}
    if (live) {{
        for (var r = 0u; r < D; r++) {{ output[r1() + c * D + r] = tile[t * D + r]; }}
    }}
    if (t == BLOCK - 1u) {{
        for (var r = 0u; r < D; r++) {{ output[t_base(0u) + wid.x * D + r] = tile[t * D + r]; }}
    }}
}}

",
                tile = block * d,
            ));
            for k in 0..levels {
                let (src, dst) = (k % 2, (k + 1) % 2);
                let shift = 1u32 << k;
                let pow = (block + k as usize) * dd;
                s.push_str(&format!(
                    "@compute @workgroup_size({WORKGROUP})
fn scan_{k}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let b = gid.x;
    let blocks = (params.p4 + BLOCK - 1u) / BLOCK;
    if (b >= blocks) {{ return; }}
    let src = t_base({src}u);
    let dst = t_base({dst}u);
    for (var r = 0u; r < D; r++) {{
        var acc = output[src + b * D + r];
        if (b >= {shift}u) {{
            let prev = src + (b - {shift}u) * D;
            for (var q = 0u; q < D; q++) {{ acc += aux[{pow}u + r * D + q] * output[prev + q]; }}
        }}
        output[dst + b * D + r] = acc;
    }}
}}

"
                ));
            }
            // Chunk c enters with X[e], e = c - 1: e's block prefix plus
            // M^(e % BLOCK + 1) times the total of every earlier block
            // (inclusive block scan, region T[levels % 2], at block - 1).
            format!(
                "    if (c > 0u) {{
        let e = c - 1u;
        let b = e / BLOCK;
        for (var j = 0u; j < D; j++) {{ s[j] = output[r1() + e * D + j]; }}
        if (b > 0u) {{
            let tot = t_base({final_region}u) + (b - 1u) * D;
            let pow = (e % BLOCK) * {dd}u;
            for (var r = 0u; r < D; r++) {{
                var acc = s[r];
                for (var q = 0u; q < D; q++) {{ acc += aux[pow + r * D + q] * output[tot + q]; }}
                s[r] = acc;
            }}
        }}
    }}
",
                final_region = levels % 2
            )
        }
    };
    s.push_str(&format!(
        "@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let c = gid.x;
    if (c >= params.p4) {{ return; }}
    var s: array<f32, {d}>;
{entry_state}    let start = c * params.p5;
    let end = min(params.n, start + params.p5);
    for (var i = start; i < end; i++) {{
        let y = filt(input[i], &s);
{store}    }}
}}
"
    ));
    s
}

fn iir_main(sections: &[[f64; 6]], decimate: bool) -> String {
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
    if decimate {
        // Keep every `params.size`-th filtered sample.
        s.push_str("        if (i % params.size == 0u) {\n            output[i / params.size] = y;\n        }\n    }\n}\n");
    } else {
        s.push_str("        output[i] = y;\n    }\n}\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::plan::{FirCoeffs, IirCoeffs, RegisterId, WindowCoeffs};
    use crate::types::FrequencyBand;

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
        for op in [ReduceOp::Skewness, ReduceOp::Kurtosis, ReduceOp::Slope] {
            steps.push(PlanStep::Reduce {
                input: r,
                output: r,
                op,
            });
        }
        for order in [0, 1] {
            steps.push(PlanStep::Detrend {
                input: r,
                output: r,
                order,
            });
        }
        steps.push(PlanStep::Envelope {
            input: r,
            output: r,
        });
        for factor in [1, 4] {
            steps.push(PlanStep::Decimate {
                input: r,
                output: r,
                factor,
            });
        }
        steps.push(PlanStep::CrossCorrelate {
            input_a: r,
            input_b: r,
            output: r,
            max_lag_samples: 20,
        });
        let input = GpuInput {
            len: 300,
            rate: 1000,
            is_spectrum: false,
            second_len: Some(250),
            default_rate: 1000,
        };
        let mut kernels: Vec<GpuKernel> = steps
            .iter()
            .map(|s| GpuKernel::from_step(s).unwrap().unwrap().bind(&input).unwrap())
            .collect();
        let band = PlanStep::BandPower {
            input: r,
            output: r,
            band: FrequencyBand::new(4.0, 12.0),
        };
        let bp = GpuKernel::from_step(&band).unwrap().unwrap();
        kernels.push(bp.bind(&input).unwrap());
        kernels.push(
            bp.bind(&GpuInput {
                len: 129,
                is_spectrum: true,
                ..input
            })
            .unwrap(),
        );
        kernels
    }

    /// Every kernel the GPU path can emit must parse and validate as WGSL
    /// (naga is the compiler wgpu uses), so a shader bug fails the build
    /// instead of silently pushing queries onto the CPU.
    fn validate_wgsl(name: &str, src: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} does not parse:\n{}\n{src}", e.emit_to_string(src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} does not validate: {e:?}\n{src}"));
        module
    }

    /// The chunked IIR kernels expose every entry point their passes
    /// dispatch, and the sequential baseline still compiles.
    #[test]
    fn iir_passes_name_real_entry_points() {
        let sections = crate::dsp::filter::CascadedBiquad::butterworth_lowpass(
            crate::types::Hertz::new(50.0),
            crate::types::SampleRate::new(1000),
            4,
        )
        .unwrap()
        .sections()
        .into_iter()
        .map(|s| [s.b0, s.b1, s.b2, 1.0, s.a1, s.a2])
        .collect::<Vec<_>>();
        let input = GpuInput {
            len: 100_000,
            rate: 1000,
            is_spectrum: false,
            second_len: None,
            default_rate: 1000,
        };
        for kernel in [
            GpuKernel::Iir {
                sections: sections.clone(),
                chunks: None,
            },
            GpuKernel::Decimate {
                factor: 4,
                sections: Vec::new(),
                chunks: None,
            },
        ] {
            let bound = kernel.bind(&input).unwrap();
            let module = validate_wgsl(bound.name(), &bound.wgsl());
            let passes = bound.passes(input.len);
            let levels = match &bound {
                GpuKernel::Iir { chunks: Some(c), .. } | GpuKernel::Decimate { chunks: Some(c), .. } => c.levels,
                other => panic!("{other:?} is not chunked"),
            };
            assert!(levels >= 1);
            assert_eq!(passes.len(), 3 + levels as usize, "{}", bound.name());
            for (entry, groups) in passes {
                assert!(groups >= 1);
                assert!(
                    module.entry_points.iter().any(|e| e.name == entry),
                    "{} lacks {entry}",
                    bound.name()
                );
            }
        }
        validate_wgsl("sequential iir", &sequential_iir_wgsl(&sections));
        let serial = validate_wgsl("serial carry iir", &serial_carry_iir_wgsl(&sections));
        for entry in ["chunk_state", "carry", "main"] {
            assert!(serial.entry_points.iter().any(|e| e.name == entry), "{entry}");
        }
    }

    /// Emulating the three passes in f64 reproduces the sequential cascade:
    /// the carry matrix is exactly the zero-input chunk map.
    #[test]
    fn chunked_scan_reproduces_sequential_cascade() {
        let sections = crate::dsp::filter::CascadedBiquad::butterworth_lowpass(
            crate::types::Hertz::new(20.0),
            crate::types::SampleRate::new(1000),
            8,
        )
        .unwrap()
        .sections()
        .into_iter()
        .map(|s| [s.b0, s.b1, s.b2, 1.0, s.a1, s.a2])
        .collect::<Vec<_>>();
        let d = 2 * sections.len();
        for (n, target) in [
            (1usize, None),
            (7, None),
            (100, None),
            (4097, None),
            (50_000, None),
            (50_000, Some(3)),
            (50_000, Some(1000)),
            (50_000, Some(64)),
            (50_000, Some(65)),
            (1 << 16, Some(1 << 16)),
            (1 << 17, Some(1 << 16)),
        ] {
            let x: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.013).sin() + 0.3 * (i as f64 * 0.41).cos())
                .collect();
            let mut st = vec![0.0; d];
            let want: Vec<f64> = x.iter().map(|&v| cascade_step(&sections, v, &mut st)).collect();

            let plan = match target {
                Some(t) => IirChunks::with_chunks(&sections, n, t),
                None => IirChunks::plan(&sections, n),
            };
            let (p, l) = (plan.chunks as usize, plan.chunk_len as usize);
            assert!(p * l >= n && (p - 1) * l < n, "n={n} p={p} l={l}");
            let local: Vec<Vec<f64>> = (0..p)
                .map(|c| {
                    let mut s = vec![0.0; d];
                    for &v in &x[c * l..((c + 1) * l).min(n)] {
                        cascade_step(&sections, v, &mut s);
                    }
                    s
                })
                .collect();
            let blocks = plan.blocks() as usize;
            assert_eq!(blocks, p.div_ceil(IIR_SCAN_BLOCK));
            assert!(1usize << plan.levels >= blocks);
            assert!(plan.levels == 0 || 1usize << (plan.levels - 1) < blocks);
            let dd = d * d;
            let matvec = |m: &[f64], v: &[f64]| -> Vec<f64> {
                (0..d).map(|r| (0..d).map(|q| m[r * d + q] * v[q]).sum()).collect()
            };
            let add = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
            // scan_block: Hillis-Steele inside each block with M^shift.
            let mut prefix = local.clone();
            let mut shift = 1;
            while shift < IIR_SCAN_BLOCK {
                let pow = &plan.powers[(shift - 1) * dd..shift * dd];
                prefix = (0..p)
                    .map(|c| {
                        if c % IIR_SCAN_BLOCK >= shift {
                            add(&prefix[c], &matvec(pow, &prefix[c - shift]))
                        } else {
                            prefix[c].clone()
                        }
                    })
                    .collect();
                shift *= 2;
            }
            // Block totals (last chunk of each full block), then scan_k.
            let mut totals: Vec<Vec<f64>> = (0..blocks)
                .map(|b| prefix[((b + 1) * IIR_SCAN_BLOCK - 1).min(p - 1)].clone())
                .collect();
            for k in 0..plan.levels as usize {
                let pow = &plan.powers[(IIR_SCAN_BLOCK + k) * dd..(IIR_SCAN_BLOCK + k + 1) * dd];
                let shift = 1 << k;
                totals = (0..blocks)
                    .map(|b| if b >= shift { add(&totals[b], &matvec(pow, &totals[b - shift])) } else { totals[b].clone() })
                    .collect();
            }
            // main's entry state.
            let init: Vec<Vec<f64>> = (0..p)
                .map(|c| {
                    if c == 0 {
                        return vec![0.0; d];
                    }
                    let e = c - 1;
                    let b = e / IIR_SCAN_BLOCK;
                    if b == 0 {
                        prefix[e].clone()
                    } else {
                        let off = e % IIR_SCAN_BLOCK;
                        add(&prefix[e], &matvec(&plan.powers[off * dd..(off + 1) * dd], &totals[b - 1]))
                    }
                })
                .collect();
            let mut got = Vec::with_capacity(n);
            for (c, s0) in init.iter().enumerate() {
                let mut s = s0.clone();
                for &v in &x[c * l..((c + 1) * l).min(n)] {
                    got.push(cascade_step(&sections, v, &mut s));
                }
            }
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() <= 1e-9 * (1.0 + b.abs()), "n={n} i={i}: {a} vs {b}");
            }
        }
    }

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
        let err = GpuKernel::from_step(&PlanStep::MedianFilter {
            input: r,
            output: r,
            kernel_size: 5,
        })
        .unwrap_err();
        assert!(err.contains("MedianFilter"), "{err}");
        // Inputs the CPU runtime would reject, or the kernel cannot take,
        // are refused at bind time with a reason.
        let input = GpuInput {
            len: 3,
            rate: 1000,
            is_spectrum: false,
            second_len: None,
            default_rate: 1000,
        };
        let kurt = GpuKernel::Stats {
            op: ReduceOp::Kurtosis,
        };
        assert!(kurt.bind(&input).unwrap_err().contains("at least 4"));
        let xcorr = GpuKernel::CrossCorrelate {
            max_lag: 3,
            len_b: 0,
        };
        assert!(xcorr.bind(&input).unwrap_err().contains("second input"));
        let spectrum_in = GpuInput {
            is_spectrum: true,
            ..input
        };
        assert!(GpuKernel::Envelope
            .bind(&spectrum_in)
            .unwrap_err()
            .contains("time-domain"));
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
            samples.iter().map(|x| (x - mean).powi(3)).sum::<f64>(),
            samples.iter().map(|x| (x - mean).powi(4)).sum::<f64>(),
            samples
                .iter()
                .enumerate()
                .map(|(i, x)| (i as f64 - (n - 1) as f64 / 2.0) * (x - mean))
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
            ReduceOp::Skewness,
            ReduceOp::Kurtosis,
            ReduceOp::Slope,
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
