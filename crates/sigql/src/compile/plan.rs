//! Execution Plan
//!
//! The intermediate representation between AST and executable code.

use smol_str::SmolStr;

use super::Target;
use crate::types::{FrequencyBand, SampleRate};

/// An execution plan ready for a specific target
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub steps: Vec<PlanStep>,
    pub target: Target,
}

impl ExecutionPlan {
    /// Estimate memory requirements in bytes
    ///
    /// Calculates the total memory needed for:
    /// - Input/output buffers
    /// - Intermediate registers
    /// - FFT twiddle factors
    /// - Filter coefficients
    pub fn estimate_memory(&self) -> usize {
        let resources = self.allocate_resources();
        let base_buffer_size = 4096; // Default buffer size in samples
        let bytes_per_sample = 4; // f32

        // Base memory: input + output buffers
        let mut total = base_buffer_size * bytes_per_sample * 2;

        // Register memory
        total += resources.num_registers as usize * base_buffer_size * bytes_per_sample;

        // Additional memory per step type
        for step in &self.steps {
            total += self.estimate_step_memory(step, base_buffer_size);
        }

        total
    }

    /// Estimate memory for a single step
    fn estimate_step_memory(&self, step: &PlanStep, buffer_size: usize) -> usize {
        let bytes_per_sample = 4; // f32

        match step {
            PlanStep::Fft { size, .. } => {
                // FFT needs: input buffer, output buffer (complex), twiddle factors
                let complex_size = size * 2 * bytes_per_sample; // re + im
                let twiddle_size = size * 2 * bytes_per_sample;
                complex_size + twiddle_size
            }
            PlanStep::Ifft { .. } => {
                buffer_size * 2 * bytes_per_sample // Complex buffer
            }
            PlanStep::IirFilter { coeffs, .. } => {
                // Coefficients + state variables (2 per section for history)
                coeffs.sections.len() * 6 * bytes_per_sample
                    + coeffs.sections.len() * 2 * bytes_per_sample
            }
            PlanStep::FirFilter { coeffs, .. } => {
                // Coefficients + delay line
                (coeffs.taps.len() * 2) * bytes_per_sample
            }
            PlanStep::Window {
                duration_samples, ..
            } => {
                // Window coefficients
                duration_samples * bytes_per_sample
            }
            PlanStep::CrossCorrelate {
                max_lag_samples, ..
            } => {
                // Output correlation buffer
                max_lag_samples * 2 * bytes_per_sample
            }
            PlanStep::MedianFilter { kernel_size, .. } => {
                // Sorting buffer
                kernel_size * bytes_per_sample
            }
            PlanStep::Resample { .. } => {
                // Resampling filter + buffer
                64 * bytes_per_sample
            }
            _ => 0, // Other operations are in-place or use minimal memory
        }
    }

    /// Estimate compute complexity in floating-point operations (FLOPs)
    ///
    /// Provides an estimate of the computational cost for the given input size.
    /// This helps with:
    /// - Choosing between CPU and GPU execution
    /// - Workload balancing in parallel processing
    /// - Performance prediction
    pub fn estimate_flops(&self, input_samples: usize) -> usize {
        let mut total_flops = 0;

        for step in &self.steps {
            total_flops += self.estimate_step_flops(step, input_samples);
        }

        total_flops
    }

    /// Estimate FLOPs for a single step
    fn estimate_step_flops(&self, step: &PlanStep, n: usize) -> usize {
        match step {
            // O(n log n) operations
            PlanStep::Fft { size, .. } => {
                // Radix-2 FFT: 5 * N * log2(N) operations
                let fft_n = *size;
                5 * fft_n * (fft_n as f64).log2() as usize
            }
            PlanStep::Ifft { .. } => {
                // Same as FFT
                5 * n * (n as f64).log2() as usize
            }

            // O(n * k) operations where k is filter length
            PlanStep::IirFilter { coeffs, .. } => {
                // Each sample: 5 muls + 4 adds per biquad section
                n * coeffs.sections.len() * 9
            }
            PlanStep::FirFilter { coeffs, .. } => {
                // Each sample: taps muls + (taps-1) adds
                n * (coeffs.taps.len() * 2 - 1)
            }
            PlanStep::MedianFilter { kernel_size, .. } => {
                // Approximate: sorting is O(k log k) per sample
                n * kernel_size * ((*kernel_size as f64).log2() as usize + 1)
            }

            // O(n) operations
            PlanStep::ElementWise { op, .. } => {
                match op {
                    ElementWiseOp::Abs | ElementWiseOp::Negate => n,
                    ElementWiseOp::Square | ElementWiseOp::Scale(_) | ElementWiseOp::Offset(_) => n,
                    ElementWiseOp::Sqrt
                    | ElementWiseOp::Log
                    | ElementWiseOp::Log10
                    | ElementWiseOp::Exp => {
                        // Transcendental functions are more expensive
                        n * 10
                    }
                }
            }
            PlanStep::Reduce { op, .. } => {
                // Reduction: n-1 operations for basic reductions
                match op {
                    ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Max | ReduceOp::Min => n,
                    ReduceOp::Rms => n * 2 + 1, // square, sum, sqrt
                    ReduceOp::Variance | ReduceOp::Std => n * 3 + 2, // mean, diff, square, sum, (sqrt)
                    ReduceOp::Kurtosis | ReduceOp::Skewness => n * 5, // higher moments
                    ReduceOp::ZeroCrossings => n,                    // comparisons
                    ReduceOp::Slope => n * 4,                        // linear regression
                    ReduceOp::PeakToPeak => n * 2,                   // min + max
                }
            }
            PlanStep::ZScore { .. } => {
                // Mean + std calculation + normalization
                n * 5
            }
            PlanStep::Diff { .. } => n,
            PlanStep::Cumsum { .. } => n,
            PlanStep::ComplexToMagnitude { .. } => n * 3, // square, add, sqrt
            PlanStep::Envelope { .. } => {
                // Hilbert transform via FFT
                10 * n * (n as f64).log2() as usize
            }
            PlanStep::Detrend { order, .. } => {
                // Polynomial fitting: O(n * order^2) + O(n)
                n * (*order as usize + 1) * (*order as usize + 1) + n
            }
            PlanStep::Resample {
                from_rate, to_rate, ..
            } => {
                // Polyphase filter: depends on rate ratio
                let ratio = to_rate.0 as f64 / from_rate.0 as f64;
                (n as f64 * ratio.max(1.0) * 32.0) as usize // 32-tap filter
            }
            PlanStep::Decimate { factor, .. } => {
                // Anti-aliasing filter + downsampling
                n * 32 / factor // Assumes 32-tap filter
            }
            PlanStep::Interpolate { factor, .. } => {
                // Upsampling + interpolation filter
                n * factor * 8 // Interpolation kernel
            }
            PlanStep::Window {
                duration_samples,
                step_samples,
                ..
            } => {
                // Windows created: (n - duration) / step
                let num_windows = (n.saturating_sub(*duration_samples)) / step_samples.max(&1);
                num_windows * *duration_samples
            }
            PlanStep::CrossCorrelate {
                max_lag_samples, ..
            } => {
                // Direct correlation: O(n * lags)
                n * max_lag_samples * 2
            }
            PlanStep::BandPower { .. } => {
                // Sum of squared magnitudes in band
                n / 4 // Approximate band width
            }
            PlanStep::DominantFrequency { .. } => {
                // Find max in spectrum
                n
            }
            PlanStep::SpectralEntropy { .. } => {
                // Normalize + log + sum
                n * 4
            }
            PlanStep::SpectralCentroid { .. } => {
                // Weighted mean
                n * 3
            }

            // Minimal operations
            PlanStep::LoadSignal { .. } | PlanStep::Store { .. } | PlanStep::Passthrough { .. } => {
                0
            }

            // MediaQL operations — O(n²) for 2D transforms
            PlanStep::Fft2d { .. } | PlanStep::Ifft2d { .. } => {
                // 2D FFT: 2 * N * 5 * N * log2(N) (row + column pass)
                let side = (n as f64).sqrt() as usize;
                2 * side * 5 * side * ((side as f64).log2() as usize).max(1)
            }
            PlanStep::Dct2d { block_size, .. } => {
                // Block DCT: (N/block)² blocks × block² × 2 * block operations
                let blocks = (n + block_size - 1) / block_size;
                blocks * blocks * block_size * block_size * 2 * block_size
            }
            PlanStep::Idct2d { .. } => {
                let side = (n as f64).sqrt() as usize;
                2 * side * 5 * side * ((side as f64).log2() as usize).max(1)
            }
            PlanStep::Mfcc { n_coefficients, .. } => {
                // FFT + mel filterbank + DCT
                5 * n * ((n as f64).log2() as usize).max(1) + n * 40 + 40 * n_coefficients
            }
            PlanStep::PerceptualHash { .. } | PlanStep::EdgeDetect { .. } => n,
        }
    }

    /// Check if this plan's workload is large enough to benefit from GPU
    /// execution (cost model heuristic).
    ///
    /// Returns true if the estimated FLOPs exceed GPU kernel launch overhead
    /// and the plan contains parallelizable operations.
    pub fn benefits_from_gpu(&self, input_samples: usize) -> bool {
        let flops = self.estimate_flops(input_samples);
        let has_parallel_ops = self.steps.iter().any(|s| {
            matches!(
                s,
                PlanStep::Fft { .. }
                    | PlanStep::ElementWise { .. }
                    | PlanStep::FirFilter { .. }
                    | PlanStep::CrossCorrelate { .. }
            )
        });

        // GPU is beneficial for large parallel workloads
        // Threshold: ~100K FLOPs and parallelizable operations
        flops > 100_000 && has_parallel_ops
    }
}

/// Individual execution step
#[derive(Debug, Clone)]
pub enum PlanStep {
    /// Load signal from source
    LoadSignal { source: SmolStr, output: RegisterId },

    /// Apply FFT
    Fft {
        input: RegisterId,
        output: RegisterId,
        size: usize,
        window: WindowCoeffs,
    },

    /// Apply inverse FFT
    Ifft {
        input: RegisterId,
        output: RegisterId,
    },

    /// Apply IIR filter
    IirFilter {
        input: RegisterId,
        output: RegisterId,
        coeffs: IirCoeffs,
    },

    /// Apply FIR filter
    FirFilter {
        input: RegisterId,
        output: RegisterId,
        coeffs: FirCoeffs,
    },

    /// Resample signal
    Resample {
        input: RegisterId,
        output: RegisterId,
        from_rate: SampleRate,
        to_rate: SampleRate,
    },

    /// Compute magnitude/phase from complex
    ComplexToMagnitude {
        input: RegisterId,
        output: RegisterId,
    },

    /// Compute envelope via Hilbert transform
    Envelope {
        input: RegisterId,
        output: RegisterId,
    },

    /// Reduce to scalar
    Reduce {
        input: RegisterId,
        output: RegisterId,
        op: ReduceOp,
    },

    /// Window signal
    Window {
        input: RegisterId,
        output: RegisterId,
        duration_samples: usize,
        step_samples: usize,
    },

    /// Cross-correlation
    CrossCorrelate {
        input_a: RegisterId,
        input_b: RegisterId,
        output: RegisterId,
        max_lag_samples: usize,
    },

    /// Band power extraction
    BandPower {
        input: RegisterId, // Spectrum
        output: RegisterId,
        band: FrequencyBand,
    },

    /// Store result
    Store { input: RegisterId, name: SmolStr },

    /// Z-score normalization
    ZScore {
        input: RegisterId,
        output: RegisterId,
    },

    /// Detrend signal (remove polynomial trend)
    Detrend {
        input: RegisterId,
        output: RegisterId,
        order: u8,
    },

    /// Element-wise operation
    ElementWise {
        input: RegisterId,
        output: RegisterId,
        op: ElementWiseOp,
    },

    /// Differentiate signal
    Diff {
        input: RegisterId,
        output: RegisterId,
    },

    /// Cumulative sum
    Cumsum {
        input: RegisterId,
        output: RegisterId,
    },

    /// Median filter
    MedianFilter {
        input: RegisterId,
        output: RegisterId,
        kernel_size: usize,
    },

    /// Decimate signal (downsample with anti-aliasing)
    Decimate {
        input: RegisterId,
        output: RegisterId,
        factor: usize,
    },

    /// Interpolate signal (upsample)
    Interpolate {
        input: RegisterId,
        output: RegisterId,
        factor: usize,
    },

    /// Find dominant frequency
    DominantFrequency {
        input: RegisterId,
        output: RegisterId,
    },

    /// Compute spectral entropy
    SpectralEntropy {
        input: RegisterId,
        output: RegisterId,
    },

    /// Compute spectral centroid
    SpectralCentroid {
        input: RegisterId,
        output: RegisterId,
    },

    /// Passthrough (no-op, for unimplemented transforms)
    Passthrough {
        input: RegisterId,
        output: RegisterId,
    },

    // ====== MediaQL Plan Steps ======

    /// 2D FFT (image frequency transform)
    Fft2d {
        input: RegisterId,
        output: RegisterId,
    },

    /// Inverse 2D FFT (image reconstruction)
    Ifft2d {
        input: RegisterId,
        output: RegisterId,
    },

    /// 2D DCT (block-based, JPEG-style encoding)
    Dct2d {
        input: RegisterId,
        output: RegisterId,
        block_size: usize,
        quality: u8,
    },

    /// Inverse 2D DCT (reconstruction from frequency coefficients)
    Idct2d {
        input: RegisterId,
        output: RegisterId,
    },

    /// MFCC extraction (audio feature)
    Mfcc {
        input: RegisterId,
        output: RegisterId,
        n_coefficients: usize,
    },

    /// Perceptual hash computation
    PerceptualHash {
        input: RegisterId,
        output: RegisterId,
    },

    /// Edge detection
    EdgeDetect {
        input: RegisterId,
        output: RegisterId,
    },
}

/// Register identifier for intermediate values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterId(pub u32);

/// Pre-computed window coefficients
#[derive(Debug, Clone)]
pub struct WindowCoeffs {
    pub coeffs: Vec<f64>,
    pub coherent_gain: f64,
    pub enbw: f64,
}

/// IIR filter coefficients (biquad sections)
#[derive(Debug, Clone)]
pub struct IirCoeffs {
    /// Second-order sections: each section is [b0, b1, b2, a0, a1, a2]
    pub sections: Vec<[f64; 6]>,
}

/// FIR filter coefficients
#[derive(Debug, Clone)]
pub struct FirCoeffs {
    pub taps: Vec<f64>,
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Variance,
    Rms,
    ZeroCrossings,
    Std,
    Kurtosis,
    Skewness,
    Slope,
    PeakToPeak,
}

/// Element-wise operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementWiseOp {
    Abs,
    Square,
    Sqrt,
    Log,
    Log10,
    Exp,
    Scale(f64),
    Offset(f64),
    Negate,
}

/// Resource allocation for execution
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub num_registers: u32,
    pub max_buffer_size: usize,
    pub requires_fft: bool,
    pub requires_complex: bool,
}

impl ExecutionPlan {
    /// Allocate resources for this plan
    pub fn allocate_resources(&self) -> ResourceAllocation {
        let mut max_reg = 0u32;
        let mut requires_fft = false;
        let mut requires_complex = false;
        let mut max_buffer_size: usize = 1024; // Default minimum

        for step in &self.steps {
            // Calculate buffer requirements for each step type
            let (output_reg, step_buffer) = match step {
                PlanStep::LoadSignal { output, .. } => (Some(output.0), 0),
                PlanStep::Fft { output, size, .. } => {
                    requires_fft = true;
                    requires_complex = true;
                    // FFT requires 2x buffer for complex values + scratch space
                    (Some(output.0), *size * 2 * std::mem::size_of::<f64>() * 2)
                }
                PlanStep::Ifft { output, .. } => (Some(output.0), 0),
                PlanStep::IirFilter { output, coeffs, .. } => {
                    // IIR filter state per section
                    let state_size = coeffs.sections.len() * 4 * std::mem::size_of::<f64>();
                    (Some(output.0), state_size)
                }
                PlanStep::FirFilter { output, coeffs, .. } => {
                    // FIR filter needs delay line of length = num_taps
                    let delay_size = coeffs.taps.len() * std::mem::size_of::<f64>();
                    (Some(output.0), delay_size)
                }
                PlanStep::Resample { output, .. } => {
                    // Resample may need temporary buffer
                    (Some(output.0), 4096 * std::mem::size_of::<f64>())
                }
                PlanStep::ComplexToMagnitude { output, .. } => (Some(output.0), 0),
                PlanStep::Envelope { output, .. } => {
                    // Envelope (Hilbert transform) needs FFT-sized buffer
                    (Some(output.0), 1024 * 2 * std::mem::size_of::<f64>())
                }
                PlanStep::Reduce { output, .. } => (Some(output.0), 0),
                PlanStep::Window {
                    output,
                    duration_samples,
                    ..
                } => {
                    // Window buffer for accumulated samples
                    (
                        Some(output.0),
                        *duration_samples * std::mem::size_of::<f64>(),
                    )
                }
                PlanStep::CrossCorrelate { output, .. } => {
                    // Cross-correlation typically done via FFT
                    (Some(output.0), 4096 * 2 * std::mem::size_of::<f64>())
                }
                PlanStep::BandPower { output, .. } => (Some(output.0), 0),
                PlanStep::Store { .. } => (None, 0),
                PlanStep::ZScore { output, .. } => (Some(output.0), 0),
                PlanStep::Detrend { output, .. } => (Some(output.0), 0),
                PlanStep::ElementWise { output, .. } => (Some(output.0), 0),
                PlanStep::Diff { output, .. } => (Some(output.0), 0),
                PlanStep::Cumsum { output, .. } => (Some(output.0), 0),
                PlanStep::MedianFilter {
                    output,
                    kernel_size,
                    ..
                } => {
                    // Median filter needs sliding window buffer
                    (Some(output.0), *kernel_size * std::mem::size_of::<f64>())
                }
                PlanStep::Decimate { output, factor, .. } => {
                    // Decimation may need anti-aliasing filter
                    (Some(output.0), *factor * 32 * std::mem::size_of::<f64>())
                }
                PlanStep::Interpolate { output, factor, .. } => {
                    // Interpolation needs upsampled buffer
                    (Some(output.0), *factor * 1024 * std::mem::size_of::<f64>())
                }
                PlanStep::DominantFrequency { output, .. } => {
                    // FFT-based analysis
                    (Some(output.0), 1024 * 2 * std::mem::size_of::<f64>())
                }
                PlanStep::SpectralEntropy { output, .. } => {
                    (Some(output.0), 1024 * std::mem::size_of::<f64>())
                }
                PlanStep::SpectralCentroid { output, .. } => {
                    (Some(output.0), 1024 * std::mem::size_of::<f64>())
                }
                PlanStep::Passthrough { output, .. } => (Some(output.0), 0),
                // MediaQL steps
                PlanStep::Fft2d { output, .. } | PlanStep::Ifft2d { output, .. } => {
                    (Some(output.0), 1024 * 1024 * std::mem::size_of::<f64>())
                }
                PlanStep::Dct2d { output, block_size, .. } => {
                    (Some(output.0), block_size * block_size * std::mem::size_of::<f64>())
                }
                PlanStep::Idct2d { output, .. } => {
                    (Some(output.0), 1024 * 1024 * std::mem::size_of::<f64>())
                }
                PlanStep::Mfcc { output, .. } => {
                    (Some(output.0), 2048 * 2 * std::mem::size_of::<f64>())
                }
                PlanStep::PerceptualHash { output, .. } => (Some(output.0), 64 * 64 * 8),
                PlanStep::EdgeDetect { output, .. } => (Some(output.0), 0),
            };

            if let Some(reg) = output_reg {
                max_reg = max_reg.max(reg);
            }
            max_buffer_size = max_buffer_size.max(step_buffer);
        }

        // Each register needs a buffer, add that to total
        let register_buffers = (max_reg as usize + 1) * 4096 * std::mem::size_of::<f64>();
        max_buffer_size = max_buffer_size.max(register_buffers);

        ResourceAllocation {
            num_registers: max_reg + 1,
            max_buffer_size,
            requires_fft,
            requires_complex,
        }
    }
}
