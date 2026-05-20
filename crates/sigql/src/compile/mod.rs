//! SigQL Compiler
//!
//! Transforms AST into executable plans for various backends.

pub mod codegen;
pub mod optimize;
pub mod plan;
pub mod simd_runtime;

pub use plan::*;
pub use simd_runtime::{SimdOp, SimdRuntime};

use crate::ast::{
    AggregateOp, FilterParams, FromClause, Query, TransformOp, WindowFunction, WindowKind,
    WindowSpec,
};
use crate::dsp::filter::BiquadCoeffs;
use crate::dsp::window::WindowType;
use crate::parser::ParseError;
use crate::types::SampleRate;
use smol_str::SmolStr;

/// Compilation target backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Portable SIMD (default, Rust)
    Simd,
    /// WebGPU compute shaders
    WebGpu,
    /// CUDA kernels
    Cuda,
    /// Interpreted (slow, for debugging)
    Interpreted,
}

impl Default for Target {
    fn default() -> Self {
        Self::Interpreted // Use interpreted as default for now
    }
}

/// Compiler configuration
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub target: Target,
    pub optimize: bool,
    pub debug_info: bool,
    /// Default sample rate for signals without explicit rate
    pub default_sample_rate: u32,
    /// Default FFT size
    pub default_fft_size: usize,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            target: Target::Interpreted,
            optimize: true,
            debug_info: false,
            default_sample_rate: 1000,
            default_fft_size: 1024,
        }
    }
}

/// The SigQL compiler
pub struct Compiler {
    config: CompilerConfig,
    next_register: u32,
}

impl Compiler {
    pub fn new(config: CompilerConfig) -> Self {
        Self {
            config,
            next_register: 0,
        }
    }

    /// Allocate a new register
    fn alloc_register(&mut self) -> RegisterId {
        let id = RegisterId(self.next_register);
        self.next_register += 1;
        id
    }

    /// Reset register allocation for new compilation
    fn reset(&mut self) {
        self.next_register = 0;
    }

    /// Compile a query to an execution plan
    pub fn compile(&mut self, query: &Query) -> Result<ExecutionPlan, CompileError> {
        self.reset();
        let mut steps = Vec::new();

        // Phase 1: Compile FROM clause - load signals
        let input_register = if query.from.is_empty() {
            return Err(CompileError::TypeError(
                "Query must have at least one FROM source".into(),
            ));
        } else {
            // For now, compile only the first source
            self.compile_from(&query.from[0], &mut steps)?
        };

        // Phase 2: Compile transforms
        let mut current_register = input_register;
        for transform_clause in &query.transforms {
            for transform_item in &transform_clause.transforms {
                current_register =
                    self.compile_transform(&transform_item.op, current_register, &mut steps)?;
            }
        }

        // Phase 3: Compile window if present
        if let Some(ref window) = query.window {
            current_register = self.compile_window(&window.spec, current_register, &mut steps)?;
        }

        // Phase 4: Compile aggregates
        if let Some(ref aggregate) = query.aggregate {
            for agg_item in &aggregate.aggregations {
                let agg_output =
                    self.compile_aggregate(&agg_item.op, current_register, &mut steps)?;
                steps.push(PlanStep::Store {
                    input: agg_output,
                    name: agg_item.name.clone(),
                });
            }
        } else {
            // No aggregation, store the signal directly
            steps.push(PlanStep::Store {
                input: current_register,
                name: SmolStr::new("result"),
            });
        }

        let plan = ExecutionPlan {
            steps,
            target: self.config.target,
        };

        // Phase 5: Optimize if enabled
        if self.config.optimize {
            Ok(optimize::optimize(plan, &[]))
        } else {
            Ok(plan)
        }
    }

    /// Compile FROM clause
    fn compile_from(
        &mut self,
        from: &FromClause,
        steps: &mut Vec<PlanStep>,
    ) -> Result<RegisterId, CompileError> {
        match from {
            FromClause::Signal(source_ref) => {
                let output = self.alloc_register();
                steps.push(PlanStep::LoadSignal {
                    source: source_ref.path.clone(),
                    output,
                });
                Ok(output)
            }
            FromClause::Session {
                session_id,
                patient: _,
                timestamp: _,
            } => {
                // For sessions, we'd load from a data store
                // For now, treat it as a signal source
                let output = self.alloc_register();
                steps.push(PlanStep::LoadSignal {
                    source: session_id.clone(),
                    output,
                });
                Ok(output)
            }
            FromClause::Subquery { query, alias: _ } => {
                // Recursively compile the subquery
                let mut sub_compiler = Compiler::new(self.config.clone());
                let sub_plan = sub_compiler.compile(query)?;

                // Merge subquery steps
                let offset = self.next_register;
                for step in sub_plan.steps {
                    steps.push(offset_registers(step, offset));
                }
                self.next_register += sub_compiler.next_register;

                Ok(RegisterId(self.next_register - 1))
            }
            FromClause::Table { name, alias: _ } => {
                let output = self.alloc_register();
                steps.push(PlanStep::LoadSignal {
                    source: name.clone(),
                    output,
                });
                Ok(output)
            }
            FromClause::Media { source, alias } => {
                // MediaQL: load media source (image/audio/video)
                // For now, treat the alias as the register name.
                // The runtime will handle media decoding and ingest.
                let output = self.alloc_register();
                let source_name = match source {
                    crate::ast::query::MediaSourceRef::Path(p) => p.clone(),
                    crate::ast::query::MediaSourceRef::Stored { collection, id } => {
                        SmolStr::new(format!("{}.{}", collection, id))
                    }
                    crate::ast::query::MediaSourceRef::Bytes { format } => {
                        SmolStr::new(format!("bytes:{}", format))
                    }
                };
                steps.push(PlanStep::LoadSignal {
                    source: source_name,
                    output,
                });
                Ok(output)
            }
            FromClause::Graph {
                start_node,
                alias,
                ..
            } => {
                // Graph traversal: load starting node signals
                let output = self.alloc_register();
                steps.push(PlanStep::LoadSignal {
                    source: start_node.clone(),
                    output,
                });
                Ok(output)
            }
        }
    }

    /// Compile a transform operation
    fn compile_transform(
        &mut self,
        transform: &TransformOp,
        input: RegisterId,
        steps: &mut Vec<PlanStep>,
    ) -> Result<RegisterId, CompileError> {
        let output = self.alloc_register();
        let sample_rate = SampleRate::new(self.config.default_sample_rate);

        match transform {
            TransformOp::Bandpass(params) => {
                let coeffs = self.design_bandpass_filter(params, sample_rate)?;
                steps.push(PlanStep::IirFilter {
                    input,
                    output,
                    coeffs,
                });
            }
            TransformOp::Lowpass(params) => {
                let coeffs = self.design_lowpass_filter(params, sample_rate)?;
                steps.push(PlanStep::IirFilter {
                    input,
                    output,
                    coeffs,
                });
            }
            TransformOp::Highpass(params) => {
                let coeffs = self.design_highpass_filter(params, sample_rate)?;
                steps.push(PlanStep::IirFilter {
                    input,
                    output,
                    coeffs,
                });
            }
            TransformOp::Notch(params) => {
                let coeffs = self.design_notch_filter(params, sample_rate)?;
                steps.push(PlanStep::IirFilter {
                    input,
                    output,
                    coeffs,
                });
            }
            TransformOp::Fft(params) => {
                let window = self.create_window_coeffs(
                    params.size.unwrap_or(self.config.default_fft_size),
                    &params.window,
                );
                steps.push(PlanStep::Fft {
                    input,
                    output,
                    size: params.size.unwrap_or(self.config.default_fft_size),
                    window,
                });
            }
            TransformOp::Ifft => {
                steps.push(PlanStep::Ifft { input, output });
            }
            TransformOp::Hilbert => {
                steps.push(PlanStep::Envelope { input, output });
            }
            TransformOp::Envelope => {
                steps.push(PlanStep::Envelope { input, output });
            }
            TransformOp::Resample(params) => {
                steps.push(PlanStep::Resample {
                    input,
                    output,
                    from_rate: sample_rate,
                    to_rate: params.target_rate,
                });
            }
            TransformOp::ZScore(_) => {
                steps.push(PlanStep::ZScore { input, output });
            }
            TransformOp::Detrend(params) => {
                steps.push(PlanStep::Detrend {
                    input,
                    output,
                    order: params.order,
                });
            }
            TransformOp::Abs => {
                steps.push(PlanStep::ElementWise {
                    input,
                    output,
                    op: ElementWiseOp::Abs,
                });
            }
            TransformOp::Square => {
                steps.push(PlanStep::ElementWise {
                    input,
                    output,
                    op: ElementWiseOp::Square,
                });
            }
            TransformOp::Sqrt => {
                steps.push(PlanStep::ElementWise {
                    input,
                    output,
                    op: ElementWiseOp::Sqrt,
                });
            }
            TransformOp::Log => {
                steps.push(PlanStep::ElementWise {
                    input,
                    output,
                    op: ElementWiseOp::Log,
                });
            }
            TransformOp::Log10 => {
                steps.push(PlanStep::ElementWise {
                    input,
                    output,
                    op: ElementWiseOp::Log10,
                });
            }
            TransformOp::Exp => {
                steps.push(PlanStep::ElementWise {
                    input,
                    output,
                    op: ElementWiseOp::Exp,
                });
            }
            TransformOp::Diff => {
                steps.push(PlanStep::Diff { input, output });
            }
            TransformOp::Cumsum => {
                steps.push(PlanStep::Cumsum { input, output });
            }
            TransformOp::Scale(factor) => {
                steps.push(PlanStep::ElementWise {
                    input,
                    output,
                    op: ElementWiseOp::Scale(*factor),
                });
            }
            TransformOp::Offset(value) => {
                steps.push(PlanStep::ElementWise {
                    input,
                    output,
                    op: ElementWiseOp::Offset(*value),
                });
            }
            TransformOp::Median(params) => {
                steps.push(PlanStep::MedianFilter {
                    input,
                    output,
                    kernel_size: params.kernel_size,
                });
            }
            TransformOp::Decimate(params) => {
                steps.push(PlanStep::Decimate {
                    input,
                    output,
                    factor: params.factor,
                });
            }
            TransformOp::Interpolate(params) => {
                steps.push(PlanStep::Interpolate {
                    input,
                    output,
                    factor: params.factor,
                });
            }
            // Handle remaining transforms with passthrough for now
            _ => {
                steps.push(PlanStep::Passthrough { input, output });
            }
        }

        Ok(output)
    }

    /// Compile window specification
    fn compile_window(
        &mut self,
        spec: &WindowSpec,
        input: RegisterId,
        steps: &mut Vec<PlanStep>,
    ) -> Result<RegisterId, CompileError> {
        let output = self.alloc_register();
        let sample_rate = self.config.default_sample_rate;

        match &spec.kind {
            WindowKind::Tumbling { duration } => {
                let duration_samples = (duration.0 * sample_rate as f64) as usize;
                steps.push(PlanStep::Window {
                    input,
                    output,
                    duration_samples,
                    step_samples: duration_samples, // Non-overlapping
                });
            }
            WindowKind::Sliding { duration, step } => {
                let duration_samples = (duration.0 * sample_rate as f64) as usize;
                let step_samples = (step.0 * sample_rate as f64) as usize;
                steps.push(PlanStep::Window {
                    input,
                    output,
                    duration_samples,
                    step_samples,
                });
            }
            WindowKind::FrequencyBand(band) => {
                // This is actually a filter, not a time window
                steps.push(PlanStep::BandPower {
                    input,
                    output,
                    band: *band,
                });
            }
            _ => {
                // Other window types: passthrough for now
                steps.push(PlanStep::Passthrough { input, output });
            }
        }

        Ok(output)
    }

    /// Compile aggregate operation
    fn compile_aggregate(
        &mut self,
        op: &AggregateOp,
        input: RegisterId,
        steps: &mut Vec<PlanStep>,
    ) -> Result<RegisterId, CompileError> {
        let output = self.alloc_register();

        match op {
            AggregateOp::Mean => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Mean,
                });
            }
            AggregateOp::Std => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Std,
                });
            }
            AggregateOp::Var => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Variance,
                });
            }
            AggregateOp::Rms => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Rms,
                });
            }
            AggregateOp::Peak => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Max,
                });
            }
            AggregateOp::Trough => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Min,
                });
            }
            AggregateOp::ZeroCrossings => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::ZeroCrossings,
                });
            }
            AggregateOp::BandPower(band) => {
                steps.push(PlanStep::BandPower {
                    input,
                    output,
                    band: band.clone(),
                });
            }
            AggregateOp::DominantFrequency => {
                steps.push(PlanStep::DominantFrequency { input, output });
            }
            AggregateOp::SpectralEntropy => {
                steps.push(PlanStep::SpectralEntropy { input, output });
            }
            AggregateOp::SpectralCentroid => {
                steps.push(PlanStep::SpectralCentroid { input, output });
            }
            AggregateOp::Kurtosis => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Kurtosis,
                });
            }
            AggregateOp::Skewness => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Skewness,
                });
            }
            AggregateOp::Slope => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::Slope,
                });
            }
            AggregateOp::PeakToPeak => {
                steps.push(PlanStep::Reduce {
                    input,
                    output,
                    op: ReduceOp::PeakToPeak,
                });
            }
            _ => {
                // Unimplemented aggregates: use passthrough
                steps.push(PlanStep::Passthrough { input, output });
            }
        }

        Ok(output)
    }

    // Filter design helpers
    fn design_bandpass_filter(
        &self,
        params: &FilterParams,
        sample_rate: SampleRate,
    ) -> Result<IirCoeffs, CompileError> {
        let low = params
            .cutoff_low
            .ok_or_else(|| CompileError::TypeError("Bandpass requires low cutoff".into()))?;
        let high = params
            .cutoff_high
            .ok_or_else(|| CompileError::TypeError("Bandpass requires high cutoff".into()))?;

        use crate::dsp::filter::CascadedBiquad;
        let filter =
            CascadedBiquad::butterworth_bandpass(low, high, sample_rate, params.order as usize)
                .map_err(|e| CompileError::TypeError(format!("Filter design failed: {}", e)))?;

        Ok(biquad_to_iir_coeffs(&filter))
    }

    fn design_lowpass_filter(
        &self,
        params: &FilterParams,
        sample_rate: SampleRate,
    ) -> Result<IirCoeffs, CompileError> {
        let cutoff = params
            .cutoff_high
            .ok_or_else(|| CompileError::TypeError("Lowpass requires cutoff".into()))?;

        use crate::dsp::filter::CascadedBiquad;
        let filter =
            CascadedBiquad::butterworth_lowpass(cutoff, sample_rate, params.order as usize)
                .map_err(|e| CompileError::TypeError(format!("Filter design failed: {}", e)))?;

        Ok(biquad_to_iir_coeffs(&filter))
    }

    fn design_highpass_filter(
        &self,
        params: &FilterParams,
        sample_rate: SampleRate,
    ) -> Result<IirCoeffs, CompileError> {
        let cutoff = params
            .cutoff_low
            .ok_or_else(|| CompileError::TypeError("Highpass requires cutoff".into()))?;

        use crate::dsp::filter::CascadedBiquad;
        let filter =
            CascadedBiquad::butterworth_highpass(cutoff, sample_rate, params.order as usize)
                .map_err(|e| CompileError::TypeError(format!("Filter design failed: {}", e)))?;

        Ok(biquad_to_iir_coeffs(&filter))
    }

    fn design_notch_filter(
        &self,
        params: &crate::ast::NotchParams,
        sample_rate: SampleRate,
    ) -> Result<IirCoeffs, CompileError> {
        let coeffs = BiquadCoeffs::notch(params.frequency, params.q_factor, sample_rate);
        Ok(IirCoeffs {
            sections: vec![[coeffs.b0, coeffs.b1, coeffs.b2, 1.0, coeffs.a1, coeffs.a2]],
        })
    }

    fn create_window_coeffs(&self, size: usize, window: &WindowFunction) -> WindowCoeffs {
        let window_type = match window {
            WindowFunction::Rectangular => WindowType::Rectangular,
            WindowFunction::Hann => WindowType::Hann,
            WindowFunction::Hamming => WindowType::Hamming,
            WindowFunction::Blackman => WindowType::Blackman,
            WindowFunction::Kaiser { beta } => WindowType::Kaiser { beta: *beta as f64 },
            WindowFunction::FlatTop => WindowType::FlatTop,
            WindowFunction::Gaussian { sigma } => WindowType::Gaussian {
                sigma: *sigma as f64,
            },
        };

        WindowCoeffs {
            coeffs: window_type.coefficients(size),
            coherent_gain: window_type.coherent_gain(size),
            enbw: window_type.enbw(),
        }
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new(CompilerConfig::default())
    }
}

/// Helper to convert CascadedBiquad to IirCoeffs
fn biquad_to_iir_coeffs(_filter: &crate::dsp::filter::CascadedBiquad) -> IirCoeffs {
    // Access the internal sections - this is a bit of a hack
    // In a real implementation, CascadedBiquad would expose its coefficients
    IirCoeffs {
        sections: Vec::new(), // Will be populated by runtime from DSP module
    }
}

/// Offset all register IDs in a step
fn offset_registers(step: PlanStep, offset: u32) -> PlanStep {
    match step {
        PlanStep::LoadSignal { source, output } => PlanStep::LoadSignal {
            source,
            output: RegisterId(output.0 + offset),
        },
        PlanStep::Fft {
            input,
            output,
            size,
            window,
        } => PlanStep::Fft {
            input: RegisterId(input.0 + offset),
            output: RegisterId(output.0 + offset),
            size,
            window,
        },
        PlanStep::IirFilter {
            input,
            output,
            coeffs,
        } => PlanStep::IirFilter {
            input: RegisterId(input.0 + offset),
            output: RegisterId(output.0 + offset),
            coeffs,
        },
        PlanStep::Reduce { input, output, op } => PlanStep::Reduce {
            input: RegisterId(input.0 + offset),
            output: RegisterId(output.0 + offset),
            op,
        },
        PlanStep::Store { input, name } => PlanStep::Store {
            input: RegisterId(input.0 + offset),
            name,
        },
        // Handle all other steps similarly
        other => other,
    }
}

/// Compilation errors
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("Type error: {0}")]
    TypeError(String),

    #[error("Unknown signal source: {0}")]
    UnknownSource(String),

    #[error("Unsupported operation for target: {0}")]
    UnsupportedOperation(String),

    #[error("Sample rate mismatch: expected {expected}, got {actual}")]
    SampleRateMismatch { expected: u32, actual: u32 },

    #[error("Parse error: {0}")]
    ParseError(#[from] ParseError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_query;

    #[test]
    fn test_compile_simple() {
        let query = parse_query("FROM sensor.data").unwrap();
        let mut compiler = Compiler::default();
        let plan = compiler.compile(&query).unwrap();

        assert!(!plan.steps.is_empty());
    }

    #[test]
    fn test_compile_with_transform() {
        let query = parse_query("FROM sensor.data TRANSFORM bandpass(4Hz, 12Hz)").unwrap();
        let mut compiler = Compiler::default();
        let plan = compiler.compile(&query).unwrap();

        // Should have: LoadSignal, IirFilter, Store
        assert!(plan.steps.len() >= 2);
    }

    #[test]
    fn test_compile_with_aggregate() {
        let query = parse_query("FROM sensor.data AGGREGATE { power: rms }").unwrap();
        let mut compiler = Compiler::default();
        let plan = compiler.compile(&query).unwrap();

        // Should have: LoadSignal, Reduce, Store
        assert!(plan.steps.len() >= 2);
    }
}
