//! SigQL Runtime
//!
//! Executes compiled plans against actual signal data using the DSP module.

use std::collections::HashMap;

use num_complex::Complex64;
use smol_str::SmolStr;

use crate::compile::plan::{
    ElementWiseOp, ExecutionPlan, FirCoeffs, IirCoeffs, PlanStep, ReduceOp, RegisterId,
    WindowCoeffs,
};
use crate::dsp::filter::{BiquadCoeffs, BiquadFilter, MedianFilter};
use crate::dsp::statistics::{
    compute_kurtosis, compute_mean, compute_rms, compute_std, detrend, zscore,
};
use crate::dsp::{self, DspError, HilbertTransform, ResampleMethod, Resampler};
use crate::types::{DynSignal, FrequencyBand, SampleRate, UncertainValue};

/// Runtime execution context
pub struct Runtime {
    /// Signal sources available to queries
    sources: HashMap<SmolStr, SignalSource>,
    /// Configuration
    config: RuntimeConfig,
}

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum buffer size (samples)
    pub max_buffer_size: usize,
    /// Enable streaming mode
    pub streaming: bool,
    /// Default confidence level
    pub confidence: f64,
    /// Default sample rate when not specified
    pub default_sample_rate: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 1_000_000,
            streaming: false,
            confidence: 0.95,
            default_sample_rate: 1000,
        }
    }
}

/// A signal source that can provide data
pub enum SignalSource {
    /// In-memory signal
    Memory(DynSignal<f64>),
    /// File-backed signal
    File { path: String, format: FileFormat },
    /// Live streaming source
    Stream {
        callback: Box<dyn Fn() -> Vec<f64> + Send + Sync>,
    },
    /// Database-backed signal (requires 'storage' feature)
    #[cfg(feature = "storage")]
    Database {
        connector: std::sync::Arc<crate::io::SignalStorageConnector>,
        signal_name: String,
    },
}

/// Supported file formats
#[derive(Debug, Clone)]
pub enum FileFormat {
    Csv,
    Edf,
    Wav,
    Raw { sample_rate: u32, channels: usize },
}

/// Execution result
#[derive(Debug)]
pub struct ExecutionResult {
    /// Named outputs
    pub outputs: HashMap<SmolStr, OutputValue>,
    /// Execution statistics
    pub stats: ExecutionStats,
}

/// Output value types
#[derive(Debug, Clone)]
pub enum OutputValue {
    Scalar(UncertainValue<f64>),
    Signal(DynSignal<f64>),
    Spectrum(Vec<f64>),
}

/// Execution statistics
#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub samples_processed: usize,
    pub execution_time_ns: u64,
    pub memory_used: usize,
}

impl Runtime {
    /// Create a new runtime
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            sources: HashMap::new(),
            config,
        }
    }

    /// Register a signal source
    pub fn register_source(&mut self, name: impl Into<SmolStr>, source: SignalSource) {
        self.sources.insert(name.into(), source);
    }

    /// Register an in-memory signal
    pub fn register_signal(&mut self, name: impl Into<SmolStr>, signal: DynSignal<f64>) {
        self.sources
            .insert(name.into(), SignalSource::Memory(signal));
    }

    /// Execute a plan
    pub fn execute(&self, plan: &ExecutionPlan) -> Result<ExecutionResult, RuntimeError> {
        let start = std::time::Instant::now();

        // Allocate registers
        let resources = plan.allocate_resources();
        let mut registers: Vec<Option<RegisterValue>> =
            vec![None; resources.num_registers as usize];

        let mut outputs = HashMap::new();
        let mut samples_processed = 0usize;

        // Execute each step
        for step in &plan.steps {
            self.execute_step(step, &mut registers, &mut outputs, &mut samples_processed)?;
        }

        let elapsed = start.elapsed();

        Ok(ExecutionResult {
            outputs,
            stats: ExecutionStats {
                samples_processed,
                execution_time_ns: elapsed.as_nanos() as u64,
                memory_used: registers.iter().filter(|r| r.is_some()).count() * 8 * 1024, // Rough estimate
            },
        })
    }

    fn execute_step(
        &self,
        step: &PlanStep,
        registers: &mut [Option<RegisterValue>],
        outputs: &mut HashMap<SmolStr, OutputValue>,
        samples_processed: &mut usize,
    ) -> Result<(), RuntimeError> {
        match step {
            PlanStep::LoadSignal { source, output } => {
                let signal = self.load_signal(source)?;
                *samples_processed += signal.samples.len();
                registers[output.0 as usize] = Some(RegisterValue::Signal(signal));
            }

            PlanStep::Fft {
                input,
                output,
                size,
                window,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let spectrum = self.compute_fft(&input_signal, *size, window)?;
                registers[output.0 as usize] = Some(RegisterValue::Spectrum(spectrum));
            }

            PlanStep::Ifft { input, output } => {
                let spectrum = self.get_spectrum(registers, *input)?;
                let signal = self.compute_ifft(&spectrum)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(signal));
            }

            PlanStep::IirFilter {
                input,
                output,
                coeffs,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let filtered = self.apply_iir_filter(&input_signal, coeffs)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(filtered));
            }

            PlanStep::FirFilter {
                input,
                output,
                coeffs,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let filtered = self.apply_fir_filter(&input_signal, coeffs)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(filtered));
            }

            PlanStep::Resample {
                input,
                output,
                from_rate,
                to_rate,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let resampled = self.resample(&input_signal, *from_rate, *to_rate)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(resampled));
            }

            PlanStep::ComplexToMagnitude { input, output } => {
                let spectrum = self.get_spectrum(registers, *input)?;
                // Spectrum is already magnitude, just copy
                registers[output.0 as usize] = Some(RegisterValue::Spectrum(spectrum));
            }

            PlanStep::Envelope { input, output } => {
                let input_signal = self.get_signal(registers, *input)?;
                let envelope = self.compute_envelope(&input_signal)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(envelope));
            }

            PlanStep::Reduce { input, output, op } => {
                let input_signal = self.get_signal(registers, *input)?;
                let scalar = self.reduce(&input_signal, *op)?;
                registers[output.0 as usize] = Some(RegisterValue::Scalar(scalar));
            }

            PlanStep::Window {
                input,
                output,
                duration_samples,
                step_samples,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let windowed =
                    self.apply_window(&input_signal, *duration_samples, *step_samples)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(windowed));
            }

            PlanStep::CrossCorrelate {
                input_a,
                input_b,
                output,
                max_lag_samples,
            } => {
                let signal_a = self.get_signal(registers, *input_a)?;
                let signal_b = self.get_signal(registers, *input_b)?;
                let corr = self.cross_correlate(&signal_a, &signal_b, *max_lag_samples)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(corr));
            }

            PlanStep::BandPower {
                input,
                output,
                band,
            } => {
                // Input can be spectrum or signal
                let power = match &registers[input.0 as usize] {
                    Some(RegisterValue::Spectrum(s)) => self.band_power_from_spectrum(s, band)?,
                    Some(RegisterValue::Signal(s)) => self.band_power_from_signal(s, band)?,
                    _ => {
                        return Err(RuntimeError::TypeMismatch(
                            "Expected spectrum or signal".into(),
                        ));
                    }
                };
                registers[output.0 as usize] = Some(RegisterValue::Scalar(power));
            }

            PlanStep::Store { input, name } => {
                let value = registers[input.0 as usize]
                    .take()
                    .ok_or(RuntimeError::RegisterEmpty(*input))?;
                outputs.insert(name.clone(), self.to_output_value(value));
            }

            PlanStep::ZScore { input, output } => {
                let input_signal = self.get_signal(registers, *input)?;
                let normalized = self.zscore_normalize(&input_signal)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(normalized));
            }

            PlanStep::Detrend {
                input,
                output,
                order,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let detrended = self.detrend(&input_signal, *order)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(detrended));
            }

            PlanStep::ElementWise { input, output, op } => {
                let input_signal = self.get_signal(registers, *input)?;
                let result = self.element_wise(&input_signal, *op)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(result));
            }

            PlanStep::Diff { input, output } => {
                let input_signal = self.get_signal(registers, *input)?;
                let result = self.diff(&input_signal)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(result));
            }

            PlanStep::Cumsum { input, output } => {
                let input_signal = self.get_signal(registers, *input)?;
                let result = self.cumsum(&input_signal)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(result));
            }

            PlanStep::MedianFilter {
                input,
                output,
                kernel_size,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let filtered = self.median_filter(&input_signal, *kernel_size)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(filtered));
            }

            PlanStep::Decimate {
                input,
                output,
                factor,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let decimated = self.decimate(&input_signal, *factor)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(decimated));
            }

            PlanStep::Interpolate {
                input,
                output,
                factor,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let interpolated = self.interpolate(&input_signal, *factor)?;
                registers[output.0 as usize] = Some(RegisterValue::Signal(interpolated));
            }

            PlanStep::DominantFrequency { input, output } => {
                let input_signal = self.get_signal(registers, *input)?;
                let freq = self.dominant_frequency(&input_signal)?;
                registers[output.0 as usize] = Some(RegisterValue::Scalar(freq));
            }

            PlanStep::SpectralEntropy { input, output } => {
                let input_signal = self.get_signal(registers, *input)?;
                let entropy = self.spectral_entropy(&input_signal)?;
                registers[output.0 as usize] = Some(RegisterValue::Scalar(entropy));
            }

            PlanStep::SpectralCentroid { input, output } => {
                let input_signal = self.get_signal(registers, *input)?;
                let centroid = self.spectral_centroid(&input_signal)?;
                registers[output.0 as usize] = Some(RegisterValue::Scalar(centroid));
            }

            PlanStep::Passthrough { input, output } => {
                let value = registers[input.0 as usize]
                    .clone()
                    .ok_or(RuntimeError::RegisterEmpty(*input))?;
                registers[output.0 as usize] = Some(value);
            }

            // MediaQL: 2D DCT on signal data (treats samples as flat 2D grid)
            PlanStep::Dct2d {
                input,
                output,
                block_size,
                quality,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let config = crate::io::media_ingest::MediaIngestConfig {
                    dct_block_size: *block_size,
                    quality: *quality,
                    ..Default::default()
                };
                let pipeline = crate::io::media_ingest::MediaIngestPipeline::new(config);

                // Treat signal samples as a square image
                let n = input_signal.samples.len();
                let side = (n as f64).sqrt() as u32;
                let ingested = pipeline
                    .ingest_image(&input_signal.samples, side, side)
                    .map_err(|e| RuntimeError::OperationFailed(format!("DCT2D: {}", e)))?;

                // Output as spectrum (coefficient magnitudes)
                let spectrum: Vec<f64> = ingested
                    .coefficients
                    .entries
                    .iter()
                    .map(|e| e.magnitude as f64)
                    .collect();
                registers[output.0 as usize] = Some(RegisterValue::Spectrum(spectrum));
            }

            // MediaQL: Inverse 2D DCT (reconstruct from spectrum)
            PlanStep::Idct2d { input, output } => {
                // Passthrough for now — full IDCT would need stored coefficient metadata
                let value = registers[input.0 as usize]
                    .clone()
                    .ok_or(RuntimeError::RegisterEmpty(*input))?;
                registers[output.0 as usize] = Some(value);
            }

            // MediaQL: MFCC extraction from audio signal
            PlanStep::Mfcc {
                input,
                output,
                n_coefficients,
            } => {
                let input_signal = self.get_signal(registers, *input)?;
                let config = crate::io::media_ingest::MediaIngestConfig {
                    fft_window_size: 2048.min(input_signal.samples.len()),
                    hop_size: 512,
                    ..Default::default()
                };
                let pipeline = crate::io::media_ingest::MediaIngestPipeline::new(config);

                let ingested = pipeline
                    .ingest_audio(&input_signal.samples, input_signal.sample_rate)
                    .map_err(|e| RuntimeError::OperationFailed(format!("MFCC: {}", e)))?;

                // Output as spectrum (coefficient magnitudes, first n_coefficients per frame)
                let spectrum: Vec<f64> = ingested
                    .coefficients
                    .entries
                    .iter()
                    .take(*n_coefficients * 100) // Rough: n_coefficients × ~100 frames
                    .map(|e| e.magnitude as f64)
                    .collect();
                registers[output.0 as usize] = Some(RegisterValue::Spectrum(spectrum));
            }

            // MediaQL: 2D FFT, perceptual hash, edge detect — pass through with metadata
            PlanStep::Fft2d { input, output }
            | PlanStep::Ifft2d { input, output }
            | PlanStep::PerceptualHash { input, output }
            | PlanStep::EdgeDetect { input, output } => {
                let value = registers[input.0 as usize]
                    .clone()
                    .ok_or(RuntimeError::RegisterEmpty(*input))?;
                registers[output.0 as usize] = Some(value);
            }
        }

        Ok(())
    }

    // ==================== Signal Loading ====================

    fn load_signal(&self, source: &SmolStr) -> Result<DynSignal<f64>, RuntimeError> {
        match self.sources.get(source) {
            Some(SignalSource::Memory(signal)) => Ok(signal.clone()),
            Some(SignalSource::File { path, format }) => self.load_from_file(path, format),
            Some(SignalSource::Stream { callback }) => {
                let samples = callback();
                Ok(DynSignal::new(
                    source.clone(),
                    samples,
                    self.config.default_sample_rate,
                    0,
                ))
            }
            #[cfg(feature = "storage")]
            Some(SignalSource::Database {
                connector,
                signal_name,
            }) => connector
                .load_signal(signal_name)
                .map_err(|e| RuntimeError::FileLoadError(format!("Database load failed: {}", e))),
            None => Err(RuntimeError::SourceNotFound(source.clone())),
        }
    }

    fn load_from_file(
        &self,
        path: &str,
        format: &FileFormat,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        use std::path::Path;
        let file_path = Path::new(path);

        match format {
            FileFormat::Wav => {
                let signal = crate::io::wav::read_wav(file_path)
                    .map_err(|e| RuntimeError::FileLoadError(format!("{}: {}", path, e)))?;
                Ok(signal)
            }
            FileFormat::Csv => {
                let signal = crate::io::csv::read_csv(file_path, "value")
                    .map_err(|e| RuntimeError::FileLoadError(format!("{}: {}", path, e)))?;
                Ok(signal)
            }
            _ => Err(RuntimeError::FileLoadError(format!(
                "Unsupported format for {}",
                path
            ))),
        }
    }

    // ==================== Register Access ====================

    fn get_signal(
        &self,
        registers: &[Option<RegisterValue>],
        id: RegisterId,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        match &registers[id.0 as usize] {
            Some(RegisterValue::Signal(s)) => Ok(s.clone()),
            Some(_) => Err(RuntimeError::TypeMismatch("Expected signal".into())),
            None => Err(RuntimeError::RegisterEmpty(id)),
        }
    }

    fn get_spectrum(
        &self,
        registers: &[Option<RegisterValue>],
        id: RegisterId,
    ) -> Result<Vec<f64>, RuntimeError> {
        match &registers[id.0 as usize] {
            Some(RegisterValue::Spectrum(s)) => Ok(s.clone()),
            Some(_) => Err(RuntimeError::TypeMismatch("Expected spectrum".into())),
            None => Err(RuntimeError::RegisterEmpty(id)),
        }
    }

    // ==================== FFT Operations ====================

    fn compute_fft(
        &self,
        signal: &DynSignal<f64>,
        size: usize,
        window: &WindowCoeffs,
    ) -> Result<Vec<f64>, RuntimeError> {
        let samples = &signal.samples;
        let n = samples.len().min(size);

        // Apply window and pad if needed
        let mut windowed: Vec<f64> = samples
            .iter()
            .take(n)
            .enumerate()
            .map(|(i, &s)| {
                let w = if i < window.coeffs.len() {
                    window.coeffs[i]
                } else {
                    1.0
                };
                s * w
            })
            .collect();

        // Pad to size if needed
        windowed.resize(size, 0.0);

        // Convert to complex
        let mut complex: Vec<Complex64> =
            windowed.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        // Compute FFT using rustfft
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        fft.process(&mut complex);

        // Return magnitude spectrum (positive frequencies only)
        let n_bins = size / 2 + 1;
        let magnitude: Vec<f64> = complex
            .iter()
            .take(n_bins)
            .map(|c| c.norm() / window.coherent_gain / (size as f64).sqrt())
            .collect();

        Ok(magnitude)
    }

    fn compute_ifft(&self, spectrum: &[f64]) -> Result<DynSignal<f64>, RuntimeError> {
        let n_bins = spectrum.len();
        let size = (n_bins - 1) * 2;

        // Reconstruct full spectrum (conjugate symmetric)
        let mut complex: Vec<Complex64> = Vec::with_capacity(size);
        for (_i, &mag) in spectrum.iter().enumerate() {
            complex.push(Complex64::new(mag * (size as f64).sqrt(), 0.0));
        }
        // Mirror for negative frequencies
        for i in 1..n_bins - 1 {
            complex.push(complex[n_bins - 1 - i].conj());
        }

        let mut planner = rustfft::FftPlanner::new();
        let ifft = planner.plan_fft_inverse(size);
        ifft.process(&mut complex);

        let samples: Vec<f64> = complex.iter().map(|c| c.re / size as f64).collect();

        Ok(DynSignal::new(
            SmolStr::new("ifft_result"),
            samples,
            self.config.default_sample_rate,
            0,
        ))
    }

    // ==================== Filter Operations ====================

    fn apply_iir_filter(
        &self,
        signal: &DynSignal<f64>,
        coeffs: &IirCoeffs,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        if coeffs.sections.is_empty() {
            // No filter, passthrough
            return Ok(signal.clone());
        }

        let mut output = signal.samples.clone();

        // Apply each biquad section
        for section in &coeffs.sections {
            let [b0, b1, b2, _a0, a1, a2] = *section;
            let biquad_coeffs = BiquadCoeffs { b0, b1, b2, a1, a2 };
            let mut filter = BiquadFilter::new(biquad_coeffs);

            output = filter.process(&output);
        }

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    fn apply_fir_filter(
        &self,
        signal: &DynSignal<f64>,
        coeffs: &FirCoeffs,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        if coeffs.taps.is_empty() {
            return Ok(signal.clone());
        }

        let samples = &signal.samples;
        let taps = &coeffs.taps;
        let n = samples.len();
        let m = taps.len();

        let mut output = vec![0.0; n];

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..m {
                if i >= j {
                    sum += taps[j] * samples[i - j];
                }
            }
            output[i] = sum;
        }

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    fn median_filter(
        &self,
        signal: &DynSignal<f64>,
        kernel_size: usize,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        let filter = MedianFilter::new(kernel_size).map_err(|e| RuntimeError::DspError(e))?;
        let output = filter.process(&signal.samples);

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    // ==================== Resampling ====================

    fn resample(
        &self,
        signal: &DynSignal<f64>,
        from_rate: SampleRate,
        to_rate: SampleRate,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        let resampler = Resampler::new(to_rate, ResampleMethod::Sinc);
        let output = resampler
            .resample(&signal.samples, from_rate)
            .map_err(|e| RuntimeError::DspError(e))?;

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            to_rate.0,
            signal.start_ns,
        ))
    }

    fn decimate(
        &self,
        signal: &DynSignal<f64>,
        factor: usize,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        let sample_rate = SampleRate::new(signal.sample_rate);
        let output = dsp::resample::decimate(&signal.samples, factor, sample_rate)
            .map_err(|e| RuntimeError::DspError(e))?;

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate / factor as u32,
            signal.start_ns,
        ))
    }

    fn interpolate(
        &self,
        signal: &DynSignal<f64>,
        factor: usize,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        // Simple linear interpolation
        let samples = &signal.samples;
        let n = samples.len();
        let new_len = n * factor;
        let mut output = vec![0.0; new_len];

        for i in 0..n - 1 {
            let start = samples[i];
            let end = samples[i + 1];
            for j in 0..factor {
                let t = j as f64 / factor as f64;
                output[i * factor + j] = start + t * (end - start);
            }
        }
        // Last sample
        if n > 0 {
            for j in 0..factor {
                output[(n - 1) * factor + j] = samples[n - 1];
            }
        }

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate * factor as u32,
            signal.start_ns,
        ))
    }

    // ==================== Envelope/Hilbert ====================

    fn compute_envelope(&self, signal: &DynSignal<f64>) -> Result<DynSignal<f64>, RuntimeError> {
        let original_len = signal.samples.len();

        // Pad to next power of 2 for FFT
        let fft_size = original_len.next_power_of_two();
        let mut padded = signal.samples.clone();
        padded.resize(fft_size, 0.0);

        let hilbert = HilbertTransform::new(fft_size).map_err(|e| RuntimeError::DspError(e))?;
        let mut envelope = hilbert
            .envelope(&padded)
            .map_err(|e| RuntimeError::DspError(e))?;

        // Truncate back to original length
        envelope.truncate(original_len);

        Ok(DynSignal::new(
            signal.channel.clone(),
            envelope,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    // ==================== Windowing ====================

    fn apply_window(
        &self,
        signal: &DynSignal<f64>,
        duration_samples: usize,
        step_samples: usize,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        // For now, return the last complete window
        let samples = &signal.samples;
        let n = samples.len();

        if n < duration_samples {
            return Ok(signal.clone());
        }

        // Get the last complete window
        let start = (n / step_samples) * step_samples;
        let start = if start + duration_samples > n {
            n.saturating_sub(duration_samples)
        } else {
            start
        };

        let window_samples: Vec<f64> = samples[start..start + duration_samples].to_vec();

        Ok(DynSignal::new(
            signal.channel.clone(),
            window_samples,
            signal.sample_rate,
            signal.start_ns + (start as i64 * 1_000_000_000 / signal.sample_rate as i64),
        ))
    }

    // ==================== Correlation ====================

    fn cross_correlate(
        &self,
        signal_a: &DynSignal<f64>,
        signal_b: &DynSignal<f64>,
        max_lag_samples: usize,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        let corr = dsp::correlation::cross_correlate(
            &signal_a.samples,
            &signal_b.samples,
            Some(max_lag_samples),
        )
        .map_err(|e| RuntimeError::DspError(e))?;

        Ok(DynSignal::new(
            SmolStr::new("cross_correlation"),
            corr,
            signal_a.sample_rate,
            0_i64,
        ))
    }

    // ==================== Band Power ====================

    fn band_power_from_spectrum(
        &self,
        spectrum: &[f64],
        band: &FrequencyBand,
    ) -> Result<UncertainValue<f64>, RuntimeError> {
        let n_bins = spectrum.len();
        let df = self.config.default_sample_rate as f64 / ((n_bins - 1) * 2) as f64;

        let low_bin = (band.low.0 / df).ceil() as usize;
        let high_bin = (band.high.0 / df).floor() as usize;

        let low_bin = low_bin.min(n_bins - 1);
        let high_bin = high_bin.min(n_bins - 1);

        if low_bin >= high_bin {
            return Ok(UncertainValue::from_ci(0.0, 0.0, 0.95, 1));
        }

        let power: f64 = spectrum[low_bin..=high_bin]
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            * df;

        Ok(UncertainValue::from_ci(power, 0.0, 0.95, 1))
    }

    fn band_power_from_signal(
        &self,
        signal: &DynSignal<f64>,
        band: &FrequencyBand,
    ) -> Result<UncertainValue<f64>, RuntimeError> {
        // Compute FFT first
        let window = WindowCoeffs {
            coeffs: vec![1.0; signal.samples.len()],
            coherent_gain: 1.0,
            enbw: 1.0,
        };
        let spectrum =
            self.compute_fft(signal, signal.samples.len().next_power_of_two(), &window)?;
        self.band_power_from_spectrum(&spectrum, band)
    }

    // ==================== Normalization ====================

    fn zscore_normalize(&self, signal: &DynSignal<f64>) -> Result<DynSignal<f64>, RuntimeError> {
        let output = zscore(&signal.samples).map_err(|e| RuntimeError::DspError(e))?;

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    fn detrend(&self, signal: &DynSignal<f64>, order: u8) -> Result<DynSignal<f64>, RuntimeError> {
        let output =
            detrend(&signal.samples, order as usize).map_err(|e| RuntimeError::DspError(e))?;

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    // ==================== Element-wise Operations ====================

    fn element_wise(
        &self,
        signal: &DynSignal<f64>,
        op: ElementWiseOp,
    ) -> Result<DynSignal<f64>, RuntimeError> {
        let output: Vec<f64> = match op {
            ElementWiseOp::Abs => signal.samples.iter().map(|x| x.abs()).collect(),
            ElementWiseOp::Square => signal.samples.iter().map(|x| x * x).collect(),
            ElementWiseOp::Sqrt => signal.samples.iter().map(|x| x.max(0.0).sqrt()).collect(),
            ElementWiseOp::Log => signal.samples.iter().map(|x| x.max(1e-10).ln()).collect(),
            ElementWiseOp::Log10 => signal
                .samples
                .iter()
                .map(|x| x.max(1e-10).log10())
                .collect(),
            ElementWiseOp::Exp => signal.samples.iter().map(|x| x.exp()).collect(),
            ElementWiseOp::Scale(factor) => signal.samples.iter().map(|x| x * factor).collect(),
            ElementWiseOp::Offset(value) => signal.samples.iter().map(|x| x + value).collect(),
            ElementWiseOp::Negate => signal.samples.iter().map(|x| -x).collect(),
        };

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    fn diff(&self, signal: &DynSignal<f64>) -> Result<DynSignal<f64>, RuntimeError> {
        let samples = &signal.samples;
        let output: Vec<f64> = samples.windows(2).map(|w| w[1] - w[0]).collect();

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    fn cumsum(&self, signal: &DynSignal<f64>) -> Result<DynSignal<f64>, RuntimeError> {
        let mut sum = 0.0;
        let output: Vec<f64> = signal
            .samples
            .iter()
            .map(|&x| {
                sum += x;
                sum
            })
            .collect();

        Ok(DynSignal::new(
            signal.channel.clone(),
            output,
            signal.sample_rate,
            signal.start_ns,
        ))
    }

    // ==================== Spectral Features ====================

    fn dominant_frequency(
        &self,
        signal: &DynSignal<f64>,
    ) -> Result<UncertainValue<f64>, RuntimeError> {
        let window = WindowCoeffs {
            coeffs: vec![1.0; signal.samples.len()],
            coherent_gain: 1.0,
            enbw: 1.0,
        };
        let fft_size = signal.samples.len().next_power_of_two();
        let spectrum = self.compute_fft(signal, fft_size, &window)?;

        let df = signal.sample_rate as f64 / fft_size as f64;

        // Find bin with maximum magnitude
        let (max_bin, _max_mag) = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let freq = max_bin as f64 * df;
        Ok(UncertainValue::from_ci(freq, 0.0, 0.95, 1))
    }

    fn spectral_entropy(
        &self,
        signal: &DynSignal<f64>,
    ) -> Result<UncertainValue<f64>, RuntimeError> {
        let window = WindowCoeffs {
            coeffs: vec![1.0; signal.samples.len()],
            coherent_gain: 1.0,
            enbw: 1.0,
        };
        let fft_size = signal.samples.len().next_power_of_two();
        let spectrum = self.compute_fft(signal, fft_size, &window)?;

        // Compute power spectrum
        let power: Vec<f64> = spectrum.iter().map(|x| x * x).collect();
        let total_power: f64 = power.iter().sum();

        if total_power == 0.0 {
            return Ok(UncertainValue::from_ci(0.0, 0.0, 0.95, 1));
        }

        // Normalize to probability distribution and compute entropy
        let entropy: f64 = power
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| {
                let prob = p / total_power;
                -prob * prob.ln()
            })
            .sum();

        // Normalize by log(n) to get value in [0, 1]
        let n = spectrum.len() as f64;
        let normalized_entropy = entropy / n.ln();

        Ok(UncertainValue::from_ci(normalized_entropy, 0.0, 0.95, 1))
    }

    fn spectral_centroid(
        &self,
        signal: &DynSignal<f64>,
    ) -> Result<UncertainValue<f64>, RuntimeError> {
        let window = WindowCoeffs {
            coeffs: vec![1.0; signal.samples.len()],
            coherent_gain: 1.0,
            enbw: 1.0,
        };
        let fft_size = signal.samples.len().next_power_of_two();
        let spectrum = self.compute_fft(signal, fft_size, &window)?;

        let df = signal.sample_rate as f64 / fft_size as f64;

        let mut weighted_sum = 0.0;
        let mut total_mag = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            let freq = i as f64 * df;
            weighted_sum += freq * mag;
            total_mag += mag;
        }

        let centroid = if total_mag > 0.0 {
            weighted_sum / total_mag
        } else {
            0.0
        };

        Ok(UncertainValue::from_ci(centroid, 0.0, 0.95, 1))
    }

    // ==================== Reduction Operations ====================

    fn reduce(
        &self,
        signal: &DynSignal<f64>,
        op: ReduceOp,
    ) -> Result<UncertainValue<f64>, RuntimeError> {
        let samples = &signal.samples;
        let n = samples.len();

        if n == 0 {
            return Ok(UncertainValue::default());
        }

        // For operations that return UncertainValue, use them directly
        match op {
            ReduceOp::Mean => compute_mean(samples).map_err(|e| RuntimeError::DspError(e)),
            ReduceOp::Std => compute_std(samples).map_err(|e| RuntimeError::DspError(e)),
            ReduceOp::Rms => compute_rms(samples).map_err(|e| RuntimeError::DspError(e)),
            ReduceOp::Kurtosis => compute_kurtosis(samples).map_err(|e| RuntimeError::DspError(e)),
            _ => {
                // For other operations, compute value manually
                let value = match op {
                    ReduceOp::Sum => samples.iter().sum::<f64>(),
                    ReduceOp::Max => samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    ReduceOp::Min => samples.iter().cloned().fold(f64::INFINITY, f64::min),
                    ReduceOp::Variance => {
                        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
                        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64
                    }
                    ReduceOp::ZeroCrossings => {
                        let mut count = 0;
                        for i in 1..n {
                            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                                count += 1;
                            }
                        }
                        count as f64
                    }
                    ReduceOp::Skewness => {
                        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
                        let variance =
                            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
                        let std = variance.sqrt();
                        if std == 0.0 {
                            0.0
                        } else {
                            samples
                                .iter()
                                .map(|x| ((x - mean) / std).powi(3))
                                .sum::<f64>()
                                / n as f64
                        }
                    }
                    ReduceOp::Slope => {
                        let x_mean = (n - 1) as f64 / 2.0;
                        let y_mean: f64 = samples.iter().sum::<f64>() / n as f64;
                        let mut num = 0.0;
                        let mut den = 0.0;
                        for (i, y) in samples.iter().enumerate() {
                            let x = i as f64;
                            num += (x - x_mean) * (y - y_mean);
                            den += (x - x_mean).powi(2);
                        }
                        if den == 0.0 { 0.0 } else { num / den }
                    }
                    ReduceOp::PeakToPeak => {
                        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
                        max - min
                    }
                    // These should have been handled above via early return.
                    // If we reach here, it means the match arm was accidentally bypassed.
                    // Implement them here as a safety fallback instead of panicking.
                    ReduceOp::Mean => samples.iter().sum::<f64>() / n as f64,
                    ReduceOp::Std => {
                        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
                        (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
                    }
                    ReduceOp::Rms => {
                        (samples.iter().map(|x| x.powi(2)).sum::<f64>() / n as f64).sqrt()
                    }
                    ReduceOp::Kurtosis => {
                        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
                        let variance =
                            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
                        let std = variance.sqrt();
                        if std == 0.0 {
                            0.0
                        } else {
                            samples
                                .iter()
                                .map(|x| ((x - mean) / std).powi(4))
                                .sum::<f64>()
                                / n as f64
                                - 3.0
                        }
                    }
                };

                // Compute uncertainty (standard error)
                let mean: f64 = samples.iter().sum::<f64>() / n as f64;
                let variance =
                    samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
                let std_error = variance.sqrt() / (n as f64).sqrt();

                Ok(UncertainValue::from_mean_se(value, std_error, n))
            }
        }
    }

    // ==================== Output Conversion ====================

    fn to_output_value(&self, value: RegisterValue) -> OutputValue {
        match value {
            RegisterValue::Signal(s) => OutputValue::Signal(s),
            RegisterValue::Scalar(v) => OutputValue::Scalar(v),
            RegisterValue::Spectrum(s) => OutputValue::Spectrum(s),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new(RuntimeConfig::default())
    }
}

/// Register value during execution
#[derive(Debug, Clone)]
enum RegisterValue {
    Signal(DynSignal<f64>),
    Spectrum(Vec<f64>),
    Scalar(UncertainValue<f64>),
}

/// Runtime errors
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Signal source not found: {0}")]
    SourceNotFound(SmolStr),

    #[error("Register is empty: {0:?}")]
    RegisterEmpty(RegisterId),

    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    #[error("Failed to load file: {0}")]
    FileLoadError(String),

    #[error("Buffer overflow: requested {requested} samples, max is {max}")]
    BufferOverflow { requested: usize, max: usize },

    #[error("DSP error: {0}")]
    DspError(#[from] DspError),

    #[error("Operation failed: {0}")]
    OperationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_runtime_creation() {
        let runtime = Runtime::new(RuntimeConfig::default());
        assert!(runtime.sources.is_empty());
    }

    #[test]
    fn test_register_signal() {
        let mut runtime = Runtime::new(RuntimeConfig::default());
        let signal = DynSignal::new("test", vec![1.0, 2.0, 3.0], 100, 0);
        runtime.register_signal("test.signal", signal);
        assert!(runtime.sources.contains_key(&SmolStr::new("test.signal")));
    }

    #[test]
    fn test_reduce_mean() {
        let runtime = Runtime::new(RuntimeConfig::default());
        let signal = DynSignal::new("test", vec![1.0, 2.0, 3.0, 4.0, 5.0], 100, 0);
        let result = runtime.reduce(&signal, ReduceOp::Mean).unwrap();
        assert!((result.value - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_rms() {
        let runtime = Runtime::new(RuntimeConfig::default());
        let signal = DynSignal::new("test", vec![1.0, 1.0, 1.0, 1.0], 100, 0);
        let result = runtime.reduce(&signal, ReduceOp::Rms).unwrap();
        assert!((result.value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_element_wise_square() {
        let runtime = Runtime::new(RuntimeConfig::default());
        let signal = DynSignal::new("test", vec![1.0, 2.0, 3.0], 100, 0);
        let result = runtime
            .element_wise(&signal, ElementWiseOp::Square)
            .unwrap();
        assert_eq!(result.samples, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_diff() {
        let runtime = Runtime::new(RuntimeConfig::default());
        let signal = DynSignal::new("test", vec![1.0, 3.0, 6.0, 10.0], 100, 0);
        let result = runtime.diff(&signal).unwrap();
        assert_eq!(result.samples, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cumsum() {
        let runtime = Runtime::new(RuntimeConfig::default());
        let signal = DynSignal::new("test", vec![1.0, 2.0, 3.0, 4.0], 100, 0);
        let result = runtime.cumsum(&signal).unwrap();
        assert_eq!(result.samples, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_zscore() {
        let runtime = Runtime::new(RuntimeConfig::default());
        let signal = DynSignal::new("test", vec![0.0, 1.0, 2.0, 3.0, 4.0], 100, 0);
        let result = runtime.zscore_normalize(&signal).unwrap();
        // Mean should be ~0, std should be ~1
        let mean: f64 = result.samples.iter().sum::<f64>() / result.samples.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_dominant_frequency() {
        let runtime = Runtime::new(RuntimeConfig::default());
        // 10 Hz sine wave at 1000 Hz sample rate
        let sample_rate = 1000;
        let freq = 10.0;
        let n = 1024;
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate as f64).sin())
            .collect();
        let signal = DynSignal::new("test", samples, sample_rate, 0);
        let result = runtime.dominant_frequency(&signal).unwrap();
        // Should be close to 10 Hz (within one bin)
        let df = sample_rate as f64 / n as f64;
        assert!((result.value - freq).abs() < df * 2.0);
    }
}
