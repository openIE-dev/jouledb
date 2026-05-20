//! Resampling operations

use super::{DspOperation, DspResult};
use crate::types::{DynSignal, SampleRate};
use std::f64::consts::PI;

/// Resampling method
#[derive(Debug, Clone, Copy, Default)]
pub enum ResampleMethod {
    /// Nearest neighbor (fastest, lowest quality)
    Nearest,
    /// Linear interpolation
    Linear,
    /// Sinc interpolation with windowed kernel
    #[default]
    Sinc,
    /// Polyphase filter (best for large ratios)
    Polyphase,
}

/// Resampler configuration
#[derive(Debug, Clone)]
pub struct Resampler {
    /// Target sample rate
    pub target_rate: SampleRate,
    /// Resampling method
    pub method: ResampleMethod,
    /// Sinc kernel half-width (for Sinc method)
    pub sinc_width: usize,
}

impl Resampler {
    /// Create new resampler
    pub fn new(target_rate: SampleRate, method: ResampleMethod) -> Self {
        Self {
            target_rate,
            method,
            sinc_width: 16,
        }
    }

    /// Resample a signal
    pub fn resample(&self, signal: &[f64], source_rate: SampleRate) -> DspResult<Vec<f64>> {
        if source_rate.0 == self.target_rate.0 {
            return Ok(signal.to_vec());
        }

        match self.method {
            ResampleMethod::Nearest => self.resample_nearest(signal, source_rate),
            ResampleMethod::Linear => self.resample_linear(signal, source_rate),
            ResampleMethod::Sinc => self.resample_sinc(signal, source_rate),
            ResampleMethod::Polyphase => self.resample_polyphase(signal, source_rate),
        }
    }

    fn resample_nearest(&self, signal: &[f64], source_rate: SampleRate) -> DspResult<Vec<f64>> {
        let ratio = source_rate.0 as f64 / self.target_rate.0 as f64;
        let output_len = ((signal.len() as f64) / ratio).ceil() as usize;

        let mut output = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let src_idx = (i as f64 * ratio).round() as usize;
            let idx = src_idx.min(signal.len() - 1);
            output.push(signal[idx]);
        }

        Ok(output)
    }

    fn resample_linear(&self, signal: &[f64], source_rate: SampleRate) -> DspResult<Vec<f64>> {
        let ratio = source_rate.0 as f64 / self.target_rate.0 as f64;
        let output_len = ((signal.len() as f64) / ratio).ceil() as usize;

        let mut output = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let src_pos = i as f64 * ratio;
            let src_idx = src_pos.floor() as usize;
            let frac = src_pos - src_idx as f64;

            if src_idx + 1 < signal.len() {
                let v = signal[src_idx] * (1.0 - frac) + signal[src_idx + 1] * frac;
                output.push(v);
            } else if src_idx < signal.len() {
                output.push(signal[src_idx]);
            }
        }

        Ok(output)
    }

    fn resample_sinc(&self, signal: &[f64], source_rate: SampleRate) -> DspResult<Vec<f64>> {
        let ratio = source_rate.0 as f64 / self.target_rate.0 as f64;
        let output_len = ((signal.len() as f64) / ratio).ceil() as usize;

        // Use antialiasing filter if downsampling
        let cutoff = if ratio > 1.0 { 1.0 / ratio } else { 1.0 };
        let width = self.sinc_width;

        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_pos = i as f64 * ratio;
            let src_idx = src_pos.floor() as isize;

            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for j in -(width as isize)..=(width as isize) {
                let idx = src_idx + j;
                if idx >= 0 && (idx as usize) < signal.len() {
                    let t = src_pos - idx as f64;
                    let w = sinc(t * cutoff) * hann_window(t / width as f64);
                    sum += signal[idx as usize] * w;
                    weight_sum += w;
                }
            }

            if weight_sum > 0.0 {
                output.push(sum / weight_sum);
            } else {
                output.push(0.0);
            }
        }

        Ok(output)
    }

    fn resample_polyphase(&self, signal: &[f64], source_rate: SampleRate) -> DspResult<Vec<f64>> {
        // For now, fall back to sinc
        // Full polyphase implementation would use filter banks
        self.resample_sinc(signal, source_rate)
    }
}

impl DspOperation for Resampler {
    fn apply(&self, signal: &DynSignal<f64>) -> DspResult<DynSignal<f64>> {
        let source_rate = SampleRate::new(signal.sample_rate);
        let resampled = self.resample(&signal.samples, source_rate)?;
        Ok(DynSignal {
            samples: resampled,
            sample_rate: self.target_rate.0,
            channel: signal.channel.clone(),
            start_ns: signal.start_ns,
            metadata: signal.metadata.clone(),
        })
    }

    fn output_sample_rate(&self, _input_rate: SampleRate) -> SampleRate {
        self.target_rate
    }
}

/// Sinc function
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// Hann window for sinc kernel
#[inline]
fn hann_window(x: f64) -> f64 {
    if x.abs() > 1.0 {
        0.0
    } else {
        0.5 * (1.0 + (PI * x).cos())
    }
}

/// Decimation (integer factor downsampling with antialiasing)
pub fn decimate(signal: &[f64], factor: usize, sample_rate: SampleRate) -> DspResult<Vec<f64>> {
    use super::filter::CascadedBiquad;
    use crate::types::Hertz;

    if factor <= 1 {
        return Ok(signal.to_vec());
    }

    // Antialiasing filter
    let cutoff = Hertz::new(sample_rate.0 as f64 / (2.0 * factor as f64) * 0.9);
    let mut filter = CascadedBiquad::butterworth_lowpass(cutoff, sample_rate, 8)?;
    let filtered = filter.process(signal);

    // Downsample
    Ok(filtered.into_iter().step_by(factor).collect())
}

/// Interpolation (integer factor upsampling with antialiasing)
pub fn interpolate(signal: &[f64], factor: usize, sample_rate: SampleRate) -> DspResult<Vec<f64>> {
    use super::filter::CascadedBiquad;
    use crate::types::Hertz;

    if factor <= 1 {
        return Ok(signal.to_vec());
    }

    // Zero-stuff
    let mut upsampled = vec![0.0; signal.len() * factor];
    for (i, &x) in signal.iter().enumerate() {
        upsampled[i * factor] = x * factor as f64; // Scale to maintain energy
    }

    // Antialiasing filter at new sample rate
    let new_rate = SampleRate::new(sample_rate.0 * factor as u32);
    let cutoff = Hertz::new(sample_rate.0 as f64 / 2.0 * 0.9);
    let mut filter = CascadedBiquad::butterworth_lowpass(cutoff, new_rate, 8)?;

    Ok(filter.process(&upsampled))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_resample_2x() {
        let signal: Vec<f64> = (0..100).map(|i| (0.1 * i as f64).sin()).collect();
        let resampler = Resampler::new(SampleRate::new(200), ResampleMethod::Linear);

        let resampled = resampler.resample(&signal, SampleRate::new(100)).unwrap();

        // Should be approximately 2x length
        assert!(resampled.len() >= 190 && resampled.len() <= 210);
    }

    #[test]
    fn test_resample_preserves_dc() {
        let signal = vec![1.0; 1000];
        let resampler = Resampler::new(SampleRate::new(500), ResampleMethod::Sinc);

        let resampled = resampler.resample(&signal, SampleRate::new(1000)).unwrap();

        // DC should be preserved
        let mean: f64 = resampled.iter().sum::<f64>() / resampled.len() as f64;
        assert_relative_eq!(mean, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_decimate() {
        let signal: Vec<f64> = (0..1000).map(|_| 1.0).collect();
        let decimated = decimate(&signal, 4, SampleRate::new(1000)).unwrap();

        assert_eq!(decimated.len(), 250);
    }
}
