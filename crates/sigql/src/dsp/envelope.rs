//! Envelope extraction and Hilbert transform

use super::{DspError, DspOperation, DspResult};
use crate::types::{DynSignal, SampleRate};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Hilbert transform implementation
pub struct HilbertTransform {
    size: usize,
}

impl HilbertTransform {
    /// Create new Hilbert transform
    pub fn new(size: usize) -> DspResult<Self> {
        if !size.is_power_of_two() {
            return Err(DspError::InvalidFftSize(size));
        }
        Ok(Self { size })
    }

    /// Compute analytic signal (signal + j*hilbert(signal))
    pub fn analytic_signal(&self, signal: &[f64]) -> DspResult<Vec<Complex64>> {
        if signal.len() < self.size {
            return Err(DspError::SignalTooShort {
                needed: self.size,
                got: signal.len(),
            });
        }

        // Use rustfft directly for proper normalization control
        let mut planner = rustfft::FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(self.size);

        // Convert to complex
        let mut spectrum: Vec<Complex64> = signal[..self.size]
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        // Forward FFT (no normalization by rustfft)
        fft_forward.process(&mut spectrum);

        // Apply Hilbert transform in frequency domain
        // H[k] = -j*sign(k) for k != 0, N/2
        // This zeroes out negative frequencies and doubles positive frequencies

        let n = self.size;
        let half = n / 2;

        // DC component unchanged
        // spectrum[0] = spectrum[0];

        // Positive frequencies: multiply by 2
        for k in 1..half {
            spectrum[k] *= 2.0;
        }

        // Nyquist component unchanged (if even length)
        // spectrum[half] = spectrum[half];

        // Negative frequencies: set to zero
        for k in (half + 1)..n {
            spectrum[k] = Complex64::new(0.0, 0.0);
        }

        // Inverse FFT
        let ifft_op = planner.plan_fft_inverse(self.size);
        ifft_op.process(&mut spectrum);

        // Normalize by N (standard IFFT normalization)
        let norm = 1.0 / self.size as f64;
        for c in &mut spectrum {
            *c *= norm;
        }

        Ok(spectrum)
    }

    /// Compute Hilbert transform (imaginary part of analytic signal)
    pub fn transform(&self, signal: &[f64]) -> DspResult<Vec<f64>> {
        let analytic = self.analytic_signal(signal)?;
        Ok(analytic.iter().map(|c| c.im).collect())
    }

    /// Compute instantaneous envelope (magnitude of analytic signal)
    pub fn envelope(&self, signal: &[f64]) -> DspResult<Vec<f64>> {
        let analytic = self.analytic_signal(signal)?;
        Ok(analytic.iter().map(|c| c.norm()).collect())
    }

    /// Compute instantaneous phase
    pub fn instantaneous_phase(&self, signal: &[f64]) -> DspResult<Vec<f64>> {
        let analytic = self.analytic_signal(signal)?;
        Ok(analytic.iter().map(|c| c.arg()).collect())
    }

    /// Compute instantaneous frequency (derivative of phase)
    pub fn instantaneous_frequency(
        &self,
        signal: &[f64],
        sample_rate: SampleRate,
    ) -> DspResult<Vec<f64>> {
        let phase = self.instantaneous_phase(signal)?;

        // Unwrap phase and differentiate
        let mut unwrapped = vec![0.0; phase.len()];
        unwrapped[0] = phase[0];

        for i in 1..phase.len() {
            let mut diff = phase[i] - phase[i - 1];

            // Unwrap: adjust jumps > π
            while diff > PI {
                diff -= 2.0 * PI;
            }
            while diff < -PI {
                diff += 2.0 * PI;
            }

            unwrapped[i] = unwrapped[i - 1] + diff;
        }

        // Differentiate and convert to Hz
        let dt = 1.0 / sample_rate.0 as f64;
        let mut freq = vec![0.0; phase.len()];

        for i in 1..phase.len() - 1 {
            // Central difference
            freq[i] = (unwrapped[i + 1] - unwrapped[i - 1]) / (2.0 * dt * 2.0 * PI);
        }

        // Edge cases: forward/backward difference
        freq[0] = (unwrapped[1] - unwrapped[0]) / (dt * 2.0 * PI);
        let last = phase.len() - 1;
        freq[last] = (unwrapped[last] - unwrapped[last - 1]) / (dt * 2.0 * PI);

        Ok(freq)
    }
}

/// Envelope extraction methods
pub struct EnvelopeExtractor;

impl EnvelopeExtractor {
    /// Extract envelope using Hilbert transform
    pub fn hilbert(signal: &[f64]) -> DspResult<Vec<f64>> {
        // Round up to next power of 2
        let size = signal.len().next_power_of_two();

        // Pad signal
        let mut padded = signal.to_vec();
        padded.resize(size, 0.0);

        let hilbert = HilbertTransform::new(size)?;
        let mut envelope = hilbert.envelope(&padded)?;

        // Trim to original length
        envelope.truncate(signal.len());
        Ok(envelope)
    }

    /// Extract envelope using peak detection
    pub fn peaks(signal: &[f64], min_distance: usize) -> Vec<f64> {
        let n = signal.len();
        let mut envelope = vec![0.0; n];
        let mut peaks = Vec::new();

        // Find local maxima
        for i in 1..n - 1 {
            if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > 0.0 {
                if peaks.is_empty() || i - peaks.last().unwrap() >= min_distance {
                    peaks.push(i);
                }
            }
        }

        // Interpolate between peaks
        if peaks.is_empty() {
            return envelope;
        }

        // Before first peak
        for i in 0..peaks[0] {
            envelope[i] = signal[peaks[0]];
        }

        // Between peaks: linear interpolation
        for w in peaks.windows(2) {
            let (i1, i2) = (w[0], w[1]);
            let (v1, v2) = (signal[i1], signal[i2]);

            for i in i1..=i2 {
                let t = (i - i1) as f64 / (i2 - i1) as f64;
                envelope[i] = v1 + t * (v2 - v1);
            }
        }

        // After last peak
        let last = *peaks.last().unwrap();
        for i in last..n {
            envelope[i] = signal[last];
        }

        envelope
    }

    /// Extract envelope using rectification + lowpass
    pub fn rectify_lowpass(
        signal: &[f64],
        cutoff_ratio: f64,
        sample_rate: SampleRate,
    ) -> DspResult<Vec<f64>> {
        use super::filter::CascadedBiquad;
        use crate::types::Hertz;

        // Full-wave rectification
        let rectified: Vec<f64> = signal.iter().map(|&x| x.abs()).collect();

        // Lowpass filter
        let cutoff = Hertz::new(cutoff_ratio * sample_rate.0 as f64);
        let mut filter = CascadedBiquad::butterworth_lowpass(cutoff, sample_rate, 4)?;

        Ok(filter.process(&rectified))
    }

    /// Smooth envelope with moving average
    pub fn smooth(envelope: &[f64], window_size: usize) -> Vec<f64> {
        if window_size <= 1 || envelope.len() < window_size {
            return envelope.to_vec();
        }

        let mut smoothed = Vec::with_capacity(envelope.len());
        let half = window_size / 2;

        for i in 0..envelope.len() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(envelope.len());
            let sum: f64 = envelope[start..end].iter().sum();
            smoothed.push(sum / (end - start) as f64);
        }

        smoothed
    }
}

impl DspOperation for HilbertTransform {
    fn apply(&self, signal: &DynSignal<f64>) -> DspResult<DynSignal<f64>> {
        let envelope = self.envelope(&signal.samples)?;
        Ok(DynSignal {
            samples: envelope,
            sample_rate: signal.sample_rate,
            channel: signal.channel.clone(),
            start_ns: signal.start_ns,
            metadata: signal.metadata.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hilbert_sine() {
        let n = 1024;
        let f = 10.0;
        let fs = 1000.0;

        // Pure sine wave
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / fs).sin())
            .collect();

        let hilbert = HilbertTransform::new(n).unwrap();
        let envelope = hilbert.envelope(&signal).unwrap();

        // Envelope of sine wave should be approximately 1.0
        // (excluding edge effects)
        let mean_envelope: f64 = envelope[100..900].iter().sum::<f64>() / 800.0;
        assert_relative_eq!(mean_envelope, 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_hilbert_am_signal() {
        let n = 2048;
        let f_carrier = 100.0;
        let f_mod = 5.0;
        let fs = 1000.0;

        // AM signal: (1 + 0.5*cos(mod))*cos(carrier)
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (1.0 + 0.5 * (2.0 * PI * f_mod * t).cos()) * (2.0 * PI * f_carrier * t).cos()
            })
            .collect();

        let hilbert = HilbertTransform::new(n).unwrap();
        let envelope = hilbert.envelope(&signal).unwrap();

        // Envelope should follow the modulation: 1 + 0.5*cos(mod)
        // Check min and max of envelope (excluding edges)
        let min_env = envelope[200..1800]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_env = envelope[200..1800]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        assert_relative_eq!(min_env, 0.5, epsilon = 0.1);
        assert_relative_eq!(max_env, 1.5, epsilon = 0.1);
    }

    #[test]
    fn test_instantaneous_frequency() {
        let n = 1024;
        let f = 50.0;
        let fs = 1000.0;

        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / fs).sin())
            .collect();

        let hilbert = HilbertTransform::new(n).unwrap();
        let inst_freq = hilbert
            .instantaneous_frequency(&signal, SampleRate::new(fs as u32))
            .unwrap();

        // Instantaneous frequency should be approximately f Hz
        let mean_freq: f64 = inst_freq[100..900].iter().sum::<f64>() / 800.0;
        assert_relative_eq!(mean_freq, f, epsilon = 1.0);
    }
}
