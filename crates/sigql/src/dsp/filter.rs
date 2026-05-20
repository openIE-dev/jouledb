//! Digital filter design and application
//!
//! Implements IIR (biquad) filters with various design methods.

use super::{DspError, DspOperation, DspResult};
use crate::types::{DynSignal, Hertz, SampleRate};
use std::f64::consts::PI;

/// Filter types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
    Notch,
    Allpass,
    Peaking { gain_db: f64 },
    LowShelf { gain_db: f64 },
    HighShelf { gain_db: f64 },
}

/// Filter design method
#[derive(Debug, Clone, Copy, Default)]
pub enum FilterDesign {
    #[default]
    Butterworth,
    Chebyshev1 {
        ripple_db: f64,
    },
    Chebyshev2 {
        stopband_db: f64,
    },
    Bessel,
}

/// Biquad filter coefficients (Direct Form II)
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Default for BiquadCoeffs {
    fn default() -> Self {
        Self::passthrough()
    }
}

impl BiquadCoeffs {
    /// Passthrough (no filtering)
    pub fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    /// Design lowpass filter
    pub fn lowpass(cutoff: Hertz, sample_rate: SampleRate, q: f64) -> Self {
        let omega = 2.0 * PI * cutoff.0 / sample_rate.0 as f64;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Design highpass filter
    pub fn highpass(cutoff: Hertz, sample_rate: SampleRate, q: f64) -> Self {
        let omega = 2.0 * PI * cutoff.0 / sample_rate.0 as f64;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Design bandpass filter (constant skirt gain)
    pub fn bandpass(center: Hertz, bandwidth: Hertz, sample_rate: SampleRate) -> Self {
        let omega = 2.0 * PI * center.0 / sample_rate.0 as f64;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega * (2.0 * PI * bandwidth.0 / sample_rate.0 as f64 / 2.0).sinh();

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Design notch filter
    pub fn notch(center: Hertz, q: f64, sample_rate: SampleRate) -> Self {
        let omega = 2.0 * PI * center.0 / sample_rate.0 as f64;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = 1.0;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Design peaking EQ filter
    pub fn peaking(center: Hertz, q: f64, gain_db: f64, sample_rate: SampleRate) -> Self {
        let a = (10.0_f64).powf(gain_db / 40.0);
        let omega = 2.0 * PI * center.0 / sample_rate.0 as f64;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha / a;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Biquad filter state
#[derive(Debug, Clone, Default)]
struct BiquadState {
    z1: f64,
    z2: f64,
}

/// Single biquad filter section
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    coeffs: BiquadCoeffs,
    state: BiquadState,
}

impl BiquadFilter {
    /// Create new filter with given coefficients
    pub fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            state: BiquadState::default(),
        }
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.state = BiquadState::default();
    }

    /// Process single sample (Direct Form II Transposed)
    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        let y = self.coeffs.b0 * x + self.state.z1;
        self.state.z1 = self.coeffs.b1 * x - self.coeffs.a1 * y + self.state.z2;
        self.state.z2 = self.coeffs.b2 * x - self.coeffs.a2 * y;
        y
    }

    /// Process block of samples
    pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    /// Process block in place
    pub fn process_inplace(&mut self, buffer: &mut [f64]) {
        for x in buffer.iter_mut() {
            *x = self.process_sample(*x);
        }
    }
}

/// Cascaded biquad filter (for higher-order filters)
#[derive(Debug, Clone)]
pub struct CascadedBiquad {
    sections: Vec<BiquadFilter>,
}

impl CascadedBiquad {
    /// Create from list of biquad sections
    pub fn new(sections: Vec<BiquadCoeffs>) -> Self {
        Self {
            sections: sections.into_iter().map(BiquadFilter::new).collect(),
        }
    }

    /// Design Butterworth lowpass filter
    pub fn butterworth_lowpass(
        cutoff: Hertz,
        sample_rate: SampleRate,
        order: usize,
    ) -> DspResult<Self> {
        if order == 0 || order > 20 {
            return Err(DspError::InvalidParameter(format!(
                "Filter order must be 1-20, got {}",
                order
            )));
        }

        let nyquist = sample_rate.0 as f64 / 2.0;
        if cutoff.0 >= nyquist {
            return Err(DspError::InvalidParameter(format!(
                "Cutoff {} must be below Nyquist {}",
                cutoff.0, nyquist
            )));
        }

        let num_sections = (order + 1) / 2;
        let mut sections = Vec::with_capacity(num_sections);

        for k in 0..num_sections {
            let q = if order % 2 == 1 && k == num_sections - 1 {
                // First-order section (use Q=0.5 which gives a first-order response)
                0.5
            } else {
                // Second-order sections
                let angle = PI * (2.0 * (k as f64) + 1.0) / (2.0 * order as f64);
                1.0 / (2.0 * angle.cos())
            };
            sections.push(BiquadCoeffs::lowpass(cutoff, sample_rate, q));
        }

        Ok(Self::new(sections))
    }

    /// Design Butterworth highpass filter
    pub fn butterworth_highpass(
        cutoff: Hertz,
        sample_rate: SampleRate,
        order: usize,
    ) -> DspResult<Self> {
        if order == 0 || order > 20 {
            return Err(DspError::InvalidParameter(format!(
                "Filter order must be 1-20, got {}",
                order
            )));
        }

        let nyquist = sample_rate.0 as f64 / 2.0;
        if cutoff.0 >= nyquist {
            return Err(DspError::InvalidParameter(format!(
                "Cutoff {} must be below Nyquist {}",
                cutoff.0, nyquist
            )));
        }

        let num_sections = (order + 1) / 2;
        let mut sections = Vec::with_capacity(num_sections);

        for k in 0..num_sections {
            let q = if order % 2 == 1 && k == num_sections - 1 {
                0.5
            } else {
                let angle = PI * (2.0 * (k as f64) + 1.0) / (2.0 * order as f64);
                1.0 / (2.0 * angle.cos())
            };
            sections.push(BiquadCoeffs::highpass(cutoff, sample_rate, q));
        }

        Ok(Self::new(sections))
    }

    /// Design Butterworth bandpass filter
    pub fn butterworth_bandpass(
        low: Hertz,
        high: Hertz,
        sample_rate: SampleRate,
        order: usize,
    ) -> DspResult<Self> {
        // Bandpass = cascade of lowpass and highpass
        let lp = Self::butterworth_lowpass(high, sample_rate, order)?;
        let hp = Self::butterworth_highpass(low, sample_rate, order)?;

        let mut sections = lp.sections;
        sections.extend(hp.sections);

        Ok(Self {
            sections: sections.into_iter().map(|f| f).collect(),
        })
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }

    /// Process single sample through cascade
    #[inline]
    pub fn process_sample(&mut self, mut x: f64) -> f64 {
        for section in &mut self.sections {
            x = section.process_sample(x);
        }
        x
    }

    /// Process block of samples
    pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    /// Filter order (total)
    pub fn order(&self) -> usize {
        self.sections.len() * 2
    }

    /// Apply zero-phase filtering (forward-backward)
    pub fn filtfilt(&mut self, input: &[f64]) -> Vec<f64> {
        // Forward pass
        self.reset();
        let mut forward: Vec<f64> = input.iter().map(|&x| self.process_sample(x)).collect();

        // Reverse
        forward.reverse();

        // Backward pass
        self.reset();
        let mut backward: Vec<f64> = forward.iter().map(|&x| self.process_sample(x)).collect();

        // Reverse again
        backward.reverse();
        backward
    }
}

impl DspOperation for CascadedBiquad {
    fn apply(&self, signal: &DynSignal<f64>) -> DspResult<DynSignal<f64>> {
        let mut filter = self.clone();
        let filtered = filter.process(&signal.samples);
        Ok(DynSignal {
            samples: filtered,
            sample_rate: signal.sample_rate,
            channel: signal.channel.clone(),
            start_ns: signal.start_ns,
            metadata: signal.metadata.clone(),
        })
    }

    fn latency_samples(&self) -> usize {
        // Each biquad section adds 2 samples of latency
        self.sections.len() * 2
    }
}

/// Median filter (nonlinear)
#[derive(Debug, Clone)]
pub struct MedianFilter {
    window_size: usize,
}

impl MedianFilter {
    pub fn new(window_size: usize) -> DspResult<Self> {
        if window_size == 0 || window_size % 2 == 0 {
            return Err(DspError::InvalidParameter(
                "Median filter window size must be odd and > 0".to_string(),
            ));
        }
        Ok(Self { window_size })
    }

    pub fn process(&self, input: &[f64]) -> Vec<f64> {
        let half = self.window_size / 2;
        let mut output = Vec::with_capacity(input.len());
        let mut window = Vec::with_capacity(self.window_size);

        for i in 0..input.len() {
            window.clear();
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(input.len());

            for j in start..end {
                window.push(input[j]);
            }

            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            output.push(window[window.len() / 2]);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lowpass_dc_passthrough() {
        let coeffs = BiquadCoeffs::lowpass(Hertz::new(100.0), SampleRate::new(1000), 0.707);
        let mut filter = BiquadFilter::new(coeffs);

        // DC should pass through (after settling)
        for _ in 0..100 {
            filter.process_sample(1.0);
        }
        let output = filter.process_sample(1.0);
        assert_relative_eq!(output, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_butterworth_lowpass() {
        let sample_rate = SampleRate::new(1000);
        let cutoff = Hertz::new(100.0);
        let mut filter = CascadedBiquad::butterworth_lowpass(cutoff, sample_rate, 4).unwrap();

        // Generate test signal: DC + high frequency
        let signal: Vec<f64> = (0..1000)
            .map(|i| 1.0 + (2.0 * PI * 400.0 * i as f64 / 1000.0).sin())
            .collect();

        let filtered = filter.process(&signal);

        // After settling, output should be ~1.0 (DC) with minimal HF
        let mean: f64 = filtered[500..].iter().sum::<f64>() / 500.0;
        assert_relative_eq!(mean, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_notch_filter() {
        let sample_rate = SampleRate::new(1000);
        let coeffs = BiquadCoeffs::notch(Hertz::new(60.0), 30.0, sample_rate);
        let mut filter = BiquadFilter::new(coeffs);

        // Generate 60 Hz hum + 100 Hz signal
        let signal: Vec<f64> = (0..2000)
            .map(|i| {
                let t = i as f64 / 1000.0;
                (2.0 * PI * 60.0 * t).sin() + (2.0 * PI * 100.0 * t).sin()
            })
            .collect();

        let filtered = filter.process(&signal);

        // The 60 Hz should be attenuated
        // Compute power in 60 Hz band vs 100 Hz band
        // (simplified check)
        let late_samples = &filtered[1000..];
        let power: f64 =
            late_samples.iter().map(|x| x * x).sum::<f64>() / late_samples.len() as f64;

        // Power should be approximately half (100 Hz remains, 60 Hz removed)
        assert!(power < 0.6);
    }

    #[test]
    fn test_median_filter() {
        let filter = MedianFilter::new(3).unwrap();
        let input = vec![1.0, 100.0, 2.0, 3.0, 4.0]; // spike at index 1
        let output = filter.process(&input);

        // Spike should be removed
        assert!(output[1] < 10.0);
    }
}
