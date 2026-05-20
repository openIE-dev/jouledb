//! FFT and spectral analysis operations

use super::{DspError, DspResult};
use crate::types::{Hertz, SampleRate};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// FFT operation
pub struct Fft {
    size: usize,
    window: WindowType,
}

/// Window type for FFT
#[derive(Debug, Clone, Copy, Default)]
pub enum WindowType {
    Rectangular,
    #[default]
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    Kaiser {
        beta: f64,
    },
    FlatTop,
}

impl Fft {
    /// Create new FFT operation
    pub fn new(size: usize) -> DspResult<Self> {
        if !size.is_power_of_two() {
            return Err(DspError::InvalidFftSize(size));
        }
        Ok(Self {
            size,
            window: WindowType::Hann,
        })
    }

    /// Create FFT with specific window
    pub fn with_window(size: usize, window: WindowType) -> DspResult<Self> {
        if !size.is_power_of_two() {
            return Err(DspError::InvalidFftSize(size));
        }
        Ok(Self { size, window })
    }

    /// Compute FFT of a signal
    pub fn compute(&self, signal: &[f64]) -> DspResult<Vec<Complex64>> {
        if signal.len() < self.size {
            return Err(DspError::SignalTooShort {
                needed: self.size,
                got: signal.len(),
            });
        }

        // Apply window
        let windowed: Vec<f64> = signal
            .iter()
            .take(self.size)
            .enumerate()
            .map(|(i, &x)| x * self.window_coefficient(i))
            .collect();

        // Convert to complex
        let mut buffer: Vec<Complex64> = windowed
            .into_iter()
            .map(|x| Complex64::new(x, 0.0))
            .collect();

        // Compute FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.size);
        fft.process(&mut buffer);

        // Normalize
        let norm = 1.0 / (self.size as f64).sqrt();
        for c in &mut buffer {
            *c *= norm;
        }

        Ok(buffer)
    }

    /// Compute magnitude spectrum
    pub fn magnitude(&self, signal: &[f64]) -> DspResult<Vec<f64>> {
        let spectrum = self.compute(signal)?;
        Ok(spectrum.iter().map(|c| c.norm()).collect())
    }

    /// Compute power spectrum (magnitude squared)
    pub fn power(&self, signal: &[f64]) -> DspResult<Vec<f64>> {
        let spectrum = self.compute(signal)?;
        Ok(spectrum.iter().map(|c| c.norm_sqr()).collect())
    }

    /// Compute power spectral density
    pub fn psd(&self, signal: &[f64], sample_rate: SampleRate) -> DspResult<Vec<f64>> {
        let power = self.power(signal)?;
        let _df = sample_rate.0 as f64 / self.size as f64;
        let enbw = self.equivalent_noise_bandwidth();

        // Scale to V²/Hz
        let scale = 2.0 / (sample_rate.0 as f64 * enbw);
        Ok(power.iter().map(|&p| p * scale).collect())
    }

    /// Get frequency bin centers
    pub fn frequency_bins(&self, sample_rate: SampleRate) -> Vec<Hertz> {
        let df = sample_rate.0 as f64 / self.size as f64;
        (0..self.size / 2 + 1)
            .map(|i| Hertz::new(i as f64 * df))
            .collect()
    }

    /// Window coefficient at index
    fn window_coefficient(&self, i: usize) -> f64 {
        let n = self.size as f64;
        let x = i as f64;

        match self.window {
            WindowType::Rectangular => 1.0,
            WindowType::Hann => 0.5 * (1.0 - (2.0 * std::f64::consts::PI * x / n).cos()),
            WindowType::Hamming => 0.54 - 0.46 * (2.0 * std::f64::consts::PI * x / n).cos(),
            WindowType::Blackman => {
                0.42 - 0.5 * (2.0 * std::f64::consts::PI * x / n).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * x / n).cos()
            }
            WindowType::BlackmanHarris => {
                0.35875 - 0.48829 * (2.0 * std::f64::consts::PI * x / n).cos()
                    + 0.14128 * (4.0 * std::f64::consts::PI * x / n).cos()
                    - 0.01168 * (6.0 * std::f64::consts::PI * x / n).cos()
            }
            WindowType::Kaiser { beta } => {
                let alpha = (n - 1.0) / 2.0;
                let arg = beta * (1.0 - ((x - alpha) / alpha).powi(2)).sqrt();
                bessel_i0(arg) / bessel_i0(beta)
            }
            WindowType::FlatTop => {
                let a0 = 0.21557895;
                let a1 = 0.41663158;
                let a2 = 0.277263158;
                let a3 = 0.083578947;
                let a4 = 0.006947368;
                a0 - a1 * (2.0 * std::f64::consts::PI * x / n).cos()
                    + a2 * (4.0 * std::f64::consts::PI * x / n).cos()
                    - a3 * (6.0 * std::f64::consts::PI * x / n).cos()
                    + a4 * (8.0 * std::f64::consts::PI * x / n).cos()
            }
        }
    }

    /// Equivalent noise bandwidth (ENBW) of window
    fn equivalent_noise_bandwidth(&self) -> f64 {
        match self.window {
            WindowType::Rectangular => 1.0,
            WindowType::Hann => 1.5,
            WindowType::Hamming => 1.36,
            WindowType::Blackman => 1.73,
            WindowType::BlackmanHarris => 2.0,
            WindowType::Kaiser { beta } => 1.0 + 0.1 * beta, // Approximate
            WindowType::FlatTop => 3.77,
        }
    }
}

/// Inverse FFT operation
pub struct Ifft {
    size: usize,
}

impl Ifft {
    pub fn new(size: usize) -> DspResult<Self> {
        if !size.is_power_of_two() {
            return Err(DspError::InvalidFftSize(size));
        }
        Ok(Self { size })
    }

    /// Compute inverse FFT
    pub fn compute(&self, spectrum: &[Complex64]) -> DspResult<Vec<f64>> {
        if spectrum.len() != self.size {
            return Err(DspError::InvalidParameter(format!(
                "Spectrum length {} doesn't match FFT size {}",
                spectrum.len(),
                self.size
            )));
        }

        let mut buffer: Vec<Complex64> = spectrum.to_vec();

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(self.size);
        ifft.process(&mut buffer);

        // Normalize and extract real part
        let norm = 1.0 / (self.size as f64).sqrt();
        Ok(buffer.iter().map(|c| c.re * norm).collect())
    }
}

/// Short-Time Fourier Transform parameters
#[derive(Debug, Clone)]
pub struct StftParams {
    /// FFT size
    pub fft_size: usize,
    /// Hop size (samples between frames)
    pub hop_size: usize,
    /// Window type
    pub window: WindowType,
}

impl Default for StftParams {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop_size: 256,
            window: WindowType::Hann,
        }
    }
}

/// Short-Time Fourier Transform
pub struct Stft {
    params: StftParams,
    fft: Fft,
}

impl Stft {
    pub fn new(params: StftParams) -> DspResult<Self> {
        let fft = Fft::with_window(params.fft_size, params.window)?;
        Ok(Self { params, fft })
    }

    /// Compute STFT, returning time-frequency matrix
    pub fn compute(&self, signal: &[f64]) -> DspResult<Vec<Vec<Complex64>>> {
        if signal.len() < self.params.fft_size {
            return Err(DspError::SignalTooShort {
                needed: self.params.fft_size,
                got: signal.len(),
            });
        }

        let num_frames = (signal.len() - self.params.fft_size) / self.params.hop_size + 1;
        let mut frames = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.params.hop_size;
            let end = start + self.params.fft_size;
            let frame = self.fft.compute(&signal[start..end])?;
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Compute magnitude spectrogram (positive frequencies only: N/2+1 bins)
    pub fn magnitude_spectrogram(&self, signal: &[f64]) -> DspResult<Vec<Vec<f64>>> {
        let stft = self.compute(signal)?;
        let num_bins = self.params.fft_size / 2 + 1;
        Ok(stft
            .into_iter()
            .map(|frame| frame.iter().take(num_bins).map(|c| c.norm()).collect())
            .collect())
    }

    /// Compute power spectrogram (dB scale)
    pub fn power_spectrogram_db(&self, signal: &[f64], ref_power: f64) -> DspResult<Vec<Vec<f64>>> {
        let stft = self.compute(signal)?;
        Ok(stft
            .into_iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|c| 10.0 * (c.norm_sqr() / ref_power).max(1e-10).log10())
                    .collect()
            })
            .collect())
    }

    /// Get time axis for spectrogram
    pub fn time_axis(&self, signal_len: usize, sample_rate: SampleRate) -> Vec<f64> {
        let num_frames = (signal_len - self.params.fft_size) / self.params.hop_size + 1;
        let dt = self.params.hop_size as f64 / sample_rate.0 as f64;
        (0..num_frames).map(|i| i as f64 * dt).collect()
    }

    /// Get frequency axis for spectrogram
    pub fn frequency_axis(&self, sample_rate: SampleRate) -> Vec<f64> {
        let df = sample_rate.0 as f64 / self.params.fft_size as f64;
        (0..self.params.fft_size / 2 + 1)
            .map(|i| i as f64 * df)
            .collect()
    }
}

/// Bessel I0 function (for Kaiser window)
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fft_sine() {
        let n = 1024;
        let fs = 1000.0;
        let f = 100.0; // 100 Hz sine wave

        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / fs).sin())
            .collect();

        let fft = Fft::new(n).unwrap();
        let magnitude = fft.magnitude(&signal).unwrap();

        // Find peak
        let (peak_bin, _peak_val) = magnitude[0..n / 2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let peak_freq = peak_bin as f64 * fs / n as f64;
        assert_relative_eq!(peak_freq, f, epsilon = 1.0);
    }

    #[test]
    fn test_ifft_roundtrip() {
        let n = 256;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let fft = Fft::with_window(n, WindowType::Rectangular).unwrap();
        let spectrum = fft.compute(&signal).unwrap();

        let ifft = Ifft::new(n).unwrap();
        let recovered = ifft.compute(&spectrum).unwrap();

        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_stft() {
        let n = 4096;
        let signal: Vec<f64> = (0..n).map(|i| (0.1 * i as f64).sin()).collect();

        let stft = Stft::new(StftParams::default()).unwrap();
        let spectrogram = stft.magnitude_spectrogram(&signal).unwrap();

        assert!(!spectrogram.is_empty());
        assert_eq!(spectrogram[0].len(), 513); // fft_size/2 + 1
    }
}
