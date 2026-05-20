//! Spectrum Types - Frequency domain representations
//!
//! These types represent signals transformed into the frequency domain.
//! They are first-class queryable objects in SigQL.

use num_complex::Complex;
use smol_str::SmolStr;

use super::signal::SignalMetadata;
use super::units::{FrequencyBand, Hertz, SampleRate};

/// Single-sided amplitude spectrum from FFT
#[derive(Debug, Clone)]
pub struct Spectrum<T> {
    /// Complex frequency bins (positive frequencies only)
    pub bins: Vec<Complex<T>>,
    /// Frequency resolution (Hz per bin)
    pub resolution: Hertz,
    /// Original sample rate
    pub sample_rate: SampleRate,
    /// Channel identifier
    pub channel: SmolStr,
    /// Window function used
    pub window: WindowFunction,
    /// FFT size (may be larger than original signal due to zero-padding)
    pub fft_size: usize,
    /// Source signal metadata
    pub metadata: SignalMetadata,
}

impl<T: num_traits::Float + Copy> Spectrum<T> {
    /// Get frequency for a given bin index
    pub fn bin_frequency(&self, bin: usize) -> Hertz {
        Hertz(bin as f64 * self.resolution.0)
    }

    /// Get bin index for a given frequency (rounds to nearest)
    pub fn frequency_bin(&self, freq: Hertz) -> usize {
        let bin = (freq.0 / self.resolution.0).round() as usize;
        bin.min(self.bins.len() - 1)
    }

    /// Maximum frequency (Nyquist)
    pub fn max_frequency(&self) -> Hertz {
        self.sample_rate.nyquist()
    }

    /// Number of frequency bins
    pub fn len(&self) -> usize {
        self.bins.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
    }
}

impl Spectrum<f64> {
    /// Get magnitude at a specific frequency
    pub fn magnitude_at(&self, freq: Hertz) -> f64 {
        let bin = self.frequency_bin(freq);
        if bin < self.bins.len() {
            self.bins[bin].norm()
        } else {
            0.0
        }
    }

    /// Get phase at a specific frequency
    pub fn phase_at(&self, freq: Hertz) -> f64 {
        let bin = self.frequency_bin(freq);
        if bin < self.bins.len() {
            self.bins[bin].arg()
        } else {
            0.0
        }
    }

    /// Get power (magnitude squared) at a specific frequency
    pub fn power_at(&self, freq: Hertz) -> f64 {
        let bin = self.frequency_bin(freq);
        if bin < self.bins.len() {
            self.bins[bin].norm_sqr()
        } else {
            0.0
        }
    }

    /// Compute Power Spectral Density (V²/Hz)
    pub fn to_psd(&self) -> PowerSpectralDensity {
        let psd: Vec<f64> = self
            .bins
            .iter()
            .map(|c| c.norm_sqr() / (self.resolution.0 * self.fft_size as f64))
            .collect();

        PowerSpectralDensity {
            bins: psd,
            resolution: self.resolution,
            sample_rate: self.sample_rate,
            channel: self.channel.clone(),
        }
    }

    /// Get magnitude spectrum
    pub fn magnitude(&self) -> Vec<f64> {
        self.bins.iter().map(|c| c.norm()).collect()
    }

    /// Get phase spectrum
    pub fn phase(&self) -> Vec<f64> {
        self.bins.iter().map(|c| c.arg()).collect()
    }

    /// Get power spectrum
    pub fn power(&self) -> Vec<f64> {
        self.bins.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Integrate power in a frequency band
    pub fn band_power(&self, band: FrequencyBand) -> f64 {
        let low_bin = self.frequency_bin(band.low);
        let high_bin = self.frequency_bin(band.high);

        self.bins[low_bin..=high_bin.min(self.bins.len() - 1)]
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            * self.resolution.0
    }

    /// Find dominant (peak) frequency
    pub fn dominant_frequency(&self) -> (Hertz, f64) {
        let (max_bin, max_power) = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.norm_sqr()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, 0.0));

        (self.bin_frequency(max_bin), max_power.sqrt())
    }

    /// Find dominant frequency within a band
    pub fn dominant_frequency_in_band(&self, band: FrequencyBand) -> (Hertz, f64) {
        let low_bin = self.frequency_bin(band.low);
        let high_bin = self.frequency_bin(band.high).min(self.bins.len() - 1);

        let (max_bin, max_power) = self.bins[low_bin..=high_bin]
            .iter()
            .enumerate()
            .map(|(i, c)| (i + low_bin, c.norm_sqr()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((low_bin, 0.0));

        (self.bin_frequency(max_bin), max_power.sqrt())
    }

    /// Compute spectral centroid (center of mass)
    pub fn spectral_centroid(&self) -> Hertz {
        let total_power: f64 = self.bins.iter().map(|c| c.norm_sqr()).sum();
        if total_power < f64::EPSILON {
            return Hertz(0.0);
        }

        let weighted_sum: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, c)| (i as f64) * self.resolution.0 * c.norm_sqr())
            .sum();

        Hertz(weighted_sum / total_power)
    }

    /// Compute spectral entropy (complexity/randomness)
    pub fn spectral_entropy(&self) -> f64 {
        let total_power: f64 = self.bins.iter().map(|c| c.norm_sqr()).sum();
        if total_power < f64::EPSILON {
            return 0.0;
        }

        let entropy: f64 = self
            .bins
            .iter()
            .map(|c| {
                let p = c.norm_sqr() / total_power;
                if p > f64::EPSILON { -p * p.ln() } else { 0.0 }
            })
            .sum();

        // Normalize by max possible entropy (uniform distribution)
        entropy / (self.bins.len() as f64).ln()
    }

    /// Compute spectral flatness (tone vs noise)
    pub fn spectral_flatness(&self) -> f64 {
        let n = self.bins.len() as f64;
        let powers: Vec<f64> = self.bins.iter().map(|c| c.norm_sqr()).collect();

        let arithmetic_mean: f64 = powers.iter().sum::<f64>() / n;
        if arithmetic_mean < f64::EPSILON {
            return 0.0;
        }

        let log_sum: f64 = powers
            .iter()
            .filter(|&&p| p > f64::EPSILON)
            .map(|p| p.ln())
            .sum();
        let geometric_mean = (log_sum / n).exp();

        geometric_mean / arithmetic_mean
    }
}

/// Power Spectral Density (real-valued, power per Hz)
#[derive(Debug, Clone)]
pub struct PowerSpectralDensity {
    /// PSD values (V²/Hz or equivalent)
    pub bins: Vec<f64>,
    /// Frequency resolution
    pub resolution: Hertz,
    /// Original sample rate
    pub sample_rate: SampleRate,
    /// Channel identifier
    pub channel: SmolStr,
}

/// Spectrogram - time-frequency representation
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Time-frequency magnitude matrix [time][frequency]
    pub magnitudes: Vec<Vec<f64>>,
    /// Time resolution (seconds per frame)
    pub time_resolution: f64,
    /// Frequency resolution (Hz per bin)
    pub freq_resolution: Hertz,
    /// Sample rate of original signal
    pub sample_rate: SampleRate,
    /// Start timestamp
    pub start_ns: i64,
    /// Window function used
    pub window: WindowFunction,
    /// FFT size
    pub fft_size: usize,
    /// Hop size (samples)
    pub hop_size: usize,
    /// Channel identifier
    pub channel: SmolStr,
}

impl Spectrogram {
    /// Number of time frames
    pub fn num_frames(&self) -> usize {
        self.magnitudes.len()
    }

    /// Number of frequency bins
    pub fn num_bins(&self) -> usize {
        if self.magnitudes.is_empty() {
            0
        } else {
            self.magnitudes[0].len()
        }
    }

    /// Get timestamp for a frame
    pub fn frame_time_ns(&self, frame: usize) -> i64 {
        self.start_ns + (frame as f64 * self.time_resolution * 1_000_000_000.0) as i64
    }

    /// Get frequency for a bin
    pub fn bin_frequency(&self, bin: usize) -> Hertz {
        Hertz(bin as f64 * self.freq_resolution.0)
    }

    /// Get magnitude at specific time/frequency
    pub fn magnitude_at(&self, frame: usize, bin: usize) -> f64 {
        self.magnitudes
            .get(frame)
            .and_then(|f| f.get(bin))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get average spectrum (collapse time dimension)
    pub fn mean_spectrum(&self) -> Vec<f64> {
        if self.magnitudes.is_empty() {
            return Vec::new();
        }

        let num_bins = self.num_bins();
        let num_frames = self.num_frames() as f64;

        (0..num_bins)
            .map(|bin| {
                self.magnitudes
                    .iter()
                    .map(|frame| frame.get(bin).copied().unwrap_or(0.0))
                    .sum::<f64>()
                    / num_frames
            })
            .collect()
    }

    /// Get time envelope at specific frequency
    pub fn envelope_at_freq(&self, freq: Hertz) -> Vec<f64> {
        let bin = (freq.0 / self.freq_resolution.0).round() as usize;
        self.magnitudes
            .iter()
            .map(|frame| frame.get(bin).copied().unwrap_or(0.0))
            .collect()
    }

    /// Extract band power over time
    pub fn band_power_over_time(&self, band: FrequencyBand) -> Vec<f64> {
        let low_bin = (band.low.0 / self.freq_resolution.0).round() as usize;
        let high_bin = (band.high.0 / self.freq_resolution.0).round() as usize;

        self.magnitudes
            .iter()
            .map(|frame| {
                frame[low_bin..=high_bin.min(frame.len() - 1)]
                    .iter()
                    .map(|m| m * m)
                    .sum::<f64>()
            })
            .collect()
    }
}

/// Window function types for spectral analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowFunction {
    /// Rectangular (no windowing)
    Rectangular,
    /// Hann window (cosine-squared)
    #[default]
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Kaiser window with parameter beta
    Kaiser(u8), // Store beta * 10 for simple repr
    /// Flat-top window (amplitude accuracy)
    FlatTop,
    /// Gaussian window
    Gaussian(u8), // Store sigma * 10
}

impl WindowFunction {
    /// Generate window coefficients
    pub fn coefficients(&self, size: usize) -> Vec<f64> {
        let n = size as f64;
        match self {
            Self::Rectangular => vec![1.0; size],
            Self::Hann => (0..size)
                .map(|i| 0.5 * (1.0 - (2.0 * core::f64::consts::PI * i as f64 / (n - 1.0)).cos()))
                .collect(),
            Self::Hamming => (0..size)
                .map(|i| 0.54 - 0.46 * (2.0 * core::f64::consts::PI * i as f64 / (n - 1.0)).cos())
                .collect(),
            Self::Blackman => (0..size)
                .map(|i| {
                    let x = 2.0 * core::f64::consts::PI * i as f64 / (n - 1.0);
                    0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
                })
                .collect(),
            Self::Kaiser(beta_10) => {
                // Simplified Kaiser window (approximation)
                let beta = *beta_10 as f64 / 10.0;
                (0..size)
                    .map(|i| {
                        let x = 2.0 * i as f64 / (n - 1.0) - 1.0;
                        let arg = beta * (1.0 - x * x).sqrt();
                        // Approximation of I0(arg) / I0(beta)
                        besseli0_approx(arg) / besseli0_approx(beta)
                    })
                    .collect()
            }
            Self::FlatTop => (0..size)
                .map(|i| {
                    let x = 2.0 * core::f64::consts::PI * i as f64 / (n - 1.0);
                    0.21557895 - 0.41663158 * x.cos() + 0.277263158 * (2.0 * x).cos()
                        - 0.083578947 * (3.0 * x).cos()
                        + 0.006947368 * (4.0 * x).cos()
                })
                .collect(),
            Self::Gaussian(sigma_10) => {
                let sigma = *sigma_10 as f64 / 10.0;
                (0..size)
                    .map(|i| {
                        let x = (i as f64 - (n - 1.0) / 2.0) / (sigma * (n - 1.0) / 2.0);
                        (-0.5 * x * x).exp()
                    })
                    .collect()
            }
        }
    }

    /// Coherent gain (sum of coefficients / N)
    pub fn coherent_gain(&self, size: usize) -> f64 {
        let coeffs = self.coefficients(size);
        coeffs.iter().sum::<f64>() / size as f64
    }

    /// Equivalent noise bandwidth (for PSD normalization)
    pub fn enbw(&self, size: usize) -> f64 {
        let coeffs = self.coefficients(size);
        let sum: f64 = coeffs.iter().sum();
        let sum_sq: f64 = coeffs.iter().map(|c| c * c).sum();
        (size as f64) * sum_sq / (sum * sum)
    }
}

/// Approximate modified Bessel function I0
fn besseli0_approx(x: f64) -> f64 {
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

/// Cross-spectrum between two signals
#[derive(Debug, Clone)]
pub struct CrossSpectrum {
    /// Complex cross-spectral density bins
    pub bins: Vec<Complex<f64>>,
    /// Frequency resolution
    pub resolution: Hertz,
    /// Sample rate
    pub sample_rate: SampleRate,
    /// First channel
    pub channel_a: SmolStr,
    /// Second channel
    pub channel_b: SmolStr,
}

impl CrossSpectrum {
    /// Compute coherence (magnitude-squared coherence)
    pub fn coherence(
        &self,
        psd_a: &PowerSpectralDensity,
        psd_b: &PowerSpectralDensity,
    ) -> Vec<f64> {
        self.bins
            .iter()
            .zip(psd_a.bins.iter())
            .zip(psd_b.bins.iter())
            .map(|((csd, pa), pb)| {
                let denom = pa * pb;
                if denom > f64::EPSILON {
                    csd.norm_sqr() / denom
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compute phase difference
    pub fn phase_difference(&self) -> Vec<f64> {
        self.bins.iter().map(|c| c.arg()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_functions() {
        let hann = WindowFunction::Hann.coefficients(64);
        assert_eq!(hann.len(), 64);
        assert!(hann[0] < 0.01); // Should be near 0 at edges
        assert!((hann[32] - 1.0).abs() < 0.01); // Should be 1 at center
    }

    #[test]
    fn test_spectral_entropy() {
        // Uniform spectrum should have high entropy
        let uniform_spec = Spectrum {
            bins: vec![Complex::new(1.0, 0.0); 64],
            resolution: Hertz(1.0),
            sample_rate: SampleRate(128),
            channel: "test".into(),
            window: WindowFunction::Hann,
            fft_size: 64,
            metadata: SignalMetadata::default(),
        };

        let entropy = uniform_spec.spectral_entropy();
        assert!(entropy > 0.99); // Should be very close to 1
    }

    #[test]
    fn test_frequency_band() {
        let band = FrequencyBand::parkinsonian_tremor();
        assert_eq!(band.low.0, 4.0);
        assert_eq!(band.high.0, 12.0);
        assert!((band.center().0 - 8.0).abs() < 0.001);
    }
}
