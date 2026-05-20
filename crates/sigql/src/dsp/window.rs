//! Windowing functions for spectral analysis

use std::f64::consts::PI;

/// Window function type
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum WindowType {
    /// Rectangular (no windowing)
    Rectangular,
    /// Hann window (raised cosine)
    #[default]
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Blackman-Harris window
    BlackmanHarris,
    /// Kaiser window with beta parameter
    Kaiser { beta: f64 },
    /// Flat-top window (accurate amplitude)
    FlatTop,
    /// Gaussian window with sigma
    Gaussian { sigma: f64 },
    /// Tukey (tapered cosine) window
    Tukey { alpha: f64 },
}

impl WindowType {
    /// Generate window coefficients
    pub fn coefficients(&self, size: usize) -> Vec<f64> {
        let _n = size as f64;
        (0..size).map(|i| self.coefficient(i, size)).collect()
    }

    /// Single window coefficient at index
    pub fn coefficient(&self, i: usize, size: usize) -> f64 {
        let n = size as f64;
        let x = i as f64;

        match self {
            WindowType::Rectangular => 1.0,

            WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * x / (n - 1.0)).cos()),

            WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * x / (n - 1.0)).cos(),

            WindowType::Blackman => {
                0.42 - 0.5 * (2.0 * PI * x / (n - 1.0)).cos()
                    + 0.08 * (4.0 * PI * x / (n - 1.0)).cos()
            }

            WindowType::BlackmanHarris => {
                0.35875 - 0.48829 * (2.0 * PI * x / (n - 1.0)).cos()
                    + 0.14128 * (4.0 * PI * x / (n - 1.0)).cos()
                    - 0.01168 * (6.0 * PI * x / (n - 1.0)).cos()
            }

            WindowType::Kaiser { beta } => {
                let alpha = (n - 1.0) / 2.0;
                let arg = beta * (1.0 - ((x - alpha) / alpha).powi(2)).max(0.0).sqrt();
                bessel_i0(arg) / bessel_i0(*beta)
            }

            WindowType::FlatTop => {
                let a0 = 0.21557895;
                let a1 = 0.41663158;
                let a2 = 0.277263158;
                let a3 = 0.083578947;
                let a4 = 0.006947368;
                a0 - a1 * (2.0 * PI * x / (n - 1.0)).cos() + a2 * (4.0 * PI * x / (n - 1.0)).cos()
                    - a3 * (6.0 * PI * x / (n - 1.0)).cos()
                    + a4 * (8.0 * PI * x / (n - 1.0)).cos()
            }

            WindowType::Gaussian { sigma } => {
                let center = (n - 1.0) / 2.0;
                (-0.5 * ((x - center) / (sigma * center)).powi(2)).exp()
            }

            WindowType::Tukey { alpha } => {
                let alpha = alpha.clamp(0.0, 1.0);
                if alpha == 0.0 {
                    1.0
                } else if x < alpha * (n - 1.0) / 2.0 {
                    0.5 * (1.0 + (PI * (2.0 * x / (alpha * (n - 1.0)) - 1.0)).cos())
                } else if x > (n - 1.0) * (1.0 - alpha / 2.0) {
                    0.5 * (1.0 + (PI * (2.0 * x / (alpha * (n - 1.0)) - 2.0 / alpha + 1.0)).cos())
                } else {
                    1.0
                }
            }
        }
    }

    /// Equivalent noise bandwidth (ENBW) relative to rectangular
    pub fn enbw(&self) -> f64 {
        match self {
            WindowType::Rectangular => 1.0,
            WindowType::Hann => 1.5,
            WindowType::Hamming => 1.36,
            WindowType::Blackman => 1.73,
            WindowType::BlackmanHarris => 2.0,
            WindowType::Kaiser { beta } => 1.0 + 0.1 * beta, // Approximate
            WindowType::FlatTop => 3.77,
            WindowType::Gaussian { sigma } => 1.0 / (sigma * (2.0 * PI).sqrt()),
            WindowType::Tukey { alpha } => 1.0 + 0.5 * alpha, // Approximate
        }
    }

    /// Coherent gain (sum of coefficients / N)
    pub fn coherent_gain(&self, size: usize) -> f64 {
        self.coefficients(size).iter().sum::<f64>() / size as f64
    }

    /// Main lobe width in bins (approximate)
    pub fn main_lobe_bins(&self) -> f64 {
        match self {
            WindowType::Rectangular => 2.0,
            WindowType::Hann => 4.0,
            WindowType::Hamming => 4.0,
            WindowType::Blackman => 6.0,
            WindowType::BlackmanHarris => 8.0,
            WindowType::Kaiser { beta } => 2.0 + *beta * 0.5,
            WindowType::FlatTop => 10.0,
            WindowType::Gaussian { sigma } => 4.0 / *sigma,
            WindowType::Tukey { alpha } => 2.0 + 2.0 * *alpha,
        }
    }

    /// Sidelobe attenuation in dB (approximate)
    pub fn sidelobe_db(&self) -> f64 {
        match self {
            WindowType::Rectangular => -13.0,
            WindowType::Hann => -32.0,
            WindowType::Hamming => -43.0,
            WindowType::Blackman => -58.0,
            WindowType::BlackmanHarris => -92.0,
            WindowType::Kaiser { beta } => -10.0 - 10.0 * beta,
            WindowType::FlatTop => -93.0,
            WindowType::Gaussian { sigma } => -30.0 * sigma,
            WindowType::Tukey { alpha } => -13.0 - 20.0 * alpha,
        }
    }
}

/// Apply window to signal
pub fn apply_window(signal: &[f64], window: WindowType) -> Vec<f64> {
    let coeffs = window.coefficients(signal.len());
    signal
        .iter()
        .zip(coeffs.iter())
        .map(|(s, w)| s * w)
        .collect()
}

/// Apply window in-place
pub fn apply_window_inplace(signal: &mut [f64], window: WindowType) {
    let n = signal.len();
    for i in 0..n {
        signal[i] *= window.coefficient(i, n);
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
    fn test_rectangular_window() {
        let w = WindowType::Rectangular.coefficients(10);
        assert!(w.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_hann_window_endpoints() {
        let w = WindowType::Hann.coefficients(100);
        // Hann window should be 0 at endpoints
        assert_relative_eq!(w[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(w[99], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hann_window_symmetry() {
        let w = WindowType::Hann.coefficients(101);
        // Should be symmetric
        for i in 0..50 {
            assert_relative_eq!(w[i], w[100 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hann_window_peak() {
        let w = WindowType::Hann.coefficients(101);
        // Peak should be at center = 1.0
        assert_relative_eq!(w[50], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_window() {
        let signal = vec![1.0; 100];
        let windowed = apply_window(&signal, WindowType::Hann);

        // Should be same as window coefficients for unit signal
        let coeffs = WindowType::Hann.coefficients(100);
        for (w, c) in windowed.iter().zip(coeffs.iter()) {
            assert_relative_eq!(w, c, epsilon = 1e-10);
        }
    }
}
