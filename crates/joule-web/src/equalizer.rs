//! Parametric equalizer with biquad filters.
//!
//! Supports N bands with configurable frequency, gain, Q factor, and filter type
//! (peak, low shelf, high shelf, notch). Computes biquad coefficients from the
//! Audio EQ Cookbook and processes samples through cascaded second-order sections.

use std::f64::consts::PI;

// ── Filter Type ─────────────────────────────────────────────────

/// Type of EQ band filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    Peak,
    LowShelf,
    HighShelf,
    Notch,
}

// ── Band Parameters ─────────────────────────────────────────────

/// Parameters for a single EQ band.
#[derive(Debug, Clone, Copy)]
pub struct BandParams {
    pub frequency: f64,
    pub gain_db: f64,
    pub q_factor: f64,
    pub filter_type: FilterType,
}

impl BandParams {
    pub fn peak(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            frequency,
            gain_db,
            q_factor,
            filter_type: FilterType::Peak,
        }
    }

    pub fn low_shelf(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            frequency,
            gain_db,
            q_factor,
            filter_type: FilterType::LowShelf,
        }
    }

    pub fn high_shelf(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            frequency,
            gain_db,
            q_factor,
            filter_type: FilterType::HighShelf,
        }
    }

    pub fn notch(frequency: f64, q_factor: f64) -> Self {
        Self {
            frequency,
            gain_db: 0.0,
            q_factor,
            filter_type: FilterType::Notch,
        }
    }
}

// ── Biquad Coefficients ─────────────────────────────────────────

/// Biquad filter coefficients (normalized so a0 = 1).
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl BiquadCoeffs {
    /// Compute biquad coefficients from band parameters and sample rate.
    /// Based on the Audio EQ Cookbook by Robert Bristow-Johnson.
    pub fn compute(params: &BandParams, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * params.frequency / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * params.q_factor);
        let amp = 10.0f64.powf(params.gain_db / 40.0);

        let (b0, b1, b2, a0, a1, a2) = match params.filter_type {
            FilterType::Peak => {
                let b0 = 1.0 + alpha * amp;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * amp;
                let a0 = 1.0 + alpha / amp;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / amp;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::LowShelf => {
                let two_sqrt_a_alpha = 2.0 * amp.sqrt() * alpha;
                let b0 = amp * ((amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = 2.0 * amp * ((amp - 1.0) - (amp + 1.0) * cos_w0);
                let b2 = amp * ((amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = -2.0 * ((amp - 1.0) + (amp + 1.0) * cos_w0);
                let a2 = (amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::HighShelf => {
                let two_sqrt_a_alpha = 2.0 * amp.sqrt() * alpha;
                let b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w0);
                let b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w0);
                let a2 = (amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        // Normalize by a0
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Compute the magnitude response at a given frequency.
    pub fn magnitude_response(&self, frequency: f64, sample_rate: f64) -> f64 {
        let w = 2.0 * PI * frequency / sample_rate;
        let cos_w = w.cos();
        let cos_2w = (2.0 * w).cos();

        let num = self.b0 * self.b0
            + self.b1 * self.b1
            + self.b2 * self.b2
            + 2.0 * (self.b0 * self.b1 + self.b1 * self.b2) * cos_w
            + 2.0 * self.b0 * self.b2 * cos_2w;

        let den = 1.0
            + self.a1 * self.a1
            + self.a2 * self.a2
            + 2.0 * (self.a1 + self.a1 * self.a2) * cos_w
            + 2.0 * self.a2 * cos_2w;

        if den <= 0.0 {
            return 0.0;
        }

        (num / den).sqrt()
    }

    /// Magnitude response in dB.
    pub fn magnitude_response_db(&self, frequency: f64, sample_rate: f64) -> f64 {
        let mag = self.magnitude_response(frequency, sample_rate);
        if mag <= 0.0 {
            return f64::NEG_INFINITY;
        }
        20.0 * mag.log10()
    }
}

// ── Biquad Filter State ─────────────────────────────────────────

/// State for a single biquad filter (Direct Form II Transposed).
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    coeffs: BiquadCoeffs,
    z1: f64,
    z2: f64,
}

impl BiquadFilter {
    pub fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs) {
        self.coeffs = coeffs;
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let output = self.coeffs.b0 * input + self.z1;
        self.z1 = self.coeffs.b1 * input - self.coeffs.a1 * output + self.z2;
        self.z2 = self.coeffs.b2 * input - self.coeffs.a2 * output;
        output
    }

    /// Process a buffer of samples in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s as f64) as f32;
        }
    }

    pub fn coeffs(&self) -> &BiquadCoeffs {
        &self.coeffs
    }
}

// ── Parametric EQ ───────────────────────────────────────────────

/// A single EQ band with parameters and filter state.
#[derive(Debug, Clone)]
pub struct EqBand {
    pub params: BandParams,
    filter: BiquadFilter,
    sample_rate: f64,
}

impl EqBand {
    pub fn new(params: BandParams, sample_rate: f64) -> Self {
        let coeffs = BiquadCoeffs::compute(&params, sample_rate);
        Self {
            params,
            filter: BiquadFilter::new(coeffs),
            sample_rate,
        }
    }

    /// Update band parameters and recompute coefficients.
    pub fn set_params(&mut self, params: BandParams) {
        self.params = params;
        let coeffs = BiquadCoeffs::compute(&params, self.sample_rate);
        self.filter.set_coeffs(coeffs);
    }

    pub fn reset(&mut self) {
        self.filter.reset();
    }
}

/// Parametric equalizer with N cascaded biquad bands.
#[derive(Debug, Clone)]
pub struct ParametricEq {
    bands: Vec<EqBand>,
    sample_rate: f64,
}

impl ParametricEq {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            bands: Vec::new(),
            sample_rate,
        }
    }

    /// Add a band and return its index.
    pub fn add_band(&mut self, params: BandParams) -> usize {
        let band = EqBand::new(params, self.sample_rate);
        self.bands.push(band);
        self.bands.len() - 1
    }

    /// Update a band's parameters.
    pub fn set_band(&mut self, index: usize, params: BandParams) {
        if let Some(band) = self.bands.get_mut(index) {
            band.set_params(params);
        }
    }

    /// Remove a band by index.
    pub fn remove_band(&mut self, index: usize) {
        if index < self.bands.len() {
            self.bands.remove(index);
        }
    }

    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Process samples through all bands in series.
    pub fn process(&mut self, samples: &mut [f32]) {
        for band in &mut self.bands {
            band.filter.process(samples);
        }
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    /// Compute the combined magnitude response at a given frequency (in dB).
    pub fn frequency_response_db(&self, frequency: f64) -> f64 {
        let mut total_db = 0.0;
        for band in &self.bands {
            let coeffs = BiquadCoeffs::compute(&band.params, self.sample_rate);
            total_db += coeffs.magnitude_response_db(frequency, self.sample_rate);
        }
        total_db
    }

    /// Compute frequency response at multiple points for plotting.
    pub fn frequency_response_curve(
        &self,
        frequencies: &[f64],
    ) -> Vec<(f64, f64)> {
        frequencies
            .iter()
            .map(|f| (*f, self.frequency_response_db(*f)))
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biquad_peak_unity_at_zero_gain() {
        let params = BandParams::peak(1000.0, 0.0, 1.0);
        let coeffs = BiquadCoeffs::compute(&params, 44100.0);
        // At 0 dB gain, peak filter should be unity
        let mag = coeffs.magnitude_response(500.0, 44100.0);
        assert!((mag - 1.0).abs() < 0.01, "Expected unity, got {}", mag);
    }

    #[test]
    fn biquad_peak_boosts_at_center() {
        let params = BandParams::peak(1000.0, 12.0, 1.0);
        let coeffs = BiquadCoeffs::compute(&params, 44100.0);
        let mag_center = coeffs.magnitude_response(1000.0, 44100.0);
        let mag_off = coeffs.magnitude_response(100.0, 44100.0);
        assert!(
            mag_center > mag_off,
            "Peak at center ({}) should be greater than off-center ({})",
            mag_center,
            mag_off
        );
    }

    #[test]
    fn biquad_notch_cuts_at_center() {
        let params = BandParams::notch(1000.0, 10.0);
        let coeffs = BiquadCoeffs::compute(&params, 44100.0);
        let mag = coeffs.magnitude_response(1000.0, 44100.0);
        assert!(mag < 0.2, "Notch at center should be near zero, got {}", mag);
    }

    #[test]
    fn biquad_filter_process() {
        let params = BandParams::peak(1000.0, 6.0, 1.0);
        let coeffs = BiquadCoeffs::compute(&params, 44100.0);
        let mut filter = BiquadFilter::new(coeffs);

        let mut samples: Vec<f32> = (0..1000)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin()
            })
            .collect();
        let input_energy: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        filter.process(&mut samples);
        let output_energy: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        assert!(output_energy > input_energy, "Peak filter should boost signal at center freq");
    }

    #[test]
    fn biquad_filter_reset() {
        let params = BandParams::peak(1000.0, 6.0, 1.0);
        let coeffs = BiquadCoeffs::compute(&params, 44100.0);
        let mut filter = BiquadFilter::new(coeffs);
        filter.process_sample(1.0);
        filter.reset();
        assert_eq!(filter.z1, 0.0);
        assert_eq!(filter.z2, 0.0);
    }

    #[test]
    fn parametric_eq_add_remove() {
        let mut eq = ParametricEq::new(44100.0);
        eq.add_band(BandParams::peak(100.0, 3.0, 1.0));
        eq.add_band(BandParams::peak(1000.0, -3.0, 1.0));
        eq.add_band(BandParams::high_shelf(8000.0, 2.0, 0.707));
        assert_eq!(eq.band_count(), 3);
        eq.remove_band(1);
        assert_eq!(eq.band_count(), 2);
    }

    #[test]
    fn parametric_eq_process() {
        let mut eq = ParametricEq::new(44100.0);
        eq.add_band(BandParams::peak(1000.0, 6.0, 1.0));

        let mut samples: Vec<f32> = (0..500)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin()
            })
            .collect();
        eq.process(&mut samples);
        // Should have non-zero output
        let energy: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn parametric_eq_frequency_response() {
        let mut eq = ParametricEq::new(44100.0);
        eq.add_band(BandParams::peak(1000.0, 12.0, 1.0));
        let response = eq.frequency_response_db(1000.0);
        // Should be around +12 dB at the center frequency
        assert!(response > 5.0, "Expected boost at center, got {} dB", response);
    }

    #[test]
    fn low_shelf_boosts_low() {
        let params = BandParams::low_shelf(200.0, 6.0, 0.707);
        let coeffs = BiquadCoeffs::compute(&params, 44100.0);
        let mag_low = coeffs.magnitude_response(50.0, 44100.0);
        let mag_high = coeffs.magnitude_response(5000.0, 44100.0);
        assert!(
            mag_low > mag_high,
            "Low shelf should boost low freq ({}) over high freq ({})",
            mag_low,
            mag_high
        );
    }

    #[test]
    fn high_shelf_boosts_high() {
        let params = BandParams::high_shelf(5000.0, 6.0, 0.707);
        let coeffs = BiquadCoeffs::compute(&params, 44100.0);
        let mag_high = coeffs.magnitude_response(15000.0, 44100.0);
        let mag_low = coeffs.magnitude_response(100.0, 44100.0);
        assert!(
            mag_high > mag_low,
            "High shelf should boost high freq ({}) over low freq ({})",
            mag_high,
            mag_low
        );
    }

    #[test]
    fn frequency_response_curve() {
        let mut eq = ParametricEq::new(44100.0);
        eq.add_band(BandParams::peak(1000.0, 6.0, 2.0));
        let freqs: Vec<f64> = vec![100.0, 500.0, 1000.0, 5000.0, 10000.0];
        let curve = eq.frequency_response_curve(&freqs);
        assert_eq!(curve.len(), 5);
        // Center should have highest response
        let center_db = curve[2].1;
        assert!(center_db > curve[0].1);
    }

    #[test]
    fn magnitude_response_db_test() {
        let params = BandParams::peak(1000.0, 0.0, 1.0);
        let coeffs = BiquadCoeffs::compute(&params, 44100.0);
        let db = coeffs.magnitude_response_db(1000.0, 44100.0);
        assert!(db.abs() < 0.1, "Unity filter should be ~0 dB, got {}", db);
    }
}
