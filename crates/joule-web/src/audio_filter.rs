//! Audio filters — biquad filters, parametric EQ, filter cascade, DC removal.
//!
//! Implements second-order IIR biquad filters (lowpass, highpass, bandpass,
//! notch, allpass, peaking EQ, low/high shelf) with coefficient calculation
//! from the Audio EQ Cookbook. Supports filter cascading and frequency response.

use std::f64::consts::PI;

// ── Filter Type ─────────────────────────────────────────────────

/// Biquad filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Allpass,
    PeakingEq,
    LowShelf,
    HighShelf,
}

// ── Biquad Coefficients ─────────────────────────────────────────

/// Biquad filter coefficients: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2).
/// Stored normalized so a0 = 1.0.
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoefficients {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl BiquadCoefficients {
    /// Calculate lowpass filter coefficients.
    pub fn lowpass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();

        let b1 = 1.0 - cos_w0;
        let b0 = b1 / 2.0;
        let b2 = b0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Calculate highpass filter coefficients.
    pub fn highpass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = b0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Calculate bandpass filter coefficients (constant skirt gain).
    pub fn bandpass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Calculate notch (band-reject) filter coefficients.
    pub fn notch(freq: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();

        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Calculate allpass filter coefficients.
    pub fn allpass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();

        let b0 = 1.0 - alpha;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 + alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Calculate peaking EQ filter coefficients.
    /// `gain_db` is the gain at center frequency in dB.
    pub fn peaking_eq(freq: f64, q: f64, gain_db: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a = 10.0f64.powf(gain_db / 40.0);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;

        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Calculate low shelf filter coefficients.
    pub fn low_shelf(freq: f64, gain_db: f64, slope: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let a = 10.0f64.powf(gain_db / 40.0);
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Calculate high shelf filter coefficients.
    pub fn high_shelf(freq: f64, gain_db: f64, slope: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let a = 10.0f64.powf(gain_db / 40.0);
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Normalize coefficients by dividing by a0.
    fn normalize(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Calculate the magnitude response at a given frequency.
    pub fn magnitude_at(&self, freq: f64, sample_rate: f64) -> f64 {
        let w = 2.0 * PI * freq / sample_rate;
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

    /// Calculate magnitude response in dB.
    pub fn magnitude_db_at(&self, freq: f64, sample_rate: f64) -> f64 {
        let mag = self.magnitude_at(freq, sample_rate);
        if mag <= 0.0 {
            return -120.0;
        }
        20.0 * mag.log10()
    }
}

// ── Biquad Filter ───────────────────────────────────────────────

/// Second-order IIR biquad filter with direct form II transposed.
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    coeffs: BiquadCoefficients,
    /// Filter state (delay line).
    z1: f64,
    z2: f64,
    filter_type: FilterType,
    frequency: f64,
    q: f64,
    sample_rate: f64,
}

impl BiquadFilter {
    /// Create a new biquad filter.
    pub fn new(filter_type: FilterType, freq: f64, q: f64, sample_rate: f64) -> Self {
        let coeffs = Self::compute_coeffs(filter_type, freq, q, 0.0, sample_rate);
        Self {
            coeffs,
            z1: 0.0,
            z2: 0.0,
            filter_type,
            frequency: freq,
            q,
            sample_rate,
        }
    }

    /// Create a peaking EQ filter.
    pub fn peaking(freq: f64, q: f64, gain_db: f64, sample_rate: f64) -> Self {
        let coeffs = BiquadCoefficients::peaking_eq(freq, q, gain_db, sample_rate);
        Self {
            coeffs,
            z1: 0.0,
            z2: 0.0,
            filter_type: FilterType::PeakingEq,
            frequency: freq,
            q,
            sample_rate,
        }
    }

    fn compute_coeffs(
        filter_type: FilterType,
        freq: f64,
        q: f64,
        gain_db: f64,
        sample_rate: f64,
    ) -> BiquadCoefficients {
        match filter_type {
            FilterType::Lowpass => BiquadCoefficients::lowpass(freq, q, sample_rate),
            FilterType::Highpass => BiquadCoefficients::highpass(freq, q, sample_rate),
            FilterType::Bandpass => BiquadCoefficients::bandpass(freq, q, sample_rate),
            FilterType::Notch => BiquadCoefficients::notch(freq, q, sample_rate),
            FilterType::Allpass => BiquadCoefficients::allpass(freq, q, sample_rate),
            FilterType::PeakingEq => BiquadCoefficients::peaking_eq(freq, q, gain_db, sample_rate),
            FilterType::LowShelf => BiquadCoefficients::low_shelf(freq, gain_db, 1.0, sample_rate),
            FilterType::HighShelf => {
                BiquadCoefficients::high_shelf(freq, gain_db, 1.0, sample_rate)
            }
        }
    }

    pub fn coefficients(&self) -> &BiquadCoefficients {
        &self.coeffs
    }

    pub fn filter_type(&self) -> FilterType {
        self.filter_type
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    pub fn q(&self) -> f64 {
        self.q
    }

    /// Update the filter frequency.
    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
        self.coeffs = Self::compute_coeffs(self.filter_type, freq, self.q, 0.0, self.sample_rate);
    }

    /// Update the Q factor.
    pub fn set_q(&mut self, q: f64) {
        self.q = q;
        self.coeffs =
            Self::compute_coeffs(self.filter_type, self.frequency, q, 0.0, self.sample_rate);
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Process a single sample (Direct Form II Transposed).
    pub fn tick(&mut self, input: f32) -> f32 {
        let x = input as f64;
        let y = self.coeffs.b0 * x + self.z1;
        self.z1 = self.coeffs.b1 * x - self.coeffs.a1 * y + self.z2;
        self.z2 = self.coeffs.b2 * x - self.coeffs.a2 * y;
        y as f32
    }

    /// Process a buffer of samples in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.tick(*s);
        }
    }

    /// Process a buffer, writing to a separate output.
    pub fn process_into(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = self.tick(input[i]);
        }
    }
}

// ── Filter Cascade ──────────────────────────────────────────────

/// A cascade (series) of biquad filters for steeper rolloff or complex EQ.
#[derive(Debug, Clone)]
pub struct FilterCascade {
    filters: Vec<BiquadFilter>,
}

impl FilterCascade {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Create a Butterworth lowpass of order N (uses N/2 biquad stages).
    pub fn butterworth_lowpass(order: usize, freq: f64, sample_rate: f64) -> Self {
        let stages = (order + 1) / 2;
        let mut cascade = Self::new();
        for k in 0..stages {
            let q = 1.0 / (2.0 * (PI * (2 * k + 1) as f64 / (2 * order) as f64).cos());
            cascade.add(BiquadFilter::new(FilterType::Lowpass, freq, q, sample_rate));
        }
        cascade
    }

    /// Create a Butterworth highpass of order N.
    pub fn butterworth_highpass(order: usize, freq: f64, sample_rate: f64) -> Self {
        let stages = (order + 1) / 2;
        let mut cascade = Self::new();
        for k in 0..stages {
            let q = 1.0 / (2.0 * (PI * (2 * k + 1) as f64 / (2 * order) as f64).cos());
            cascade.add(BiquadFilter::new(
                FilterType::Highpass,
                freq,
                q,
                sample_rate,
            ));
        }
        cascade
    }

    /// Add a filter stage.
    pub fn add(&mut self, filter: BiquadFilter) {
        self.filters.push(filter);
    }

    pub fn stage_count(&self) -> usize {
        self.filters.len()
    }

    /// Process a single sample through all stages.
    pub fn tick(&mut self, input: f32) -> f32 {
        let mut sample = input;
        for filter in &mut self.filters {
            sample = filter.tick(sample);
        }
        sample
    }

    /// Process a buffer in place through all stages.
    pub fn process(&mut self, samples: &mut [f32]) {
        for filter in &mut self.filters {
            filter.process(samples);
        }
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }

    /// Calculate the combined magnitude response at a frequency.
    pub fn magnitude_at(&self, freq: f64, sample_rate: f64) -> f64 {
        let mut mag = 1.0;
        for filter in &self.filters {
            mag *= filter.coefficients().magnitude_at(freq, sample_rate);
        }
        mag
    }
}

impl Default for FilterCascade {
    fn default() -> Self {
        Self::new()
    }
}

// ── Parametric EQ ───────────────────────────────────────────────

/// A parametric EQ band.
#[derive(Debug, Clone)]
pub struct EqBand {
    pub filter_type: FilterType,
    pub frequency: f64,
    pub q: f64,
    pub gain_db: f64,
    pub enabled: bool,
}

/// Multi-band parametric equalizer.
#[derive(Debug, Clone)]
pub struct ParametricEq {
    bands: Vec<EqBand>,
    filters: Vec<BiquadFilter>,
    sample_rate: f64,
}

impl ParametricEq {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            bands: Vec::new(),
            filters: Vec::new(),
            sample_rate,
        }
    }

    /// Add a band and return its index.
    pub fn add_band(&mut self, band: EqBand) -> usize {
        let filter = match band.filter_type {
            FilterType::PeakingEq => BiquadFilter::peaking(
                band.frequency,
                band.q,
                band.gain_db,
                self.sample_rate,
            ),
            FilterType::LowShelf => {
                let mut f =
                    BiquadFilter::new(FilterType::LowShelf, band.frequency, band.q, self.sample_rate);
                f.coeffs =
                    BiquadCoefficients::low_shelf(band.frequency, band.gain_db, 1.0, self.sample_rate);
                f
            }
            FilterType::HighShelf => {
                let mut f =
                    BiquadFilter::new(FilterType::HighShelf, band.frequency, band.q, self.sample_rate);
                f.coeffs = BiquadCoefficients::high_shelf(
                    band.frequency,
                    band.gain_db,
                    1.0,
                    self.sample_rate,
                );
                f
            }
            _ => BiquadFilter::new(band.filter_type, band.frequency, band.q, self.sample_rate),
        };
        self.bands.push(band);
        self.filters.push(filter);
        self.bands.len() - 1
    }

    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Update a band's parameters.
    pub fn update_band(&mut self, index: usize, band: EqBand) {
        if index >= self.bands.len() {
            return;
        }
        let filter = match band.filter_type {
            FilterType::PeakingEq => BiquadFilter::peaking(
                band.frequency,
                band.q,
                band.gain_db,
                self.sample_rate,
            ),
            _ => BiquadFilter::new(band.filter_type, band.frequency, band.q, self.sample_rate),
        };
        self.bands[index] = band;
        self.filters[index] = filter;
    }

    /// Enable or disable a band.
    pub fn set_band_enabled(&mut self, index: usize, enabled: bool) {
        if let Some(band) = self.bands.get_mut(index) {
            band.enabled = enabled;
        }
    }

    /// Process a single sample through all enabled bands.
    pub fn tick(&mut self, input: f32) -> f32 {
        let mut sample = input;
        for (band, filter) in self.bands.iter().zip(self.filters.iter_mut()) {
            if band.enabled {
                sample = filter.tick(sample);
            }
        }
        sample
    }

    /// Process a buffer in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.tick(*s);
        }
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }
}

// ── DC Offset Removal ──────────────────────────────────────────

/// Removes DC offset from audio using a first-order highpass filter.
#[derive(Debug, Clone)]
pub struct DcBlocker {
    x_prev: f64,
    y_prev: f64,
    /// Coefficient controlling the cutoff (close to 1.0 for low cutoff).
    coeff: f64,
}

impl DcBlocker {
    /// Create a DC blocker. `r` controls the cutoff (typically 0.995 to 0.999).
    pub fn new(r: f64) -> Self {
        Self {
            x_prev: 0.0,
            y_prev: 0.0,
            coeff: r.clamp(0.9, 0.9999),
        }
    }

    /// Default DC blocker with r = 0.995.
    pub fn default_blocker() -> Self {
        Self::new(0.995)
    }

    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }

    /// Process a single sample.
    pub fn tick(&mut self, input: f32) -> f32 {
        let x = input as f64;
        let y = x - self.x_prev + self.coeff * self.y_prev;
        self.x_prev = x;
        self.y_prev = y;
        y as f32
    }

    /// Process a buffer in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.tick(*s);
        }
    }
}

// ── Frequency Response ──────────────────────────────────────────

/// A point on a frequency response curve.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyResponsePoint {
    pub frequency_hz: f64,
    pub magnitude_db: f64,
}

/// Generate frequency response points for a set of coefficients.
pub fn frequency_response(
    coeffs: &BiquadCoefficients,
    sample_rate: f64,
    num_points: usize,
) -> Vec<FrequencyResponsePoint> {
    let nyquist = sample_rate / 2.0;
    let log_min = 20.0f64.log10();
    let log_max = nyquist.log10();

    (0..num_points)
        .map(|i| {
            let t = i as f64 / (num_points - 1).max(1) as f64;
            let freq = 10.0f64.powf(log_min + t * (log_max - log_min));
            let mag_db = coeffs.magnitude_db_at(freq, sample_rate);
            FrequencyResponsePoint {
                frequency_hz: freq,
                magnitude_db: mag_db,
            }
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;

    #[test]
    fn lowpass_attenuates_high_frequencies() {
        let coeffs = BiquadCoefficients::lowpass(1000.0, 0.707, SR);
        let mag_low = coeffs.magnitude_at(100.0, SR);
        let mag_high = coeffs.magnitude_at(10000.0, SR);
        assert!(
            mag_low > mag_high,
            "lowpass should attenuate highs: mag_low={mag_low}, mag_high={mag_high}"
        );
    }

    #[test]
    fn highpass_attenuates_low_frequencies() {
        let coeffs = BiquadCoefficients::highpass(1000.0, 0.707, SR);
        let mag_low = coeffs.magnitude_at(100.0, SR);
        let mag_high = coeffs.magnitude_at(10000.0, SR);
        assert!(
            mag_high > mag_low,
            "highpass should attenuate lows: mag_low={mag_low}, mag_high={mag_high}"
        );
    }

    #[test]
    fn bandpass_peaks_at_center() {
        let coeffs = BiquadCoefficients::bandpass(1000.0, 5.0, SR);
        let mag_center = coeffs.magnitude_at(1000.0, SR);
        let mag_low = coeffs.magnitude_at(100.0, SR);
        let mag_high = coeffs.magnitude_at(10000.0, SR);
        assert!(mag_center > mag_low);
        assert!(mag_center > mag_high);
    }

    #[test]
    fn notch_rejects_center() {
        let coeffs = BiquadCoefficients::notch(1000.0, 10.0, SR);
        let mag_center = coeffs.magnitude_at(1000.0, SR);
        let mag_side = coeffs.magnitude_at(500.0, SR);
        assert!(
            mag_center < mag_side,
            "notch should reject center: center={mag_center}, side={mag_side}"
        );
    }

    #[test]
    fn allpass_unity_magnitude() {
        let coeffs = BiquadCoefficients::allpass(1000.0, 0.707, SR);
        for freq in [100.0, 500.0, 1000.0, 5000.0, 10000.0] {
            let mag = coeffs.magnitude_at(freq, SR);
            assert!(
                (mag - 1.0).abs() < 0.05,
                "allpass should have unity magnitude at {freq} Hz, got {mag}"
            );
        }
    }

    #[test]
    fn biquad_filter_process() {
        let mut filter = BiquadFilter::new(FilterType::Lowpass, 1000.0, 0.707, SR);
        let mut samples = vec![1.0f32; 100];
        filter.process(&mut samples);
        // After filtering a step, output should approach 1.0
        assert!(samples[99].abs() > 0.5);
    }

    #[test]
    fn biquad_filter_reset() {
        let mut filter = BiquadFilter::new(FilterType::Lowpass, 1000.0, 0.707, SR);
        filter.tick(1.0);
        filter.tick(1.0);
        filter.reset();
        // After reset, processing zero should give ~zero
        let val = filter.tick(0.0);
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn biquad_set_frequency() {
        let mut filter = BiquadFilter::new(FilterType::Lowpass, 1000.0, 0.707, SR);
        assert!((filter.frequency() - 1000.0).abs() < 1e-6);
        filter.set_frequency(2000.0);
        assert!((filter.frequency() - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn filter_cascade_basic() {
        let mut cascade = FilterCascade::new();
        cascade.add(BiquadFilter::new(FilterType::Lowpass, 1000.0, 0.707, SR));
        cascade.add(BiquadFilter::new(FilterType::Lowpass, 1000.0, 0.707, SR));
        assert_eq!(cascade.stage_count(), 2);

        let mut samples = vec![1.0f32; 100];
        cascade.process(&mut samples);
        // Two lowpass filters cascaded should still settle
        assert!(samples[99].abs() > 0.1);
    }

    #[test]
    fn butterworth_lowpass_steepness() {
        let cascade_2nd = FilterCascade::butterworth_lowpass(2, 1000.0, SR);
        let cascade_4th = FilterCascade::butterworth_lowpass(4, 1000.0, SR);

        let mag_2nd = cascade_2nd.magnitude_at(4000.0, SR);
        let mag_4th = cascade_4th.magnitude_at(4000.0, SR);

        // 4th order should attenuate more than 2nd
        assert!(
            mag_4th < mag_2nd,
            "4th order should be steeper: 2nd={mag_2nd}, 4th={mag_4th}"
        );
    }

    #[test]
    fn parametric_eq_basic() {
        let mut eq = ParametricEq::new(SR);
        eq.add_band(EqBand {
            filter_type: FilterType::PeakingEq,
            frequency: 1000.0,
            q: 1.0,
            gain_db: 6.0,
            enabled: true,
        });
        assert_eq!(eq.band_count(), 1);

        let mut samples = vec![0.5f32; 100];
        eq.process(&mut samples);
        // Should produce some output
        assert!(samples.iter().any(|s| s.abs() > 0.01));
    }

    #[test]
    fn parametric_eq_disable_band() {
        let mut eq = ParametricEq::new(SR);
        eq.add_band(EqBand {
            filter_type: FilterType::Lowpass,
            frequency: 100.0,
            q: 0.707,
            gain_db: 0.0,
            enabled: true,
        });

        // Process with enabled
        let mut samples1 = vec![1.0f32; 100];
        eq.process(&mut samples1);
        eq.reset();

        // Process with disabled
        eq.set_band_enabled(0, false);
        let mut samples2 = vec![1.0f32; 100];
        eq.process(&mut samples2);

        // Disabled should pass through unchanged
        assert!((samples2[50] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dc_blocker_removes_offset() {
        let mut blocker = DcBlocker::default_blocker();
        // Input: DC + sine
        let mut samples: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f64 / SR;
                (0.5 + (440.0 * 2.0 * PI * t).sin() * 0.5) as f32
            })
            .collect();
        blocker.process(&mut samples);

        // DC component should be largely removed at the end
        let last_chunk_mean: f32 =
            samples[40000..44000].iter().sum::<f32>() / 4000.0;
        assert!(
            last_chunk_mean.abs() < 0.1,
            "DC should be removed, got mean = {last_chunk_mean}"
        );
    }

    #[test]
    fn dc_blocker_reset() {
        let mut blocker = DcBlocker::new(0.995);
        blocker.tick(1.0);
        blocker.tick(1.0);
        blocker.reset();
        let val = blocker.tick(0.0);
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn frequency_response_points() {
        let coeffs = BiquadCoefficients::lowpass(1000.0, 0.707, SR);
        let points = frequency_response(&coeffs, SR, 100);
        assert_eq!(points.len(), 100);
        // First point should be near 20 Hz
        assert!(points[0].frequency_hz > 19.0 && points[0].frequency_hz < 21.0);
        // Low frequencies should be near 0 dB for lowpass
        assert!(points[0].magnitude_db.abs() < 1.0);
    }

    #[test]
    fn peaking_eq_boost() {
        let coeffs = BiquadCoefficients::peaking_eq(1000.0, 2.0, 12.0, SR);
        let mag_at_center = coeffs.magnitude_db_at(1000.0, SR);
        let mag_at_100 = coeffs.magnitude_db_at(100.0, SR);
        assert!(
            mag_at_center > mag_at_100,
            "peaking EQ should boost center: center={mag_at_center} dB, 100Hz={mag_at_100} dB"
        );
    }

    #[test]
    fn process_into_buffer() {
        let mut filter = BiquadFilter::new(FilterType::Lowpass, 5000.0, 0.707, SR);
        let input = vec![1.0f32; 50];
        let mut output = vec![0.0f32; 50];
        filter.process_into(&input, &mut output);
        assert!(output[49] > 0.5);
    }

    #[test]
    fn cascade_magnitude_response() {
        let cascade = FilterCascade::butterworth_lowpass(4, 1000.0, SR);
        let mag_pass = cascade.magnitude_at(100.0, SR);
        let mag_stop = cascade.magnitude_at(10000.0, SR);
        assert!(mag_pass > 0.9, "passband should be near unity");
        assert!(mag_stop < 0.01, "stopband should be attenuated");
    }

    #[test]
    fn filter_type_getters() {
        let filter = BiquadFilter::new(FilterType::Highpass, 2000.0, 1.0, SR);
        assert_eq!(filter.filter_type(), FilterType::Highpass);
        assert!((filter.frequency() - 2000.0).abs() < 1e-6);
        assert!((filter.q() - 1.0).abs() < 1e-6);
    }
}
