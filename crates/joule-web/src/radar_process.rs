//! Radar processing — range-Doppler map computation via 2D FFT, CFAR
//! (Constant False Alarm Rate) detection, target tracking with alpha-beta
//! filter, and clutter suppression using MTI (Moving Target Indication).
//!
//! Pure-Rust radar signal processing pipeline for pulsed-Doppler systems,
//! suitable for embedded tracking and surveillance workloads.

use std::f64::consts::PI;
use std::fmt;

// ── Complex Number ──────────────────────────────────────────────

/// Minimal complex number for FFT operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn from_polar(mag: f64, phase: f64) -> Self {
        Self { re: mag * phase.cos(), im: mag * phase.sin() }
    }

    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conjugate(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn add(&self, other: &Complex) -> Complex {
        Complex { re: self.re + other.re, im: self.im + other.im }
    }

    pub fn sub(&self, other: &Complex) -> Complex {
        Complex { re: self.re - other.re, im: self.im - other.im }
    }

    pub fn scale(&self, s: f64) -> Complex {
        Complex { re: self.re * s, im: self.im * s }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.4}+{:.4}j", self.re, self.im)
        } else {
            write!(f, "{:.4}{:.4}j", self.re, self.im)
        }
    }
}

// ── FFT ─────────────────────────────────────────────────────────

/// In-place radix-2 Cooley-Tukey FFT. `data` length must be a power of 2.
pub fn fft(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_sign = if inverse { 1.0 } else { -1.0 };
        let angle = angle_sign * 2.0 * PI / len as f64;
        let wn = Complex::from_polar(1.0, angle);
        let mut start = 0;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = data[start + k];
                let t = w.mul(&data[start + k + half]);
                data[start + k] = u.add(&t);
                data[start + k + half] = u.sub(&t);
                w = w.mul(&wn);
            }
            start += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv_n = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x = x.scale(inv_n);
        }
    }
}

/// Next power of 2 >= n.
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

// ── Range-Doppler Map ───────────────────────────────────────────

/// Configuration for range-Doppler map generation.
#[derive(Debug, Clone)]
pub struct RangeDopplerConfig {
    pub num_range_bins: usize,
    pub num_pulses: usize,
    pub sample_rate_hz: f64,
    pub prf_hz: f64,
    pub speed_of_light: f64,
    pub center_freq_hz: f64,
    pub apply_window: bool,
}

impl RangeDopplerConfig {
    pub fn new(num_range_bins: usize, num_pulses: usize) -> Self {
        Self {
            num_range_bins,
            num_pulses,
            sample_rate_hz: 10e6,
            prf_hz: 1000.0,
            speed_of_light: 3e8,
            center_freq_hz: 10e9,
            apply_window: true,
        }
    }

    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate_hz = rate;
        self
    }

    pub fn with_prf(mut self, prf: f64) -> Self {
        self.prf_hz = prf;
        self
    }

    pub fn with_center_freq(mut self, freq: f64) -> Self {
        self.center_freq_hz = freq;
        self
    }

    pub fn with_window(mut self, enable: bool) -> Self {
        self.apply_window = enable;
        self
    }

    /// Range resolution in meters.
    pub fn range_resolution(&self) -> f64 {
        self.speed_of_light / (2.0 * self.sample_rate_hz)
    }

    /// Velocity resolution in m/s.
    pub fn velocity_resolution(&self) -> f64 {
        let wavelength = self.speed_of_light / self.center_freq_hz;
        wavelength * self.prf_hz / (2.0 * self.num_pulses as f64)
    }

    /// Maximum unambiguous range in meters.
    pub fn max_unambiguous_range(&self) -> f64 {
        self.speed_of_light / (2.0 * self.prf_hz)
    }

    /// Range for a given bin index.
    pub fn bin_to_range(&self, bin: usize) -> f64 {
        bin as f64 * self.range_resolution()
    }

    /// Velocity for a given Doppler bin index.
    pub fn bin_to_velocity(&self, bin: usize) -> f64 {
        let wavelength = self.speed_of_light / self.center_freq_hz;
        let max_vel = wavelength * self.prf_hz / 4.0;
        let norm = bin as f64 / self.num_pulses as f64;
        if norm <= 0.5 {
            norm * 2.0 * max_vel
        } else {
            (norm - 1.0) * 2.0 * max_vel
        }
    }
}

impl fmt::Display for RangeDopplerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RDConfig(range_bins={}, pulses={}, range_res={:.2}m, vel_res={:.2}m/s)",
            self.num_range_bins,
            self.num_pulses,
            self.range_resolution(),
            self.velocity_resolution(),
        )
    }
}

/// Range-Doppler map: 2D magnitude grid.
#[derive(Debug, Clone)]
pub struct RangeDopplerMap {
    pub data: Vec<Vec<f64>>,
    pub num_range_bins: usize,
    pub num_doppler_bins: usize,
}

impl RangeDopplerMap {
    /// Build a range-Doppler map from raw pulse data.
    /// `pulses` is a num_pulses x num_range_bins matrix of IQ samples.
    pub fn from_pulses(config: &RangeDopplerConfig, pulses: &[Vec<Complex>]) -> Self {
        let nr = next_pow2(config.num_range_bins);
        let nd = next_pow2(config.num_pulses);
        let num_pulses = pulses.len().min(nd);

        // Range FFT per pulse
        let mut range_fft: Vec<Vec<Complex>> = Vec::with_capacity(num_pulses);
        for pulse in pulses.iter().take(num_pulses) {
            let mut row = vec![Complex::zero(); nr];
            for (i, &sample) in pulse.iter().enumerate().take(nr) {
                let window = if config.apply_window {
                    0.54 - 0.46 * (2.0 * PI * i as f64 / (nr - 1).max(1) as f64).cos()
                } else {
                    1.0
                };
                row[i] = sample.scale(window);
            }
            fft(&mut row, false);
            range_fft.push(row);
        }

        // Doppler FFT per range bin
        let mut map_data = vec![vec![0.0; nd]; nr];
        for rbin in 0..nr {
            let mut col = vec![Complex::zero(); nd];
            for (pidx, row) in range_fft.iter().enumerate() {
                if rbin < row.len() {
                    let window = if config.apply_window {
                        0.54 - 0.46 * (2.0 * PI * pidx as f64 / (nd - 1).max(1) as f64).cos()
                    } else {
                        1.0
                    };
                    col[pidx] = row[rbin].scale(window);
                }
            }
            fft(&mut col, false);
            for (dbin, c) in col.iter().enumerate() {
                map_data[rbin][dbin] = c.magnitude();
            }
        }

        Self {
            data: map_data,
            num_range_bins: nr,
            num_doppler_bins: nd,
        }
    }

    /// Peak value and its (range_bin, doppler_bin) location.
    pub fn peak(&self) -> (f64, usize, usize) {
        let mut max_val = 0.0;
        let mut max_r = 0;
        let mut max_d = 0;
        for (r, row) in self.data.iter().enumerate() {
            for (d, &val) in row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_r = r;
                    max_d = d;
                }
            }
        }
        (max_val, max_r, max_d)
    }
}

impl fmt::Display for RangeDopplerMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (peak_val, pr, pd) = self.peak();
        write!(
            f,
            "RDMap({}x{}, peak={:.1} at [{}, {}])",
            self.num_range_bins, self.num_doppler_bins, peak_val, pr, pd
        )
    }
}

// ── CFAR Detection ──────────────────────────────────────────────

/// Cell-Averaging CFAR detector (CA-CFAR).
#[derive(Debug, Clone)]
pub struct CfarDetector {
    pub guard_cells: usize,
    pub training_cells: usize,
    pub threshold_factor: f64,
}

impl CfarDetector {
    pub fn new() -> Self {
        Self {
            guard_cells: 2,
            training_cells: 8,
            threshold_factor: 4.0,
        }
    }

    pub fn with_guard_cells(mut self, n: usize) -> Self {
        self.guard_cells = n;
        self
    }

    pub fn with_training_cells(mut self, n: usize) -> Self {
        self.training_cells = n;
        self
    }

    pub fn with_threshold_factor(mut self, f: f64) -> Self {
        self.threshold_factor = f;
        self
    }

    /// 1D CA-CFAR detection on a slice. Returns indices of detections.
    pub fn detect_1d(&self, signal: &[f64]) -> Vec<usize> {
        let n = signal.len();
        let window = self.guard_cells + self.training_cells;
        let mut detections = Vec::new();
        if n < 2 * window + 1 {
            return detections;
        }
        for i in window..(n - window) {
            let mut sum = 0.0;
            let mut count = 0;
            // Leading training cells
            for j in (i - window)..(i - self.guard_cells) {
                sum += signal[j];
                count += 1;
            }
            // Lagging training cells
            for j in (i + self.guard_cells + 1)..=(i + window) {
                if j < n {
                    sum += signal[j];
                    count += 1;
                }
            }
            if count > 0 {
                let threshold = (sum / count as f64) * self.threshold_factor;
                if signal[i] > threshold {
                    detections.push(i);
                }
            }
        }
        detections
    }

    /// 2D CA-CFAR detection on a range-Doppler map. Returns (range_bin, doppler_bin) pairs.
    pub fn detect_2d(&self, map: &RangeDopplerMap) -> Vec<(usize, usize)> {
        let mut detections = Vec::new();
        let nr = map.num_range_bins;
        let nd = map.num_doppler_bins;
        let window = self.guard_cells + self.training_cells;

        for r in window..(nr.saturating_sub(window)) {
            for d in window..(nd.saturating_sub(window)) {
                let mut sum = 0.0;
                let mut count = 0;
                for ri in (r - window)..=(r + window) {
                    for di in (d - window)..=(d + window) {
                        if ri >= nr || di >= nd {
                            continue;
                        }
                        let dr = if ri > r { ri - r } else { r - ri };
                        let dd = if di > d { di - d } else { d - di };
                        if dr <= self.guard_cells && dd <= self.guard_cells {
                            continue; // Skip guard + CUT
                        }
                        sum += map.data[ri][di];
                        count += 1;
                    }
                }
                if count > 0 {
                    let threshold = (sum / count as f64) * self.threshold_factor;
                    if map.data[r][d] > threshold {
                        detections.push((r, d));
                    }
                }
            }
        }
        detections
    }
}

impl fmt::Display for CfarDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CFAR(guard={}, train={}, alpha={:.1})",
            self.guard_cells, self.training_cells, self.threshold_factor
        )
    }
}

// ── Target Tracking ─────────────────────────────────────────────

/// Radar target detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarDetection {
    pub range_m: f64,
    pub velocity_mps: f64,
    pub azimuth_rad: f64,
    pub snr_db: f64,
    pub timestamp_s: f64,
}

impl fmt::Display for RadarDetection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Det(r={:.1}m, v={:.1}m/s, az={:.1}deg, snr={:.1}dB)",
            self.range_m, self.velocity_mps, self.azimuth_rad.to_degrees(), self.snr_db
        )
    }
}

/// Alpha-beta target tracker.
#[derive(Debug, Clone)]
pub struct AlphaBetaTracker {
    pub alpha: f64,
    pub beta: f64,
    pub range: f64,
    pub range_rate: f64,
    pub last_time: f64,
    pub track_id: usize,
    pub update_count: usize,
}

impl AlphaBetaTracker {
    pub fn new(track_id: usize, alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta,
            range: 0.0,
            range_rate: 0.0,
            last_time: 0.0,
            track_id,
            update_count: 0,
        }
    }

    pub fn with_initial_state(mut self, range: f64, rate: f64, time: f64) -> Self {
        self.range = range;
        self.range_rate = rate;
        self.last_time = time;
        self
    }

    /// Predict state to time t.
    pub fn predict(&self, t: f64) -> (f64, f64) {
        let dt = t - self.last_time;
        let predicted_range = self.range + self.range_rate * dt;
        (predicted_range, self.range_rate)
    }

    /// Update tracker with a new measurement.
    pub fn update(&mut self, measured_range: f64, t: f64) {
        let dt = t - self.last_time;
        if dt <= 0.0 && self.update_count > 0 {
            return;
        }
        // Predict
        let predicted = self.range + self.range_rate * dt;
        // Residual
        let residual = measured_range - predicted;
        // Update
        self.range = predicted + self.alpha * residual;
        if dt > 0.0 {
            self.range_rate += self.beta * residual / dt;
        }
        self.last_time = t;
        self.update_count += 1;
    }
}

impl fmt::Display for AlphaBetaTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Track(id={}, r={:.1}m, dr={:.1}m/s, updates={})",
            self.track_id, self.range, self.range_rate, self.update_count
        )
    }
}

// ── Clutter Filter (MTI) ───────────────────────────────────────

/// Moving Target Indication (MTI) clutter canceller.
#[derive(Debug, Clone)]
pub struct MtiFilter {
    pub order: usize,
    pub coefficients: Vec<f64>,
}

impl MtiFilter {
    /// Single-pulse canceller (order 1): y[n] = x[n] - x[n-1].
    pub fn single_canceller() -> Self {
        Self { order: 1, coefficients: vec![1.0, -1.0] }
    }

    /// Double-pulse canceller (order 2): y[n] = x[n] - 2x[n-1] + x[n-2].
    pub fn double_canceller() -> Self {
        Self { order: 2, coefficients: vec![1.0, -2.0, 1.0] }
    }

    /// Custom canceller with given coefficients.
    pub fn with_coefficients(coefficients: Vec<f64>) -> Self {
        let order = if coefficients.is_empty() { 0 } else { coefficients.len() - 1 };
        Self { order, coefficients }
    }

    /// Apply MTI filter to a sequence of pulse returns at the same range bin.
    pub fn apply(&self, pulses: &[f64]) -> Vec<f64> {
        if pulses.len() <= self.order {
            return Vec::new();
        }
        let mut output = Vec::with_capacity(pulses.len() - self.order);
        for i in self.order..pulses.len() {
            let mut val = 0.0;
            for (k, &coeff) in self.coefficients.iter().enumerate() {
                val += coeff * pulses[i - k];
            }
            output.push(val);
        }
        output
    }

    /// Compute the MTI filter frequency response (magnitude) at normalized freq.
    pub fn frequency_response(&self, num_points: usize) -> Vec<f64> {
        let mut response = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let freq = i as f64 / num_points as f64;
            let mut re = 0.0;
            let mut im = 0.0;
            for (k, &coeff) in self.coefficients.iter().enumerate() {
                let angle = -2.0 * PI * freq * k as f64;
                re += coeff * angle.cos();
                im += coeff * angle.sin();
            }
            response.push((re * re + im * im).sqrt());
        }
        response
    }
}

impl fmt::Display for MtiFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MTI(order={}, taps={})", self.order, self.coefficients.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_magnitude() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.magnitude() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_complex_multiply() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a.mul(&b);
        assert!((c.re - (-5.0)).abs() < 1e-12);
        assert!((c.im - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_complex_display() {
        let c = Complex::new(1.0, -2.0);
        let s = format!("{c}");
        assert!(s.contains("-2.0000j"));
    }

    #[test]
    fn test_fft_impulse() {
        let mut data = vec![Complex::zero(); 8];
        data[0] = Complex::new(1.0, 0.0);
        fft(&mut data, false);
        for c in &data {
            assert!((c.magnitude() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_inverse() {
        let original = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, -1.0),
        ];
        let mut data = original.clone();
        fft(&mut data, false);
        fft(&mut data, true);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1e-10);
            assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_range_doppler_config() {
        let config = RangeDopplerConfig::new(64, 16)
            .with_sample_rate(20e6)
            .with_prf(2000.0);
        assert!(config.range_resolution() > 0.0);
        assert!(config.velocity_resolution() > 0.0);
        assert!(config.max_unambiguous_range() > 0.0);
    }

    #[test]
    fn test_range_doppler_config_display() {
        let config = RangeDopplerConfig::new(64, 16);
        let s = format!("{config}");
        assert!(s.contains("RDConfig"));
    }

    #[test]
    fn test_range_doppler_map_simple() {
        let config = RangeDopplerConfig::new(8, 4).with_window(false);
        let mut pulses = Vec::new();
        for p in 0..4 {
            let mut row = Vec::new();
            for r in 0..8 {
                let val = if r == 3 { 10.0 } else { 0.1 };
                let phase = 2.0 * PI * 0.25 * p as f64;
                row.push(Complex::new(val * phase.cos(), val * phase.sin()));
            }
            pulses.push(row);
        }
        let map = RangeDopplerMap::from_pulses(&config, &pulses);
        let (peak, _, _) = map.peak();
        assert!(peak > 1.0);
    }

    #[test]
    fn test_cfar_1d_basic() {
        let mut signal = vec![1.0; 50];
        signal[25] = 20.0; // Target
        let cfar = CfarDetector::new()
            .with_guard_cells(1)
            .with_training_cells(4)
            .with_threshold_factor(3.0);
        let dets = cfar.detect_1d(&signal);
        assert!(dets.contains(&25));
    }

    #[test]
    fn test_cfar_no_detection() {
        let signal = vec![1.0; 50];
        let cfar = CfarDetector::new();
        let dets = cfar.detect_1d(&signal);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_cfar_display() {
        let cfar = CfarDetector::new();
        let s = format!("{cfar}");
        assert!(s.contains("CFAR"));
    }

    #[test]
    fn test_tracker_predict() {
        let tracker = AlphaBetaTracker::new(0, 0.5, 0.1)
            .with_initial_state(100.0, 10.0, 0.0);
        let (pred_r, pred_dr) = tracker.predict(1.0);
        assert!((pred_r - 110.0).abs() < 1e-10);
        assert!((pred_dr - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracker_update() {
        let mut tracker = AlphaBetaTracker::new(1, 0.8, 0.2)
            .with_initial_state(100.0, 10.0, 0.0);
        tracker.update(112.0, 1.0);
        assert!(tracker.update_count == 1);
        assert!((tracker.range - 100.0).abs() < 20.0);
    }

    #[test]
    fn test_tracker_display() {
        let tracker = AlphaBetaTracker::new(5, 0.5, 0.1)
            .with_initial_state(500.0, 20.0, 0.0);
        let s = format!("{tracker}");
        assert!(s.contains("Track"));
        assert!(s.contains("id=5"));
    }

    #[test]
    fn test_mti_single_canceller() {
        let mti = MtiFilter::single_canceller();
        let pulses = vec![5.0, 5.0, 5.0, 15.0, 5.0];
        let out = mti.apply(&pulses);
        assert_eq!(out.len(), 4);
        // Constant clutter should cancel
        assert!((out[0]).abs() < 1e-12);
        assert!((out[1]).abs() < 1e-12);
        // Moving target should survive
        assert!((out[2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_mti_double_canceller() {
        let mti = MtiFilter::double_canceller();
        let pulses = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Linear trend
        let out = mti.apply(&pulses);
        // Linear trend cancels with double canceller
        for v in &out {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_mti_frequency_response() {
        let mti = MtiFilter::single_canceller();
        let resp = mti.frequency_response(64);
        // DC rejection: response at f=0 should be ~0
        assert!(resp[0].abs() < 1e-10);
        // Max response near f=0.5
        let max_val = resp.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_val > 1.5);
    }

    #[test]
    fn test_mti_display() {
        let mti = MtiFilter::single_canceller();
        let s = format!("{mti}");
        assert!(s.contains("MTI"));
    }

    #[test]
    fn test_detection_display() {
        let det = RadarDetection {
            range_m: 500.0,
            velocity_mps: 15.0,
            azimuth_rad: 0.5,
            snr_db: 12.0,
            timestamp_s: 0.0,
        };
        let s = format!("{det}");
        assert!(s.contains("500.0m"));
    }
}
