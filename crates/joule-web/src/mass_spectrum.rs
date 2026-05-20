//! Mass spectrum representation, peak detection, and centroiding.
//!
//! Models raw and processed mass spectra with m/z–intensity pairs,
//! provides local-maxima peak detection with configurable noise
//! thresholds, Gaussian centroiding for accurate mass assignment,
//! and total/base-peak ion current summaries.

use std::fmt;

// ── MzIntensity ─────────────────────────────────────────────────

/// A single m/z–intensity data point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MzIntensity {
    pub mz: f64,
    pub intensity: f64,
}

impl MzIntensity {
    pub fn new(mz: f64, intensity: f64) -> Self {
        Self { mz, intensity }
    }
}

impl fmt::Display for MzIntensity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m/z={:.4} I={:.1}", self.mz, self.intensity)
    }
}

// ── SpectrumKind ────────────────────────────────────────────────

/// Whether a spectrum is profile (continuous) or centroid (stick).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectrumKind {
    Profile,
    Centroid,
}

impl fmt::Display for SpectrumKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile => write!(f, "profile"),
            Self::Centroid => write!(f, "centroid"),
        }
    }
}

// ── MsLevel ─────────────────────────────────────────────────────

/// MS acquisition level (MS1, MS2, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsLevel(pub u8);

impl MsLevel {
    pub fn ms1() -> Self { Self(1) }
    pub fn ms2() -> Self { Self(2) }
    pub fn ms3() -> Self { Self(3) }
}

impl fmt::Display for MsLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MS{}", self.0)
    }
}

// ── MassSpectrum ────────────────────────────────────────────────

/// A complete mass spectrum with metadata and data points.
#[derive(Debug, Clone)]
pub struct MassSpectrum {
    pub scan_number: u32,
    pub ms_level: MsLevel,
    pub kind: SpectrumKind,
    pub retention_time_sec: f64,
    pub precursor_mz: Option<f64>,
    pub points: Vec<MzIntensity>,
}

impl MassSpectrum {
    pub fn new(scan_number: u32, ms_level: MsLevel) -> Self {
        Self {
            scan_number,
            ms_level,
            kind: SpectrumKind::Profile,
            retention_time_sec: 0.0,
            precursor_mz: None,
            points: Vec::new(),
        }
    }

    pub fn with_kind(mut self, kind: SpectrumKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_retention_time(mut self, rt: f64) -> Self {
        self.retention_time_sec = rt;
        self
    }

    pub fn with_precursor(mut self, mz: f64) -> Self {
        self.precursor_mz = Some(mz);
        self
    }

    pub fn with_points(mut self, points: Vec<MzIntensity>) -> Self {
        self.points = points;
        self
    }

    /// Total ion current (sum of all intensities).
    pub fn tic(&self) -> f64 {
        self.points.iter().map(|p| p.intensity).sum()
    }

    /// Base peak intensity (maximum intensity).
    pub fn base_peak_intensity(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.intensity)
            .fold(0.0_f64, f64::max)
    }

    /// Base peak m/z (m/z of the most intense point).
    pub fn base_peak_mz(&self) -> Option<f64> {
        self.points
            .iter()
            .max_by(|a, b| a.intensity.partial_cmp(&b.intensity).unwrap())
            .map(|p| p.mz)
    }

    /// m/z range as (min, max).
    pub fn mz_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let min = self.points.iter().map(|p| p.mz).fold(f64::INFINITY, f64::min);
        let max = self.points.iter().map(|p| p.mz).fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the spectrum has no data points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Filter points below a minimum intensity threshold.
    pub fn filter_by_intensity(&self, min_intensity: f64) -> Self {
        let filtered = self
            .points
            .iter()
            .filter(|p| p.intensity >= min_intensity)
            .cloned()
            .collect();
        Self {
            points: filtered,
            ..self.clone()
        }
    }

    /// Filter points within a given m/z window.
    pub fn filter_by_mz_range(&self, low: f64, high: f64) -> Self {
        let filtered = self
            .points
            .iter()
            .filter(|p| p.mz >= low && p.mz <= high)
            .cloned()
            .collect();
        Self {
            points: filtered,
            ..self.clone()
        }
    }

    /// Normalize intensities to [0, max_val].
    pub fn normalize(&self, max_val: f64) -> Self {
        let bpi = self.base_peak_intensity();
        if bpi <= 0.0 {
            return self.clone();
        }
        let pts = self
            .points
            .iter()
            .map(|p| MzIntensity::new(p.mz, p.intensity / bpi * max_val))
            .collect();
        Self {
            points: pts,
            ..self.clone()
        }
    }
}

impl fmt::Display for MassSpectrum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Scan#{} {} {} RT={:.1}s pts={} TIC={:.0}",
            self.scan_number,
            self.ms_level,
            self.kind,
            self.retention_time_sec,
            self.points.len(),
            self.tic()
        )
    }
}

// ── Peak detection ──────────────────────────────────────────────

/// Configuration for local-maxima peak detection.
#[derive(Debug, Clone)]
pub struct PeakDetectorConfig {
    pub noise_threshold: f64,
    pub snr_minimum: f64,
    pub min_peak_distance_mz: f64,
}

impl PeakDetectorConfig {
    pub fn new() -> Self {
        Self {
            noise_threshold: 100.0,
            snr_minimum: 3.0,
            min_peak_distance_mz: 0.01,
        }
    }

    pub fn with_noise_threshold(mut self, t: f64) -> Self {
        self.noise_threshold = t;
        self
    }

    pub fn with_snr_minimum(mut self, snr: f64) -> Self {
        self.snr_minimum = snr;
        self
    }

    pub fn with_min_peak_distance(mut self, d: f64) -> Self {
        self.min_peak_distance_mz = d;
        self
    }
}

impl Default for PeakDetectorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A detected peak with centroid information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedPeak {
    pub mz: f64,
    pub intensity: f64,
    pub snr: f64,
    pub fwhm: f64,
}

impl DetectedPeak {
    pub fn new(mz: f64, intensity: f64, snr: f64, fwhm: f64) -> Self {
        Self { mz, intensity, snr, fwhm }
    }
}

impl fmt::Display for DetectedPeak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Peak m/z={:.4} I={:.1} SNR={:.1} FWHM={:.4}",
            self.mz, self.intensity, self.snr, self.fwhm
        )
    }
}

/// Detect local-maxima peaks in a profile spectrum.
///
/// A point is a local maximum if it is greater than both neighbours
/// and exceeds `noise_threshold * snr_minimum`.
pub fn detect_peaks(spectrum: &MassSpectrum, config: &PeakDetectorConfig) -> Vec<DetectedPeak> {
    let pts = &spectrum.points;
    if pts.len() < 3 {
        return Vec::new();
    }
    let intensity_cutoff = config.noise_threshold * config.snr_minimum;
    let mut peaks = Vec::new();

    for i in 1..pts.len() - 1 {
        let prev = pts[i - 1].intensity;
        let curr = pts[i].intensity;
        let next = pts[i + 1].intensity;

        if curr > prev && curr > next && curr >= intensity_cutoff {
            let snr = curr / config.noise_threshold;
            let half_max = curr / 2.0;

            // Estimate FWHM by linear interpolation to neighbours.
            let left_frac = if (curr - prev).abs() > 1e-15 {
                (curr - half_max) / (curr - prev)
            } else {
                0.5
            };
            let right_frac = if (curr - next).abs() > 1e-15 {
                (curr - half_max) / (curr - next)
            } else {
                0.5
            };
            let left_mz = pts[i].mz - left_frac * (pts[i].mz - pts[i - 1].mz);
            let right_mz = pts[i].mz + right_frac * (pts[i + 1].mz - pts[i].mz);
            let fwhm = right_mz - left_mz;

            // Enforce minimum distance from previous peak.
            if let Some(last) = peaks.last() {
                let last_peak: &DetectedPeak = last;
                if pts[i].mz - last_peak.mz < config.min_peak_distance_mz {
                    if curr > last_peak.intensity {
                        peaks.pop();
                    } else {
                        continue;
                    }
                }
            }

            peaks.push(DetectedPeak::new(pts[i].mz, curr, snr, fwhm));
        }
    }
    peaks
}

// ── Centroiding ─────────────────────────────────────────────────

/// Gaussian centroiding: fits a parabola to log-intensities of the
/// apex and its two neighbours to refine the m/z estimate.
pub fn gaussian_centroid(left: &MzIntensity, apex: &MzIntensity, right: &MzIntensity) -> f64 {
    let ln_l = (left.intensity.max(1.0)).ln();
    let ln_a = (apex.intensity.max(1.0)).ln();
    let ln_r = (right.intensity.max(1.0)).ln();

    let denom = 2.0 * (ln_l - 2.0 * ln_a + ln_r);
    if denom.abs() < 1e-15 {
        return apex.mz;
    }

    let delta = (ln_l - ln_r) / denom;
    let spacing = right.mz - left.mz;
    apex.mz + delta * spacing / 2.0
}

/// Centroid a full profile spectrum, producing centroid peaks.
pub fn centroid_spectrum(spectrum: &MassSpectrum, config: &PeakDetectorConfig) -> MassSpectrum {
    let pts = &spectrum.points;
    let detected = detect_peaks(spectrum, config);
    let mut centroid_pts = Vec::with_capacity(detected.len());

    for peak in &detected {
        // Find the index of the apex closest to this detected peak m/z.
        if let Some(idx) = pts.iter().position(|p| (p.mz - peak.mz).abs() < 1e-10) {
            if idx > 0 && idx < pts.len() - 1 {
                let refined_mz = gaussian_centroid(&pts[idx - 1], &pts[idx], &pts[idx + 1]);
                centroid_pts.push(MzIntensity::new(refined_mz, peak.intensity));
            } else {
                centroid_pts.push(MzIntensity::new(peak.mz, peak.intensity));
            }
        }
    }

    MassSpectrum {
        kind: SpectrumKind::Centroid,
        points: centroid_pts,
        ..spectrum.clone()
    }
}

// ── Noise estimation ────────────────────────────────────────────

/// Estimate noise level as the median intensity.
pub fn estimate_noise_median(spectrum: &MassSpectrum) -> f64 {
    if spectrum.points.is_empty() {
        return 0.0;
    }
    let mut vals: Vec<f64> = spectrum.points.iter().map(|p| p.intensity).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = vals.len() / 2;
    if vals.len() % 2 == 0 {
        (vals[mid - 1] + vals[mid]) / 2.0
    } else {
        vals[mid]
    }
}

/// Compute signal-to-noise ratio for a given intensity.
pub fn compute_snr(intensity: f64, noise: f64) -> f64 {
    if noise <= 0.0 { return f64::INFINITY; }
    intensity / noise
}

// ── m/z tolerance ───────────────────────────────────────────────

/// Check whether two m/z values are within a given ppm tolerance.
pub fn within_ppm(mz1: f64, mz2: f64, ppm: f64) -> bool {
    let delta = (mz1 - mz2).abs();
    let threshold = mz1.max(mz2) * ppm * 1e-6;
    delta <= threshold
}

/// Convert ppm tolerance to Dalton at a given m/z.
pub fn ppm_to_da(mz: f64, ppm: f64) -> f64 {
    mz * ppm * 1e-6
}

/// Convert Dalton tolerance to ppm at a given m/z.
pub fn da_to_ppm(mz: f64, da: f64) -> f64 {
    if mz <= 0.0 { return 0.0; }
    da / mz * 1e6
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> MassSpectrum {
        let pts = vec![
            MzIntensity::new(500.0, 50.0),
            MzIntensity::new(500.1, 200.0),
            MzIntensity::new(500.2, 1000.0),
            MzIntensity::new(500.3, 800.0),
            MzIntensity::new(500.4, 100.0),
            MzIntensity::new(500.5, 50.0),
            MzIntensity::new(500.6, 300.0),
            MzIntensity::new(500.7, 2000.0),
            MzIntensity::new(500.8, 400.0),
            MzIntensity::new(500.9, 60.0),
        ];
        MassSpectrum::new(1, MsLevel::ms1())
            .with_kind(SpectrumKind::Profile)
            .with_retention_time(120.5)
            .with_points(pts)
    }

    #[test]
    fn test_tic() {
        let s = sample_profile();
        let tic = s.tic();
        assert!((tic - 4960.0).abs() < 1e-6);
    }

    #[test]
    fn test_base_peak() {
        let s = sample_profile();
        assert!((s.base_peak_intensity() - 2000.0).abs() < 1e-6);
        assert!((s.base_peak_mz().unwrap() - 500.7).abs() < 1e-6);
    }

    #[test]
    fn test_mz_range() {
        let s = sample_profile();
        let (lo, hi) = s.mz_range().unwrap();
        assert!((lo - 500.0).abs() < 1e-6);
        assert!((hi - 500.9).abs() < 1e-6);
    }

    #[test]
    fn test_filter_by_intensity() {
        let s = sample_profile().filter_by_intensity(500.0);
        assert!(s.points.iter().all(|p| p.intensity >= 500.0));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn test_filter_by_mz() {
        let s = sample_profile().filter_by_mz_range(500.2, 500.7);
        assert!(s.points.iter().all(|p| p.mz >= 500.2 && p.mz <= 500.7));
    }

    #[test]
    fn test_normalize() {
        let s = sample_profile().normalize(100.0);
        assert!((s.base_peak_intensity() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_spectrum() {
        let s = MassSpectrum::new(0, MsLevel::ms1());
        assert!(s.is_empty());
        assert_eq!(s.tic(), 0.0);
        assert_eq!(s.base_peak_mz(), None);
        assert_eq!(s.mz_range(), None);
    }

    #[test]
    fn test_detect_peaks_basic() {
        let s = sample_profile();
        let cfg = PeakDetectorConfig::new().with_noise_threshold(40.0);
        let peaks = detect_peaks(&s, &cfg);
        assert!(!peaks.is_empty());
        assert!(peaks.iter().any(|p| (p.mz - 500.7).abs() < 0.15));
    }

    #[test]
    fn test_detect_peaks_high_threshold() {
        let s = sample_profile();
        let cfg = PeakDetectorConfig::new().with_noise_threshold(800.0);
        let peaks = detect_peaks(&s, &cfg);
        // Only the strongest peak should survive.
        assert!(peaks.len() <= 1);
    }

    #[test]
    fn test_gaussian_centroid() {
        let left = MzIntensity::new(100.0, 500.0);
        let apex = MzIntensity::new(100.1, 1000.0);
        let right = MzIntensity::new(100.2, 500.0);
        let c = gaussian_centroid(&left, &apex, &right);
        // Symmetric peak ⟹ centroid ≈ apex.
        assert!((c - 100.1).abs() < 0.01);
    }

    #[test]
    fn test_centroid_spectrum() {
        let s = sample_profile();
        let cfg = PeakDetectorConfig::new().with_noise_threshold(40.0);
        let cs = centroid_spectrum(&s, &cfg);
        assert_eq!(cs.kind, SpectrumKind::Centroid);
        assert!(!cs.is_empty());
    }

    #[test]
    fn test_estimate_noise() {
        let s = sample_profile();
        let n = estimate_noise_median(&s);
        assert!(n > 0.0 && n < 2000.0);
    }

    #[test]
    fn test_compute_snr() {
        assert!((compute_snr(1000.0, 100.0) - 10.0).abs() < 1e-6);
        assert_eq!(compute_snr(500.0, 0.0), f64::INFINITY);
    }

    #[test]
    fn test_within_ppm() {
        assert!(within_ppm(500.0, 500.0025, 10.0));
        assert!(!within_ppm(500.0, 500.1, 10.0));
    }

    #[test]
    fn test_ppm_da_conversions() {
        let da = ppm_to_da(1000.0, 10.0);
        assert!((da - 0.01).abs() < 1e-9);
        let ppm = da_to_ppm(1000.0, 0.01);
        assert!((ppm - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_spectrum() {
        let s = sample_profile();
        let d = format!("{}", s);
        assert!(d.contains("Scan#1"));
        assert!(d.contains("MS1"));
    }

    #[test]
    fn test_display_mz_intensity() {
        let p = MzIntensity::new(500.1234, 1000.0);
        let d = format!("{}", p);
        assert!(d.contains("500.1234"));
    }

    #[test]
    fn test_ms_level_display() {
        assert_eq!(format!("{}", MsLevel::ms2()), "MS2");
    }

    #[test]
    fn test_precursor() {
        let s = MassSpectrum::new(5, MsLevel::ms2()).with_precursor(750.38);
        assert_eq!(s.precursor_mz, Some(750.38));
    }

    #[test]
    fn test_detected_peak_display() {
        let p = DetectedPeak::new(500.25, 1000.0, 10.0, 0.05);
        let d = format!("{}", p);
        assert!(d.contains("500.25"));
        assert!(d.contains("SNR"));
    }
}
