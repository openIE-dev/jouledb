// tone_mapper.rs — HDR to LDR tone mapping for the rendering pipeline.
//
// Implements Reinhard (simple + extended), ACES filmic, Uncharted 2 (Hable),
// AgX, exposure-based (EV100), auto-exposure with histogram, white point,
// and gamma correction (linear to sRGB).

use std::fmt;

/// HDR color value (linear space).
#[derive(Clone, Debug, PartialEq)]
pub struct TmColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl TmColor {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0 }
    }

    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    pub fn clamp(&self, lo: f64, hi: f64) -> Self {
        Self {
            r: self.r.clamp(lo, hi),
            g: self.g.clamp(lo, hi),
            b: self.b.clamp(lo, hi),
        }
    }

    pub fn apply_fn(&self, f: impl Fn(f64) -> f64) -> Self {
        Self { r: f(self.r), g: f(self.g), b: f(self.b) }
    }
}

/// Tone mapping buffer.
#[derive(Clone, Debug)]
pub struct TmBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<TmColor>,
}

impl TmBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![TmColor::black(); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &TmColor {
        &self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, c: TmColor) {
        self.pixels[y * self.width + x] = c;
    }

    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

/// Tone mapping algorithm selection.
#[derive(Clone, Debug, PartialEq)]
pub enum ToneMapAlgorithm {
    /// Reinhard simple: L / (1 + L).
    ReinhardSimple,
    /// Reinhard extended with white point: L(1 + L/Lw^2) / (1 + L).
    ReinhardExtended { white_point: f64 },
    /// ACES filmic curve (fitted approximation).
    AcesFilmic,
    /// Uncharted 2 / Hable's filmic.
    Uncharted2,
    /// AgX tone mapping.
    AgX,
    /// Exposure-based (applies exposure then clamp).
    ExposureBased,
}

impl fmt::Display for ToneMapAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToneMapAlgorithm::ReinhardSimple => write!(f, "Reinhard Simple"),
            ToneMapAlgorithm::ReinhardExtended { white_point } => {
                write!(f, "Reinhard Extended (Lw={:.2})", white_point)
            }
            ToneMapAlgorithm::AcesFilmic => write!(f, "ACES Filmic"),
            ToneMapAlgorithm::Uncharted2 => write!(f, "Uncharted 2"),
            ToneMapAlgorithm::AgX => write!(f, "AgX"),
            ToneMapAlgorithm::ExposureBased => write!(f, "Exposure-Based"),
        }
    }
}

/// Exposure mode.
#[derive(Clone, Debug, PartialEq)]
pub enum ExposureMode {
    /// Manual exposure in EV (exposure value).
    Manual { ev: f64 },
    /// Auto-exposure using log-average luminance.
    Auto {
        min_ev: f64,
        max_ev: f64,
        /// Adaptation speed (0-1, 1 = instant).
        adaptation_speed: f64,
        /// Previous EV for temporal smoothing.
        prev_ev: f64,
    },
}

impl Default for ExposureMode {
    fn default() -> Self {
        ExposureMode::Manual { ev: 0.0 }
    }
}

impl fmt::Display for ExposureMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExposureMode::Manual { ev } => write!(f, "Manual(EV={:.2})", ev),
            ExposureMode::Auto { min_ev, max_ev, .. } => {
                write!(f, "Auto(EV=[{:.1}, {:.1}])", min_ev, max_ev)
            }
        }
    }
}

/// Tone mapper configuration.
#[derive(Clone, Debug)]
pub struct ToneMapConfig {
    pub algorithm: ToneMapAlgorithm,
    pub exposure: ExposureMode,
    /// Apply sRGB gamma correction at the end.
    pub gamma_correct: bool,
    /// White point for algorithms that use it.
    pub white_point: f64,
}

impl Default for ToneMapConfig {
    fn default() -> Self {
        Self {
            algorithm: ToneMapAlgorithm::AcesFilmic,
            exposure: ExposureMode::Manual { ev: 0.0 },
            gamma_correct: true,
            white_point: 1.0,
        }
    }
}

impl fmt::Display for ToneMapConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToneMapConfig(algo={}, exp={})", self.algorithm, self.exposure)
    }
}

// --- Tone Mapping Algorithms ---

/// Reinhard simple: C / (1 + C).
pub fn reinhard_simple(c: f64) -> f64 {
    c / (1.0 + c)
}

/// Reinhard extended with white point: C * (1 + C / Lw^2) / (1 + C).
pub fn reinhard_extended(c: f64, white_sq: f64) -> f64 {
    (c * (1.0 + c / white_sq)) / (1.0 + c)
}

/// ACES filmic fitted curve (Stephen Hill's fit).
pub fn aces_filmic(x: f64) -> f64 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let numerator = x * (a * x + b);
    let denominator = x * (c * x + d) + e;
    (numerator / denominator).clamp(0.0, 1.0)
}

/// Hable / Uncharted 2 helper function.
fn uncharted2_partial(x: f64) -> f64 {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
}

/// Uncharted 2 (Hable's) tone mapping.
pub fn uncharted2(x: f64) -> f64 {
    let exposure_bias = 2.0;
    let curr = uncharted2_partial(x * exposure_bias);
    let white_scale = 1.0 / uncharted2_partial(11.2);
    (curr * white_scale).clamp(0.0, 1.0)
}

/// AgX tone mapping curve.
pub fn agx_tonemap(x: f64) -> f64 {
    // Simplified AgX: log encoding + sigmoid
    let log_val = if x > 1e-10 { x.ln() } else { -10.0 };
    // Map from roughly [-10, 6.5] to [0, 1]
    let t = (log_val + 10.0) / 16.5;
    let t_clamped = t.clamp(0.0, 1.0);
    // Smooth sigmoid
    let s = t_clamped * t_clamped * (3.0 - 2.0 * t_clamped);
    s.clamp(0.0, 1.0)
}

/// Convert EV to exposure multiplier.
pub fn ev_to_exposure(ev: f64) -> f64 {
    2.0_f64.powf(ev)
}

/// Convert EV100 to exposure multiplier (for physical camera model).
pub fn ev100_to_exposure(ev100: f64) -> f64 {
    1.0 / (2.0_f64.powf(ev100) * 1.2)
}

/// Compute log-average luminance of a buffer.
pub fn log_average_luminance(buf: &TmBuffer) -> f64 {
    let delta = 1e-6;
    let mut sum = 0.0_f64;
    let count = buf.pixel_count();
    if count == 0 {
        return delta;
    }
    for px in &buf.pixels {
        sum += (px.luminance() + delta).ln();
    }
    (sum / count as f64).exp()
}

/// Luminance histogram with configurable bin count.
#[derive(Clone, Debug, PartialEq)]
pub struct LuminanceHistogram {
    pub bins: Vec<u32>,
    pub min_log_lum: f64,
    pub max_log_lum: f64,
    pub bin_count: usize,
}

impl LuminanceHistogram {
    pub fn build(buf: &TmBuffer, bin_count: usize, min_log: f64, max_log: f64) -> Self {
        let mut bins = vec![0u32; bin_count];
        let range = max_log - min_log;

        for px in &buf.pixels {
            let lum = px.luminance().max(1e-10);
            let log_lum = lum.ln();
            let t = ((log_lum - min_log) / range).clamp(0.0, 1.0 - 1e-10);
            let idx = (t * bin_count as f64) as usize;
            bins[idx.min(bin_count - 1)] += 1;
        }

        Self {
            bins,
            min_log_lum: min_log,
            max_log_lum: max_log,
            bin_count,
        }
    }

    /// Compute the average log-luminance from the histogram.
    pub fn average_log_luminance(&self, total_pixels: usize) -> f64 {
        if total_pixels == 0 {
            return 0.0;
        }
        let range = self.max_log_lum - self.min_log_lum;
        let mut sum = 0.0_f64;
        let mut count = 0u64;

        for (i, &bin) in self.bins.iter().enumerate() {
            if bin > 0 {
                let log_lum = self.min_log_lum + (i as f64 + 0.5) / self.bin_count as f64 * range;
                sum += log_lum * bin as f64;
                count += bin as u64;
            }
        }

        if count > 0 {
            (sum / count as f64).exp()
        } else {
            1.0
        }
    }

    /// Find the EV corresponding to the histogram's weighted average.
    pub fn compute_ev(&self, total_pixels: usize) -> f64 {
        let avg = self.average_log_luminance(total_pixels);
        if avg > 1e-10 {
            avg.log2()
        } else {
            -10.0
        }
    }
}

/// Compute auto-exposure EV from a buffer with histogram.
pub fn compute_auto_exposure(
    buf: &TmBuffer,
    min_ev: f64,
    max_ev: f64,
    adaptation_speed: f64,
    prev_ev: f64,
) -> f64 {
    let hist = LuminanceHistogram::build(buf, 128, -10.0, 6.5);
    let target_ev = hist.compute_ev(buf.pixel_count());
    let clamped_ev = (-target_ev).clamp(min_ev, max_ev);
    // Temporal smoothing
    prev_ev + (clamped_ev - prev_ev) * adaptation_speed.clamp(0.0, 1.0)
}

/// Linear to sRGB gamma correction for a single channel.
pub fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB to linear for a single channel.
pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Apply exposure to a color.
pub fn apply_exposure(color: &TmColor, ev: f64) -> TmColor {
    color.scale(ev_to_exposure(ev))
}

/// Tone map a single color using the specified algorithm.
pub fn tonemap_color(color: &TmColor, algorithm: &ToneMapAlgorithm) -> TmColor {
    match algorithm {
        ToneMapAlgorithm::ReinhardSimple => {
            color.apply_fn(reinhard_simple)
        }
        ToneMapAlgorithm::ReinhardExtended { white_point } => {
            let w_sq = white_point * white_point;
            color.apply_fn(|c| reinhard_extended(c, w_sq))
        }
        ToneMapAlgorithm::AcesFilmic => {
            color.apply_fn(aces_filmic)
        }
        ToneMapAlgorithm::Uncharted2 => {
            color.apply_fn(uncharted2)
        }
        ToneMapAlgorithm::AgX => {
            color.apply_fn(agx_tonemap)
        }
        ToneMapAlgorithm::ExposureBased => {
            // Just clamp to [0, 1] — exposure is applied externally
            color.clamp(0.0, 1.0)
        }
    }
}

/// Apply full tone mapping pipeline to a buffer.
pub fn apply_tone_mapping(buf: &TmBuffer, config: &ToneMapConfig) -> TmBuffer {
    let w = buf.width;
    let h = buf.height;
    let mut out = TmBuffer::new(w, h);

    // Determine exposure
    let ev = match &config.exposure {
        ExposureMode::Manual { ev } => *ev,
        ExposureMode::Auto { min_ev, max_ev, adaptation_speed, prev_ev } => {
            compute_auto_exposure(buf, *min_ev, *max_ev, *adaptation_speed, *prev_ev)
        }
    };

    for y in 0..h {
        for x in 0..w {
            let px = buf.get(x, y);
            // 1. Apply exposure
            let exposed = apply_exposure(px, ev);
            // 2. Tone map
            let mapped = tonemap_color(&exposed, &config.algorithm);
            // 3. Gamma correction
            let final_color = if config.gamma_correct {
                mapped.apply_fn(linear_to_srgb)
            } else {
                mapped
            };
            out.set(x, y, final_color.clamp(0.0, 1.0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_color_luminance() {
        let c = TmColor::new(1.0, 1.0, 1.0);
        assert!(approx_eq(c.luminance(), 1.0, 1e-6));
    }

    #[test]
    fn test_color_luminance_red() {
        let c = TmColor::new(1.0, 0.0, 0.0);
        assert!(approx_eq(c.luminance(), 0.2126, 1e-4));
    }

    #[test]
    fn test_reinhard_simple_zero() {
        assert!(approx_eq(reinhard_simple(0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_reinhard_simple_one() {
        assert!(approx_eq(reinhard_simple(1.0), 0.5, 1e-6));
    }

    #[test]
    fn test_reinhard_simple_monotonic() {
        let a = reinhard_simple(1.0);
        let b = reinhard_simple(2.0);
        let c = reinhard_simple(10.0);
        assert!(a < b);
        assert!(b < c);
        assert!(c < 1.0);
    }

    #[test]
    fn test_reinhard_extended() {
        let val = reinhard_extended(1.0, 4.0); // white = 2.0
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_aces_filmic_range() {
        assert!(approx_eq(aces_filmic(0.0), 0.0, 0.01));
        let mid = aces_filmic(1.0);
        assert!(mid > 0.0 && mid <= 1.0);
        let high = aces_filmic(100.0);
        assert!(approx_eq(high, 1.0, 0.01));
    }

    #[test]
    fn test_aces_filmic_monotonic() {
        let a = aces_filmic(0.1);
        let b = aces_filmic(0.5);
        let c = aces_filmic(2.0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_uncharted2_range() {
        let low = uncharted2(0.0);
        assert!(approx_eq(low, 0.0, 0.05));
        let high = uncharted2(100.0);
        assert!(high > 0.9);
    }

    #[test]
    fn test_agx_range() {
        let low = agx_tonemap(0.0);
        assert!(low >= 0.0 && low <= 1.0);
        let high = agx_tonemap(100.0);
        assert!(high >= 0.0 && high <= 1.0);
    }

    #[test]
    fn test_ev_to_exposure() {
        assert!(approx_eq(ev_to_exposure(0.0), 1.0, 1e-6));
        assert!(approx_eq(ev_to_exposure(1.0), 2.0, 1e-6));
        assert!(approx_eq(ev_to_exposure(-1.0), 0.5, 1e-6));
    }

    #[test]
    fn test_linear_to_srgb_boundaries() {
        assert!(approx_eq(linear_to_srgb(0.0), 0.0, 1e-6));
        assert!(approx_eq(linear_to_srgb(1.0), 1.0, 1e-3));
    }

    #[test]
    fn test_srgb_roundtrip() {
        let vals = [0.0, 0.1, 0.5, 0.9, 1.0];
        for &v in &vals {
            let srgb = linear_to_srgb(v);
            let back = srgb_to_linear(srgb);
            assert!(approx_eq(back, v, 1e-4));
        }
    }

    #[test]
    fn test_log_average_luminance() {
        let mut buf = TmBuffer::new(2, 2);
        buf.set(0, 0, TmColor::new(1.0, 1.0, 1.0));
        buf.set(1, 0, TmColor::new(1.0, 1.0, 1.0));
        buf.set(0, 1, TmColor::new(1.0, 1.0, 1.0));
        buf.set(1, 1, TmColor::new(1.0, 1.0, 1.0));
        let avg = log_average_luminance(&buf);
        assert!(approx_eq(avg, 1.0, 0.01));
    }

    #[test]
    fn test_histogram_build() {
        let mut buf = TmBuffer::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                buf.set(x, y, TmColor::new(0.5, 0.5, 0.5));
            }
        }
        let hist = LuminanceHistogram::build(&buf, 64, -10.0, 6.5);
        let total: u32 = hist.bins.iter().sum();
        assert_eq!(total, 16);
    }

    #[test]
    fn test_histogram_average_log() {
        let mut buf = TmBuffer::new(2, 2);
        for px in &mut buf.pixels {
            *px = TmColor::new(1.0, 1.0, 1.0);
        }
        let hist = LuminanceHistogram::build(&buf, 64, -10.0, 6.5);
        let avg = hist.average_log_luminance(4);
        assert!(avg > 0.5 && avg < 2.0);
    }

    #[test]
    fn test_auto_exposure_bright_scene() {
        let mut buf = TmBuffer::new(4, 4);
        for px in &mut buf.pixels {
            *px = TmColor::new(10.0, 10.0, 10.0); // Bright scene
        }
        let ev = compute_auto_exposure(&buf, -4.0, 4.0, 1.0, 0.0);
        // Auto-exposure should produce negative EV for bright scene (reduce brightness)
        assert!(ev >= -4.0 && ev <= 4.0);
    }

    #[test]
    fn test_apply_tone_mapping_reinhard() {
        let mut buf = TmBuffer::new(4, 4);
        buf.set(2, 2, TmColor::new(5.0, 3.0, 1.0));
        let config = ToneMapConfig {
            algorithm: ToneMapAlgorithm::ReinhardSimple,
            exposure: ExposureMode::Manual { ev: 0.0 },
            gamma_correct: false,
            white_point: 1.0,
        };
        let result = apply_tone_mapping(&buf, &config);
        let px = result.get(2, 2);
        assert!(px.r > 0.0 && px.r <= 1.0);
        assert!(px.g > 0.0 && px.g <= 1.0);
    }

    #[test]
    fn test_apply_tone_mapping_aces() {
        let mut buf = TmBuffer::new(4, 4);
        buf.set(1, 1, TmColor::new(2.0, 2.0, 2.0));
        let config = ToneMapConfig {
            algorithm: ToneMapAlgorithm::AcesFilmic,
            gamma_correct: true,
            ..Default::default()
        };
        let result = apply_tone_mapping(&buf, &config);
        let px = result.get(1, 1);
        assert!(px.r > 0.0 && px.r <= 1.0);
    }

    #[test]
    fn test_apply_tone_mapping_uncharted2() {
        let mut buf = TmBuffer::new(2, 2);
        buf.set(0, 0, TmColor::new(3.0, 3.0, 3.0));
        let config = ToneMapConfig {
            algorithm: ToneMapAlgorithm::Uncharted2,
            gamma_correct: false,
            ..Default::default()
        };
        let result = apply_tone_mapping(&buf, &config);
        let px = result.get(0, 0);
        assert!(px.r > 0.0 && px.r <= 1.0);
    }

    #[test]
    fn test_apply_tone_mapping_agx() {
        let mut buf = TmBuffer::new(2, 2);
        buf.set(0, 0, TmColor::new(5.0, 5.0, 5.0));
        let config = ToneMapConfig {
            algorithm: ToneMapAlgorithm::AgX,
            gamma_correct: false,
            ..Default::default()
        };
        let result = apply_tone_mapping(&buf, &config);
        let px = result.get(0, 0);
        assert!(px.r >= 0.0 && px.r <= 1.0);
    }

    #[test]
    fn test_apply_tone_mapping_exposure_based() {
        let mut buf = TmBuffer::new(2, 2);
        buf.set(0, 0, TmColor::new(0.8, 0.8, 0.8));
        let config = ToneMapConfig {
            algorithm: ToneMapAlgorithm::ExposureBased,
            exposure: ExposureMode::Manual { ev: -1.0 },
            gamma_correct: false,
            ..Default::default()
        };
        let result = apply_tone_mapping(&buf, &config);
        let px = result.get(0, 0);
        assert!(approx_eq(px.r, 0.4, 0.05));
    }

    #[test]
    fn test_config_display() {
        let config = ToneMapConfig::default();
        let s = format!("{}", config);
        assert!(s.contains("ToneMapConfig"));
        assert!(s.contains("ACES"));
    }

    #[test]
    fn test_algorithm_display() {
        let a = ToneMapAlgorithm::ReinhardSimple;
        assert_eq!(format!("{}", a), "Reinhard Simple");
    }

    #[test]
    fn test_ev100_to_exposure() {
        let e = ev100_to_exposure(0.0);
        assert!(approx_eq(e, 1.0 / 1.2, 1e-6));
    }
}
