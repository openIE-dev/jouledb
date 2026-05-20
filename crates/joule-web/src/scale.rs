//! D3-style scales: Linear, Log, Pow, Band, Ordinal, Time.
//! Ticks generation, invert, clamp.

use chrono::{DateTime, Utc, TimeDelta};

// ── Scale Trait ─────────────────────────────────────────────────

/// Common interface for continuous scales.
pub trait ContinuousScale {
    /// Map a domain value to a range value.
    fn scale(&self, value: f64) -> f64;

    /// Map a range value back to a domain value.
    fn invert(&self, value: f64) -> f64;

    /// Generate tick values in the domain.
    fn ticks(&self, count: usize) -> Vec<f64>;
}

// ── LinearScale ─────────────────────────────────────────────────

/// Linear scale: domain [d0, d1] -> range [r0, r1].
#[derive(Debug, Clone)]
pub struct LinearScale {
    pub domain: (f64, f64),
    pub range: (f64, f64),
    pub clamp: bool,
}

impl LinearScale {
    pub fn new(domain: (f64, f64), range: (f64, f64)) -> Self {
        Self { domain, range, clamp: false }
    }

    pub fn with_clamp(mut self, clamp: bool) -> Self {
        self.clamp = clamp;
        self
    }

    fn normalize(&self, value: f64) -> f64 {
        let d = self.domain.1 - self.domain.0;
        if d.abs() < f64::EPSILON {
            return 0.5;
        }
        (value - self.domain.0) / d
    }

    fn clamp_t(&self, t: f64) -> f64 {
        if self.clamp { t.clamp(0.0, 1.0) } else { t }
    }
}

impl ContinuousScale for LinearScale {
    fn scale(&self, value: f64) -> f64 {
        let t = self.clamp_t(self.normalize(value));
        self.range.0 + t * (self.range.1 - self.range.0)
    }

    fn invert(&self, value: f64) -> f64 {
        let r = self.range.1 - self.range.0;
        if r.abs() < f64::EPSILON {
            return self.domain.0;
        }
        let t = (value - self.range.0) / r;
        let t = if self.clamp { t.clamp(0.0, 1.0) } else { t };
        self.domain.0 + t * (self.domain.1 - self.domain.0)
    }

    fn ticks(&self, count: usize) -> Vec<f64> {
        nice_ticks(self.domain.0, self.domain.1, count)
    }
}

// ── LogScale ────────────────────────────────────────────────────

/// Logarithmic scale (base 10 by default).
#[derive(Debug, Clone)]
pub struct LogScale {
    pub domain: (f64, f64),
    pub range: (f64, f64),
    pub base: f64,
    pub clamp: bool,
}

impl LogScale {
    pub fn new(domain: (f64, f64), range: (f64, f64)) -> Self {
        Self { domain, range, base: 10.0, clamp: false }
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    pub fn with_clamp(mut self, clamp: bool) -> Self {
        self.clamp = clamp;
        self
    }

    fn log_val(&self, v: f64) -> f64 {
        v.max(f64::EPSILON).ln() / self.base.ln()
    }
}

impl ContinuousScale for LogScale {
    fn scale(&self, value: f64) -> f64 {
        let log_d0 = self.log_val(self.domain.0);
        let log_d1 = self.log_val(self.domain.1);
        let log_v = self.log_val(value);
        let d = log_d1 - log_d0;
        if d.abs() < f64::EPSILON {
            return self.range.0;
        }
        let mut t = (log_v - log_d0) / d;
        if self.clamp {
            t = t.clamp(0.0, 1.0);
        }
        self.range.0 + t * (self.range.1 - self.range.0)
    }

    fn invert(&self, value: f64) -> f64 {
        let r = self.range.1 - self.range.0;
        if r.abs() < f64::EPSILON {
            return self.domain.0;
        }
        let mut t = (value - self.range.0) / r;
        if self.clamp {
            t = t.clamp(0.0, 1.0);
        }
        let log_d0 = self.log_val(self.domain.0);
        let log_d1 = self.log_val(self.domain.1);
        let log_v = log_d0 + t * (log_d1 - log_d0);
        self.base.powf(log_v)
    }

    fn ticks(&self, count: usize) -> Vec<f64> {
        let log_min = self.log_val(self.domain.0).floor() as i64;
        let log_max = self.log_val(self.domain.1).ceil() as i64;
        let mut ticks = Vec::new();
        for exp in log_min..=log_max {
            let v = self.base.powi(exp as i32);
            if v >= self.domain.0 && v <= self.domain.1 {
                ticks.push(v);
            }
            if ticks.len() >= count {
                break;
            }
        }
        ticks
    }
}

// ── PowScale ────────────────────────────────────────────────────

/// Power scale with configurable exponent.
#[derive(Debug, Clone)]
pub struct PowScale {
    pub domain: (f64, f64),
    pub range: (f64, f64),
    pub exponent: f64,
    pub clamp: bool,
}

impl PowScale {
    pub fn new(domain: (f64, f64), range: (f64, f64), exponent: f64) -> Self {
        Self { domain, range, exponent, clamp: false }
    }

    pub fn with_clamp(mut self, clamp: bool) -> Self {
        self.clamp = clamp;
        self
    }

    fn pow_val(&self, v: f64) -> f64 {
        v.signum() * v.abs().powf(self.exponent)
    }

    fn inv_pow(&self, v: f64) -> f64 {
        v.signum() * v.abs().powf(1.0 / self.exponent)
    }
}

impl ContinuousScale for PowScale {
    fn scale(&self, value: f64) -> f64 {
        let pd0 = self.pow_val(self.domain.0);
        let pd1 = self.pow_val(self.domain.1);
        let pv = self.pow_val(value);
        let d = pd1 - pd0;
        if d.abs() < f64::EPSILON {
            return self.range.0;
        }
        let mut t = (pv - pd0) / d;
        if self.clamp {
            t = t.clamp(0.0, 1.0);
        }
        self.range.0 + t * (self.range.1 - self.range.0)
    }

    fn invert(&self, value: f64) -> f64 {
        let r = self.range.1 - self.range.0;
        if r.abs() < f64::EPSILON {
            return self.domain.0;
        }
        let mut t = (value - self.range.0) / r;
        if self.clamp {
            t = t.clamp(0.0, 1.0);
        }
        let pd0 = self.pow_val(self.domain.0);
        let pd1 = self.pow_val(self.domain.1);
        let pv = pd0 + t * (pd1 - pd0);
        self.inv_pow(pv)
    }

    fn ticks(&self, count: usize) -> Vec<f64> {
        nice_ticks(self.domain.0, self.domain.1, count)
    }
}

// ── BandScale ───────────────────────────────────────────────────

/// Band scale: discrete domain -> equal-width bands with padding.
#[derive(Debug, Clone)]
pub struct BandScale {
    pub domain: Vec<String>,
    pub range: (f64, f64),
    /// Padding between bands as fraction of band step (0.0..1.0).
    pub padding_inner: f64,
    /// Padding on outer edges as fraction of band step.
    pub padding_outer: f64,
}

impl BandScale {
    pub fn new(domain: Vec<String>, range: (f64, f64)) -> Self {
        Self {
            domain,
            range,
            padding_inner: 0.0,
            padding_outer: 0.0,
        }
    }

    pub fn with_padding(mut self, inner: f64, outer: f64) -> Self {
        self.padding_inner = inner.clamp(0.0, 1.0);
        self.padding_outer = outer.clamp(0.0, 1.0);
        self
    }

    /// Width of a single band.
    pub fn bandwidth(&self) -> f64 {
        let n = self.domain.len();
        if n == 0 {
            return 0.0;
        }
        let total = (self.range.1 - self.range.0).abs();
        let step = total / (n as f64 + self.padding_outer * 2.0 - self.padding_inner + self.padding_inner * n as f64);
        step * (1.0 - self.padding_inner)
    }

    /// Step size (band + inner padding).
    pub fn step(&self) -> f64 {
        let n = self.domain.len();
        if n == 0 {
            return 0.0;
        }
        let total = (self.range.1 - self.range.0).abs();
        total / (n as f64 + self.padding_outer * 2.0 - self.padding_inner + self.padding_inner * n as f64)
    }

    /// Get the start position for a domain value.
    pub fn scale(&self, value: &str) -> Option<f64> {
        let idx = self.domain.iter().position(|v| v == value)?;
        let step = self.step();
        let offset = self.padding_outer * step;
        Some(self.range.0 + offset + idx as f64 * step)
    }

    /// Get the center of the band for a domain value.
    pub fn scale_center(&self, value: &str) -> Option<f64> {
        self.scale(value).map(|s| s + self.bandwidth() / 2.0)
    }
}

// ── OrdinalScale ────────────────────────────────────────────────

/// Ordinal scale: discrete domain -> discrete range.
#[derive(Debug, Clone)]
pub struct OrdinalScale<T: Clone> {
    pub domain: Vec<String>,
    pub range: Vec<T>,
}

impl<T: Clone> OrdinalScale<T> {
    pub fn new(domain: Vec<String>, range: Vec<T>) -> Self {
        Self { domain, range }
    }

    pub fn scale(&self, value: &str) -> Option<T> {
        let idx = self.domain.iter().position(|v| v == value)?;
        if self.range.is_empty() {
            return None;
        }
        Some(self.range[idx % self.range.len()].clone())
    }
}

// ── TimeScale ───────────────────────────────────────────────────

/// Time scale: DateTime domain -> numeric range.
#[derive(Debug, Clone)]
pub struct TimeScale {
    pub domain: (DateTime<Utc>, DateTime<Utc>),
    pub range: (f64, f64),
    pub clamp: bool,
}

impl TimeScale {
    pub fn new(domain: (DateTime<Utc>, DateTime<Utc>), range: (f64, f64)) -> Self {
        Self { domain, range, clamp: false }
    }

    pub fn with_clamp(mut self, clamp: bool) -> Self {
        self.clamp = clamp;
        self
    }

    /// Map a DateTime to a range value.
    pub fn scale(&self, value: &DateTime<Utc>) -> f64 {
        let d0 = self.domain.0.timestamp_millis() as f64;
        let d1 = self.domain.1.timestamp_millis() as f64;
        let v = value.timestamp_millis() as f64;
        let d = d1 - d0;
        if d.abs() < f64::EPSILON {
            return self.range.0;
        }
        let mut t = (v - d0) / d;
        if self.clamp {
            t = t.clamp(0.0, 1.0);
        }
        self.range.0 + t * (self.range.1 - self.range.0)
    }

    /// Map a range value back to a DateTime.
    pub fn invert(&self, value: f64) -> DateTime<Utc> {
        let r = self.range.1 - self.range.0;
        let d0 = self.domain.0.timestamp_millis() as f64;
        let d1 = self.domain.1.timestamp_millis() as f64;
        if r.abs() < f64::EPSILON {
            return self.domain.0;
        }
        let mut t = (value - self.range.0) / r;
        if self.clamp {
            t = t.clamp(0.0, 1.0);
        }
        let ms = d0 + t * (d1 - d0);
        DateTime::from_timestamp_millis(ms as i64).unwrap_or(self.domain.0)
    }

    /// Generate time ticks (evenly spaced DateTimes).
    pub fn ticks(&self, count: usize) -> Vec<DateTime<Utc>> {
        if count == 0 {
            return Vec::new();
        }
        let d0 = self.domain.0.timestamp_millis();
        let d1 = self.domain.1.timestamp_millis();
        let step = (d1 - d0) / count as i64;
        if step == 0 {
            return vec![self.domain.0];
        }
        let mut ticks = Vec::new();
        let mut ms = d0;
        while ms <= d1 {
            if let Some(dt) = DateTime::from_timestamp_millis(ms) {
                ticks.push(dt);
            }
            ms += step;
        }
        ticks
    }
}

// ── Nice Ticks ──────────────────────────────────────────────────

/// Generate nice round tick values for a range.
pub fn nice_ticks(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 || start == stop {
        return if start == stop { vec![start] } else { Vec::new() };
    }
    let step = nice_step(start, stop, count);
    if step <= 0.0 || !step.is_finite() {
        return vec![start, stop];
    }
    let lo = (start / step).ceil();
    let hi = (stop / step).floor();
    let mut ticks = Vec::new();
    let mut i = lo;
    while i <= hi {
        let v = i * step;
        // Avoid floating-point noise
        ticks.push((v * 1e12).round() / 1e12);
        i += 1.0;
    }
    ticks
}

/// Compute a nice step size for tick generation.
fn nice_step(start: f64, stop: f64, count: usize) -> f64 {
    let range = (stop - start).abs();
    let rough_step = range / count as f64;
    let mag = 10.0_f64.powf(rough_step.log10().floor());
    let residual = rough_step / mag;
    let nice = if residual <= 1.5 {
        1.0
    } else if residual <= 3.0 {
        2.0
    } else if residual <= 7.0 {
        5.0
    } else {
        10.0
    };
    nice * mag
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_linear_scale() {
        let s = LinearScale::new((0.0, 100.0), (0.0, 500.0));
        assert!((s.scale(50.0) - 250.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_invert() {
        let s = LinearScale::new((0.0, 100.0), (0.0, 500.0));
        assert!((s.invert(250.0) - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_clamp() {
        let s = LinearScale::new((0.0, 100.0), (0.0, 500.0)).with_clamp(true);
        assert!((s.scale(150.0) - 500.0).abs() < 0.001);
        assert!((s.scale(-50.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_log_scale() {
        let s = LogScale::new((1.0, 1000.0), (0.0, 300.0));
        assert!((s.scale(10.0) - 100.0).abs() < 0.001);
        assert!((s.scale(100.0) - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_log_invert() {
        let s = LogScale::new((1.0, 1000.0), (0.0, 300.0));
        assert!((s.invert(100.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_pow_scale_sqrt() {
        let s = PowScale::new((0.0, 100.0), (0.0, 100.0), 0.5);
        // sqrt(25) / sqrt(100) = 5/10 = 0.5 -> 50
        assert!((s.scale(25.0) - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_pow_invert() {
        let s = PowScale::new((0.0, 100.0), (0.0, 100.0), 2.0);
        let v = s.scale(50.0);
        let back = s.invert(v);
        assert!((back - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_band_scale() {
        let s = BandScale::new(
            vec!["A".into(), "B".into(), "C".into()],
            (0.0, 300.0),
        );
        assert!((s.bandwidth() - 100.0).abs() < 0.01);
        assert!((s.scale("A").unwrap() - 0.0).abs() < 0.01);
        assert!((s.scale("B").unwrap() - 100.0).abs() < 0.01);
        assert!(s.scale("Z").is_none());
    }

    #[test]
    fn test_band_scale_with_padding() {
        let s = BandScale::new(
            vec!["X".into(), "Y".into()],
            (0.0, 200.0),
        ).with_padding(0.2, 0.1);
        assert!(s.bandwidth() > 0.0);
        assert!(s.step() > s.bandwidth());
    }

    #[test]
    fn test_ordinal_scale() {
        let s = OrdinalScale::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        );
        assert_eq!(s.scale("b"), Some("green".to_string()));
        assert_eq!(s.scale("z"), None);
    }

    #[test]
    fn test_time_scale() {
        let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap();
        let s = TimeScale::new((t0, t1), (0.0, 1000.0));
        let mid = t0 + TimeDelta::try_days(183).unwrap();
        let v = s.scale(&mid);
        assert!(v > 400.0 && v < 600.0);
    }

    #[test]
    fn test_time_invert() {
        let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap();
        let s = TimeScale::new((t0, t1), (0.0, 1000.0));
        let back = s.invert(0.0);
        assert_eq!(back, t0);
    }

    #[test]
    fn test_nice_ticks() {
        let ticks = nice_ticks(0.0, 100.0, 5);
        assert!(!ticks.is_empty());
        // All ticks should be nice round numbers
        for t in &ticks {
            assert!(t >= &0.0 && t <= &100.0);
        }
    }

    #[test]
    fn test_nice_ticks_small_range() {
        let ticks = nice_ticks(0.0, 1.0, 5);
        assert!(!ticks.is_empty());
        for t in &ticks {
            assert!(t >= &0.0 && t <= &1.0);
        }
    }

    #[test]
    fn test_time_ticks() {
        let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap();
        let s = TimeScale::new((t0, t1), (0.0, 1000.0));
        let ticks = s.ticks(12);
        assert!(ticks.len() >= 10);
    }
}
