//! Physical Units and Dimensional Analysis
//!
//! Compile-time enforcement of dimensional correctness.
//! You cannot add Hz to seconds or confuse amplitude with power.

use core::ops::Add;

/// Hertz - frequency unit
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Hertz(pub f64);

impl Hertz {
    pub const fn new(hz: f64) -> Self {
        Self(hz)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }

    pub fn period_seconds(self) -> Seconds {
        Seconds(1.0 / self.0)
    }

    pub fn angular(self) -> RadiansPerSecond {
        RadiansPerSecond(self.0 * 2.0 * core::f64::consts::PI)
    }
}

impl From<f64> for Hertz {
    fn from(hz: f64) -> Self {
        Self(hz)
    }
}

/// Sample rate - samples per second
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRate(pub u32);

impl SampleRate {
    pub const fn new(rate: u32) -> Self {
        Self(rate)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub fn nyquist(self) -> Hertz {
        Hertz(self.0 as f64 / 2.0)
    }

    pub fn period_ns(self) -> i64 {
        1_000_000_000 / (self.0 as i64)
    }
}

/// Seconds - time unit
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Seconds(pub f64);

impl Seconds {
    pub const fn new(s: f64) -> Self {
        Self(s)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }

    pub fn to_nanoseconds(self) -> i64 {
        (self.0 * 1_000_000_000.0) as i64
    }

    pub fn to_milliseconds(self) -> f64 {
        self.0 * 1000.0
    }

    pub fn from_nanoseconds(ns: i64) -> Self {
        Self(ns as f64 / 1_000_000_000.0)
    }

    pub fn from_milliseconds(ms: f64) -> Self {
        Self(ms / 1000.0)
    }
}

/// Radians per second - angular frequency
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RadiansPerSecond(pub f64);

impl RadiansPerSecond {
    pub fn to_hertz(self) -> Hertz {
        Hertz(self.0 / (2.0 * core::f64::consts::PI))
    }
}

/// Decibels - logarithmic ratio
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Decibels(pub f64);

impl Decibels {
    pub const fn new(db: f64) -> Self {
        Self(db)
    }

    pub fn from_power_ratio(ratio: f64) -> Self {
        Self(10.0 * ratio.log10())
    }

    pub fn from_amplitude_ratio(ratio: f64) -> Self {
        Self(20.0 * ratio.log10())
    }

    pub fn to_power_ratio(self) -> f64 {
        10.0_f64.powf(self.0 / 10.0)
    }

    pub fn to_amplitude_ratio(self) -> f64 {
        10.0_f64.powf(self.0 / 20.0)
    }
}

impl Add for Decibels {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        // dB addition: convert to linear, add, convert back
        Self::from_power_ratio(self.to_power_ratio() + rhs.to_power_ratio())
    }
}

/// Frequency band specification
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyBand {
    pub low: Hertz,
    pub high: Hertz,
}

impl FrequencyBand {
    pub fn new(low_hz: f64, high_hz: f64) -> Self {
        Self {
            low: Hertz(low_hz),
            high: Hertz(high_hz),
        }
    }

    pub fn center(&self) -> Hertz {
        Hertz((self.low.0 + self.high.0) / 2.0)
    }

    pub fn bandwidth(&self) -> Hertz {
        Hertz(self.high.0 - self.low.0)
    }

    pub fn contains(&self, freq: Hertz) -> bool {
        freq.0 >= self.low.0 && freq.0 <= self.high.0
    }

    /// Standard physiological frequency bands
    pub fn delta() -> Self {
        Self::new(0.5, 4.0)
    }

    pub fn theta() -> Self {
        Self::new(4.0, 8.0)
    }

    pub fn alpha() -> Self {
        Self::new(8.0, 13.0)
    }

    pub fn beta() -> Self {
        Self::new(13.0, 30.0)
    }

    pub fn gamma() -> Self {
        Self::new(30.0, 100.0)
    }

    /// Parkinsonian tremor band
    pub fn parkinsonian_tremor() -> Self {
        Self::new(4.0, 12.0)
    }

    /// Essential tremor band
    pub fn essential_tremor() -> Self {
        Self::new(4.0, 8.0)
    }
}

/// Physical quantity marker traits for type safety
pub mod markers {
    /// Marker for time-domain quantities
    pub trait TimeDomain {}

    /// Marker for frequency-domain quantities
    pub trait FrequencyDomain {}

    /// Marker for power spectral density
    pub trait PowerSpectrum {}

    /// Marker for phase information
    pub trait PhaseSpectrum {}
}

/// Power Spectral Density at a single frequency point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsdPoint {
    pub value: f64,
    pub frequency: Hertz,
    pub bandwidth: Hertz,
}

/// Coherence value (0-1, unitless)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coherence(pub f64);

impl Coherence {
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}

/// Phase in radians (-π to π)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Phase(pub f64);

impl Phase {
    pub fn new(radians: f64) -> Self {
        // Wrap to [-π, π]
        let mut r = radians % (2.0 * core::f64::consts::PI);
        if r > core::f64::consts::PI {
            r -= 2.0 * core::f64::consts::PI;
        } else if r < -core::f64::consts::PI {
            r += 2.0 * core::f64::consts::PI;
        }
        Self(r)
    }

    pub fn from_degrees(degrees: f64) -> Self {
        Self::new(degrees * core::f64::consts::PI / 180.0)
    }

    pub fn to_degrees(self) -> f64 {
        self.0 * 180.0 / core::f64::consts::PI
    }

    pub fn as_radians(self) -> f64 {
        self.0
    }
}

/// Time window specification
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeWindow {
    pub duration: Seconds,
    pub step: Option<Seconds>,
    pub offset: Seconds,
}

impl TimeWindow {
    pub fn tumbling(duration_s: f64) -> Self {
        Self {
            duration: Seconds(duration_s),
            step: Some(Seconds(duration_s)),
            offset: Seconds(0.0),
        }
    }

    pub fn sliding(duration_s: f64, step_s: f64) -> Self {
        Self {
            duration: Seconds(duration_s),
            step: Some(Seconds(step_s)),
            offset: Seconds(0.0),
        }
    }

    pub fn session(duration_s: f64) -> Self {
        Self {
            duration: Seconds(duration_s),
            step: None,
            offset: Seconds(0.0),
        }
    }

    pub fn with_offset(mut self, offset_s: f64) -> Self {
        self.offset = Seconds(offset_s);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hertz_period() {
        let hz = Hertz::new(100.0);
        let period = hz.period_seconds();
        assert!((period.0 - 0.01).abs() < 0.0001);
    }

    #[test]
    fn test_decibels_conversion() {
        let db = Decibels::from_power_ratio(100.0);
        assert!((db.0 - 20.0).abs() < 0.001);

        let ratio = db.to_power_ratio();
        assert!((ratio - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_frequency_band() {
        let band = FrequencyBand::parkinsonian_tremor();
        assert!(band.contains(Hertz(6.0)));
        assert!(!band.contains(Hertz(15.0)));
    }

    #[test]
    fn test_phase_wrapping() {
        let p1 = Phase::new(4.0 * core::f64::consts::PI + 0.5);
        assert!((p1.0 - 0.5).abs() < 0.001);

        let p2 = Phase::new(-5.0 * core::f64::consts::PI);
        assert!((p2.0 - (-core::f64::consts::PI)).abs() < 0.001);
    }
}
