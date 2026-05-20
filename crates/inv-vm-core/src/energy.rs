use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Sub};

/// Energy in joules. The fundamental unit of measurement.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
pub struct Joules(f64);

impl Joules {
    pub const ZERO: Self = Self(0.0);

    pub fn new(value: f64) -> Self {
        Self(value)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }

    pub fn millijoules(self) -> f64 {
        self.0 * 1000.0
    }

    pub fn microjoules(self) -> f64 {
        self.0 * 1_000_000.0
    }
}

impl fmt::Display for Joules {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 < 0.001 {
            write!(f, "{:.2}µJ", self.microjoules())
        } else if self.0 < 1.0 {
            write!(f, "{:.2}mJ", self.millijoules())
        } else {
            write!(f, "{:.2}J", self.0)
        }
    }
}

impl Add for Joules {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Joules {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Joules {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

/// Power in watts. Instantaneous measurement.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
pub struct Watts(f64);

impl Watts {
    pub const ZERO: Self = Self(0.0);

    pub fn new(value: f64) -> Self {
        Self(value)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }

    /// Convert to joules over a duration in seconds.
    pub fn to_joules(self, seconds: f64) -> Joules {
        Joules::new(self.0 * seconds)
    }
}

impl fmt::Display for Watts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 < 1.0 {
            write!(f, "{:.1}mW", self.0 * 1000.0)
        } else {
            write!(f, "{:.1}W", self.0)
        }
    }
}

/// A point-in-time energy reading from a meter.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EnergyReading {
    /// Total energy consumed since meter start.
    pub joules: Joules,
    /// Current instantaneous power draw.
    pub watts_current: Watts,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl EnergyReading {
    pub fn new(joules: Joules, watts: Watts, timestamp_ms: u64) -> Self {
        Self {
            joules,
            watts_current: watts,
            timestamp_ms,
        }
    }

    /// Compute the energy delta between two readings.
    pub fn delta(&self, previous: &EnergyReading) -> Joules {
        self.joules - previous.joules
    }
}

/// An energy budget that can be enforced at runtime.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EnergyBudget {
    /// Maximum joules this workload may consume.
    pub max_joules: Joules,
    /// Joules consumed so far.
    pub consumed: Joules,
}

impl EnergyBudget {
    pub fn new(max_joules: Joules) -> Self {
        Self {
            max_joules,
            consumed: Joules::ZERO,
        }
    }

    /// Remaining energy in the budget.
    pub fn remaining(&self) -> Joules {
        self.max_joules - self.consumed
    }

    /// Whether the budget has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.consumed.as_f64() >= self.max_joules.as_f64()
    }

    /// Record energy consumption.
    pub fn consume(&mut self, joules: Joules) {
        self.consumed += joules;
    }

    /// Fraction of budget consumed (0.0 to 1.0+).
    pub fn utilization(&self) -> f64 {
        if self.max_joules.as_f64() == 0.0 {
            return 0.0;
        }
        self.consumed.as_f64() / self.max_joules.as_f64()
    }
}

/// The source of energy powering a node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnergySource {
    /// Plugged into wall power (grid).
    WallPower,
    /// Running on battery.
    Battery,
    /// Solar powered.
    Solar,
    /// Orbital solar — satellite photovoltaic panels in direct sunlight.
    /// Zero carbon, zero cooling cost.
    OrbitalSolar,
    /// Orbital battery — satellite in Earth's shadow, draining stored energy.
    OrbitalBattery,
    /// Unknown energy source.
    #[default]
    Unknown,
}

/// Thermal state of a node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThermalState {
    /// Normal operating temperature.
    #[default]
    Normal,
    /// Warm — approaching throttling threshold.
    Warm,
    /// Actively throttled to reduce temperature.
    Throttled,
    /// Critical — shedding workloads.
    Critical,
    /// Orbital vacuum — radiative cooling only, no convection.
    OrbitalVacuum,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joules_arithmetic() {
        let a = Joules::new(1.5);
        let b = Joules::new(2.3);
        assert!((a + b).as_f64() - 3.8 < 1e-10);
        assert!((b - a).as_f64() - 0.8 < 1e-10);
    }

    #[test]
    fn joules_display() {
        assert_eq!(format!("{}", Joules::new(1.5)), "1.50J");
        assert_eq!(format!("{}", Joules::new(0.5)), "500.00mJ");
        assert_eq!(format!("{}", Joules::new(0.0005)), "500.00µJ");
    }

    #[test]
    fn watts_to_joules() {
        let watts = Watts::new(10.0);
        let joules = watts.to_joules(3.0);
        assert!((joules.as_f64() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn energy_budget_lifecycle() {
        let mut budget = EnergyBudget::new(Joules::new(5.0));
        assert!(!budget.is_exceeded());
        assert!((budget.remaining().as_f64() - 5.0).abs() < 1e-10);

        budget.consume(Joules::new(3.0));
        assert!(!budget.is_exceeded());
        assert!((budget.remaining().as_f64() - 2.0).abs() < 1e-10);
        assert!((budget.utilization() - 0.6).abs() < 1e-10);

        budget.consume(Joules::new(2.5));
        assert!(budget.is_exceeded());
    }

    #[test]
    fn energy_reading_delta() {
        let prev = EnergyReading::new(Joules::new(10.0), Watts::new(5.0), 1000);
        let curr = EnergyReading::new(Joules::new(15.0), Watts::new(6.0), 2000);
        let delta = curr.delta(&prev);
        assert!((delta.as_f64() - 5.0).abs() < 1e-10);
    }
}
