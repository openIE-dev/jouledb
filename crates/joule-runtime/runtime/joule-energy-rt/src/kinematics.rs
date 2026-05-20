//! Kinematic Energy Telemetry
//!
//! Provides full derivative chain for energy and thermal measurements,
//! following the kinematic approach: position → velocity → acceleration →
//! jerk → snap → crackle → pop.
//!
//! # Energy Derivatives
//!
//! | Order | Name | Symbol | Unit | Physical Meaning |
//! |-------|------|--------|------|------------------|
//! | 0 | Energy | E | J | Total work done |
//! | 1 | Power | P = dE/dt | W | Consumption rate |
//! | 2 | Power Rate | dP/dt | W/s | Consumption acceleration |
//! | 3 | Power Jerk | d²P/dt² | W/s² | Smoothness of power ramps |
//! | 4 | Power Snap | d³P/dt³ | W/s³ | Transient behavior |
//! | 5 | Power Crackle | d⁴P/dt⁴ | W/s⁴ | Micro-fluctuations |
//! | 6 | Power Pop | d⁵P/dt⁵ | W/s⁵ | Ultra-fine dynamics |
//!
//! # Thermal Derivatives
//!
//! | Order | Name | Symbol | Unit |
//! |-------|------|--------|------|
//! | 0 | Temperature | T | °C |
//! | 1 | Heating Rate | dT/dt | °C/s |
//! | 2 | Thermal Accel | d²T/dt² | °C/s² |
//! | 3+ | Higher | ... | ... |
//!
//! # Example
//!
//! ```ignore
//! use joule_energy_rt::kinematics::{KinematicMonitor, KinematicConfig};
//!
//! let config = KinematicConfig::default()
//!     .with_sample_rate_hz(1000.0)  // 1kHz sampling
//!     .with_derivative_order(6);     // Up to pop
//!
//! let mut monitor = KinematicMonitor::new(config)?;
//! monitor.start()?;
//!
//! // ... computation ...
//!
//! let telemetry = monitor.stop()?;
//! println!("Power jerk: {:.3} W/s²", telemetry.energy.jerk);
//! println!("Heating rate: {:.3} °C/s", telemetry.thermal.velocity);
//! ```

use crate::error::{Error, Result};
use crate::platform::{EnergyReader, create_reader};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for kinematic monitoring
#[derive(Debug, Clone)]
pub struct KinematicConfig {
    /// Sample rate in Hz (samples per second)
    pub sample_rate_hz: f64,
    /// Maximum derivative order to compute (1-6)
    pub derivative_order: usize,
    /// Ring buffer size (number of samples to retain)
    pub buffer_size: usize,
    /// Whether to compute thermal derivatives
    pub track_thermal: bool,
    /// Whether to compute efficiency metrics
    pub track_efficiency: bool,
    /// Smoothing window for derivative computation (samples)
    pub smoothing_window: usize,
}

impl Default for KinematicConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 100.0, // 100 Hz default
            derivative_order: 6,   // Up to pop
            buffer_size: 1024,     // ~10 seconds at 100Hz
            track_thermal: true,
            track_efficiency: true,
            smoothing_window: 5, // 5-point smoothing
        }
    }
}

impl KinematicConfig {
    /// Set sample rate in Hz
    #[must_use]
    pub fn with_sample_rate_hz(mut self, hz: f64) -> Self {
        self.sample_rate_hz = hz.max(1.0).min(10000.0);
        self
    }

    /// Set maximum derivative order (1-6)
    #[must_use]
    pub fn with_derivative_order(mut self, order: usize) -> Self {
        self.derivative_order = order.clamp(1, 6);
        self
    }

    /// Set buffer size
    #[must_use]
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size.max(16);
        self
    }

    /// Enable/disable thermal tracking
    #[must_use]
    pub fn with_thermal(mut self, enabled: bool) -> Self {
        self.track_thermal = enabled;
        self
    }

    /// Enable/disable efficiency metrics
    #[must_use]
    pub fn with_efficiency(mut self, enabled: bool) -> Self {
        self.track_efficiency = enabled;
        self
    }

    /// Set smoothing window size
    #[must_use]
    pub fn with_smoothing(mut self, window: usize) -> Self {
        self.smoothing_window = window.max(1).min(32);
        self
    }

    /// Get sample interval as Duration
    pub fn sample_interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.sample_rate_hz)
    }
}

// ============================================================================
// Time-Series Sample
// ============================================================================

/// A single timestamped sample
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Timestamp (seconds from start)
    pub time_s: f64,
    /// Energy reading (Joules)
    pub energy_j: f64,
    /// Temperature reading (Celsius), if available
    pub temp_c: Option<f64>,
    /// Operation count (for efficiency), if available
    pub ops: Option<u64>,
    /// Bytes processed (for efficiency), if available
    pub bytes: Option<u64>,
}

// ============================================================================
// Derivative Chain
// ============================================================================

/// Complete derivative chain for a quantity
/// Named after kinematic derivatives: position → velocity → acceleration →
/// jerk → snap → crackle → pop
#[derive(Debug, Clone, Copy, Default)]
pub struct DerivativeChain {
    /// 0th derivative: the base quantity
    pub value: f64,
    /// 1st derivative: rate of change (velocity)
    pub velocity: f64,
    /// 2nd derivative: acceleration
    pub acceleration: f64,
    /// 3rd derivative: jerk
    pub jerk: f64,
    /// 4th derivative: snap (jounce)
    pub snap: f64,
    /// 5th derivative: crackle
    pub crackle: f64,
    /// 6th derivative: pop
    pub pop: f64,
}

impl DerivativeChain {
    /// Create a new derivative chain with just the base value
    pub fn new(value: f64) -> Self {
        Self {
            value,
            ..Default::default()
        }
    }

    /// Get derivative by order (0-6)
    pub fn get(&self, order: usize) -> f64 {
        match order {
            0 => self.value,
            1 => self.velocity,
            2 => self.acceleration,
            3 => self.jerk,
            4 => self.snap,
            5 => self.crackle,
            6 => self.pop,
            _ => 0.0,
        }
    }

    /// Set derivative by order (0-6)
    pub fn set(&mut self, order: usize, value: f64) {
        match order {
            0 => self.value = value,
            1 => self.velocity = value,
            2 => self.acceleration = value,
            3 => self.jerk = value,
            4 => self.snap = value,
            5 => self.crackle = value,
            6 => self.pop = value,
            _ => {}
        }
    }

    /// Get the name of a derivative order
    pub fn name(order: usize) -> &'static str {
        match order {
            0 => "value",
            1 => "velocity",
            2 => "acceleration",
            3 => "jerk",
            4 => "snap",
            5 => "crackle",
            6 => "pop",
            _ => "unknown",
        }
    }
}

/// Energy derivative chain with proper units
#[derive(Debug, Clone, Copy, Default)]
pub struct EnergyDerivatives {
    /// Energy (J)
    pub energy_j: f64,
    /// Power = dE/dt (W)
    pub power_w: f64,
    /// Power rate = d²E/dt² (W/s)
    pub power_rate_w_per_s: f64,
    /// Power jerk = d³E/dt³ (W/s²)
    pub power_jerk_w_per_s2: f64,
    /// Power snap = d⁴E/dt⁴ (W/s³)
    pub power_snap_w_per_s3: f64,
    /// Power crackle = d⁵E/dt⁵ (W/s⁴)
    pub power_crackle_w_per_s4: f64,
    /// Power pop = d⁶E/dt⁶ (W/s⁵)
    pub power_pop_w_per_s5: f64,
}

impl EnergyDerivatives {
    /// Create from a generic derivative chain
    pub fn from_chain(chain: &DerivativeChain) -> Self {
        Self {
            energy_j: chain.value,
            power_w: chain.velocity,
            power_rate_w_per_s: chain.acceleration,
            power_jerk_w_per_s2: chain.jerk,
            power_snap_w_per_s3: chain.snap,
            power_crackle_w_per_s4: chain.crackle,
            power_pop_w_per_s5: chain.pop,
        }
    }
}

/// Thermal derivative chain with proper units
#[derive(Debug, Clone, Copy, Default)]
pub struct ThermalDerivatives {
    /// Temperature (°C)
    pub temp_c: f64,
    /// Heating rate = dT/dt (°C/s)
    pub heating_rate_c_per_s: f64,
    /// Thermal acceleration = d²T/dt² (°C/s²)
    pub thermal_accel_c_per_s2: f64,
    /// Thermal jerk = d³T/dt³ (°C/s³)
    pub thermal_jerk_c_per_s3: f64,
    /// Thermal snap = d⁴T/dt⁴ (°C/s⁴)
    pub thermal_snap_c_per_s4: f64,
    /// Thermal crackle = d⁵T/dt⁵ (°C/s⁵)
    pub thermal_crackle_c_per_s5: f64,
    /// Thermal pop = d⁶T/dt⁶ (°C/s⁶)
    pub thermal_pop_c_per_s6: f64,
}

impl ThermalDerivatives {
    /// Create from a generic derivative chain
    pub fn from_chain(chain: &DerivativeChain) -> Self {
        Self {
            temp_c: chain.value,
            heating_rate_c_per_s: chain.velocity,
            thermal_accel_c_per_s2: chain.acceleration,
            thermal_jerk_c_per_s3: chain.jerk,
            thermal_snap_c_per_s4: chain.snap,
            thermal_crackle_c_per_s5: chain.crackle,
            thermal_pop_c_per_s6: chain.pop,
        }
    }
}

// ============================================================================
// Thermodynamic Coupling
// ============================================================================

/// Thermodynamic relationships between energy and temperature
#[derive(Debug, Clone, Copy, Default)]
pub struct ThermodynamicCoupling {
    /// Effective heat capacity: dE/dT (J/°C)
    /// How much energy is needed to raise temperature by 1°C
    pub heat_capacity_j_per_c: f64,

    /// Thermal resistance: dT/dP (°C/W)
    /// Temperature rise per watt of power
    pub thermal_resistance_c_per_w: f64,

    /// Thermal time constant: τ = R × C (seconds)
    /// Characteristic time for thermal response
    pub thermal_time_constant_s: f64,

    /// Instantaneous entropy rate: dS/dt (J/(°C·s))
    /// Rate of entropy generation (approximation)
    pub entropy_rate_j_per_c_s: f64,

    /// Coefficient of performance (dimensionless)
    /// Ratio of useful work to total energy
    pub coefficient_of_performance: f64,
}

impl ThermodynamicCoupling {
    /// Compute thermodynamic coupling from energy and thermal derivatives
    pub fn compute(energy: &EnergyDerivatives, thermal: &ThermalDerivatives) -> Self {
        // Heat capacity: dE/dT ≈ (dE/dt) / (dT/dt) = P / (dT/dt)
        let heat_capacity = if thermal.heating_rate_c_per_s.abs() > 1e-10 {
            energy.power_w / thermal.heating_rate_c_per_s
        } else {
            0.0
        };

        // Thermal resistance: dT/dP ≈ (dT/dt) / (dP/dt)
        let thermal_resistance = if energy.power_rate_w_per_s.abs() > 1e-10 {
            thermal.heating_rate_c_per_s / energy.power_rate_w_per_s
        } else if energy.power_w.abs() > 1e-10 {
            // Fallback: steady state approximation T/P
            thermal.temp_c / energy.power_w
        } else {
            0.0
        };

        // Time constant τ = R × C
        let time_constant = heat_capacity.abs() * thermal_resistance.abs();

        // Entropy rate: dS/dt ≈ P / T (simplified Clausius inequality)
        let entropy_rate = if thermal.temp_c > 0.0 {
            energy.power_w / (thermal.temp_c + 273.15) // Convert to Kelvin
        } else {
            0.0
        };

        // COP: in a computational context, this is related to efficiency
        // Here we use a simplified version based on thermal overhead
        let cop = if thermal.heating_rate_c_per_s > 0.0 {
            1.0 / (1.0 + thermal.heating_rate_c_per_s * 0.01)
        } else {
            1.0
        };

        Self {
            heat_capacity_j_per_c: heat_capacity,
            thermal_resistance_c_per_w: thermal_resistance,
            thermal_time_constant_s: time_constant,
            entropy_rate_j_per_c_s: entropy_rate,
            coefficient_of_performance: cop,
        }
    }
}

// ============================================================================
// Efficiency Metrics
// ============================================================================

/// Computational efficiency metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct EfficiencyMetrics {
    /// Operations per Joule (ops/J)
    pub ops_per_joule: f64,

    /// FLOPS per Watt (FLOP/W) - if tracking floating point ops
    pub flops_per_watt: f64,

    /// Bytes processed per Joule (B/J)
    pub bytes_per_joule: f64,

    /// Bandwidth per Watt (B/s/W = B/J)
    pub bandwidth_per_watt: f64,

    /// Energy-Delay Product (J·s) - lower is better
    /// Measures both energy and time together
    pub energy_delay_product_j_s: f64,

    /// Energy-Delay² Product (J·s²) - for parallel scaling
    pub energy_delay_squared_j_s2: f64,

    /// Power-Delay Product (W·s = J) - just energy
    pub power_delay_product_j: f64,

    /// Specific energy: energy per operation (J/op)
    pub specific_energy_j_per_op: f64,
}

impl EfficiencyMetrics {
    /// Compute efficiency metrics from telemetry
    pub fn compute(
        total_energy_j: f64,
        total_time_s: f64,
        total_ops: u64,
        total_bytes: u64,
        average_power_w: f64,
    ) -> Self {
        let ops = total_ops as f64;
        let bytes = total_bytes as f64;

        // Ops per Joule
        let ops_per_joule = if total_energy_j > 0.0 {
            ops / total_energy_j
        } else {
            0.0
        };

        // FLOPS per Watt (assuming ops are FLOPs)
        let flops_per_watt = if average_power_w > 0.0 && total_time_s > 0.0 {
            ops / total_time_s / average_power_w
        } else {
            0.0
        };

        // Bytes per Joule
        let bytes_per_joule = if total_energy_j > 0.0 {
            bytes / total_energy_j
        } else {
            0.0
        };

        // Bandwidth per Watt
        let bandwidth_per_watt = if average_power_w > 0.0 && total_time_s > 0.0 {
            bytes / total_time_s / average_power_w
        } else {
            0.0
        };

        // Energy-Delay Product
        let edp = total_energy_j * total_time_s;

        // Energy-Delay² Product
        let ed2p = total_energy_j * total_time_s * total_time_s;

        // Power-Delay Product (= Energy)
        let pdp = average_power_w * total_time_s;

        // Specific energy (J/op)
        let specific_energy = if ops > 0.0 { total_energy_j / ops } else { 0.0 };

        Self {
            ops_per_joule,
            flops_per_watt,
            bytes_per_joule,
            bandwidth_per_watt,
            energy_delay_product_j_s: edp,
            energy_delay_squared_j_s2: ed2p,
            power_delay_product_j: pdp,
            specific_energy_j_per_op: specific_energy,
        }
    }
}

// ============================================================================
// Complete Telemetry
// ============================================================================

/// Complete kinematic telemetry output
#[derive(Debug, Clone, Default)]
pub struct KinematicTelemetry {
    /// Total duration of measurement
    pub duration_s: f64,

    /// Number of samples collected
    pub sample_count: usize,

    /// Sample rate achieved (Hz)
    pub actual_sample_rate_hz: f64,

    /// Energy derivatives (up to pop)
    pub energy: EnergyDerivatives,

    /// Thermal derivatives (up to pop)
    pub thermal: ThermalDerivatives,

    /// Thermodynamic coupling metrics
    pub thermodynamics: ThermodynamicCoupling,

    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics,

    /// Raw samples (if retained)
    pub samples: Vec<Sample>,

    /// Statistics
    pub stats: TelemetryStats,
}

/// Statistical summary of telemetry
#[derive(Debug, Clone, Default)]
pub struct TelemetryStats {
    /// Minimum power observed (W)
    pub power_min_w: f64,
    /// Maximum power observed (W)
    pub power_max_w: f64,
    /// Mean power (W)
    pub power_mean_w: f64,
    /// Standard deviation of power (W)
    pub power_stddev_w: f64,

    /// Minimum temperature observed (°C)
    pub temp_min_c: f64,
    /// Maximum temperature observed (°C)
    pub temp_max_c: f64,
    /// Mean temperature (°C)
    pub temp_mean_c: f64,

    /// Peak power jerk (W/s²)
    pub peak_jerk_w_per_s2: f64,
    /// Peak heating rate (°C/s)
    pub peak_heating_rate_c_per_s: f64,
}

// ============================================================================
// Derivative Computation
// ============================================================================

/// Compute derivatives from a time series using finite differences
/// with optional smoothing (Savitzky-Golay style)
pub struct DerivativeComputer {
    /// Maximum derivative order
    max_order: usize,
    /// Smoothing window size
    window_size: usize,
}

impl DerivativeComputer {
    /// Create a new derivative computer
    pub fn new(max_order: usize, window_size: usize) -> Self {
        Self {
            max_order: max_order.clamp(1, 6),
            window_size: window_size.max(1),
        }
    }

    /// Compute all derivatives from samples
    /// Returns derivative chains for energy and temperature
    pub fn compute_derivatives(&self, samples: &[Sample]) -> (DerivativeChain, DerivativeChain) {
        if samples.len() < 2 {
            return (DerivativeChain::default(), DerivativeChain::default());
        }

        let n = samples.len();
        let dt = if n > 1 {
            (samples[n - 1].time_s - samples[0].time_s) / (n - 1) as f64
        } else {
            1.0
        };

        // Extract energy and temperature time series
        let energy: Vec<f64> = samples.iter().map(|s| s.energy_j).collect();
        let temp: Vec<f64> = samples.iter().map(|s| s.temp_c.unwrap_or(0.0)).collect();

        // Compute derivatives for both
        let energy_chain = self.compute_chain(&energy, dt);
        let temp_chain = self.compute_chain(&temp, dt);

        (energy_chain, temp_chain)
    }

    /// Compute derivative chain for a single time series
    fn compute_chain(&self, values: &[f64], dt: f64) -> DerivativeChain {
        let mut chain = DerivativeChain::default();

        if values.is_empty() {
            return chain;
        }

        // 0th derivative: use last value
        chain.value = *values.last().unwrap();

        // Compute successive derivatives
        let mut current = values.to_vec();

        for order in 1..=self.max_order {
            let derivative = self.differentiate(&current, dt);
            if derivative.is_empty() {
                break;
            }

            // Use smoothed final value
            let smoothed = self.smooth(&derivative);
            chain.set(order, *smoothed.last().unwrap_or(&0.0));

            current = derivative;
        }

        chain
    }

    /// Compute first derivative using central differences
    fn differentiate(&self, values: &[f64], dt: f64) -> Vec<f64> {
        if values.len() < 2 {
            return vec![];
        }

        let n = values.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let derivative = if i == 0 {
                // Forward difference
                (values[1] - values[0]) / dt
            } else if i == n - 1 {
                // Backward difference
                (values[n - 1] - values[n - 2]) / dt
            } else {
                // Central difference
                (values[i + 1] - values[i - 1]) / (2.0 * dt)
            };
            result.push(derivative);
        }

        result
    }

    /// Apply simple moving average smoothing
    fn smooth(&self, values: &[f64]) -> Vec<f64> {
        if values.len() <= self.window_size {
            return values.to_vec();
        }

        let half_window = self.window_size / 2;
        let mut result = Vec::with_capacity(values.len());

        for i in 0..values.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(values.len());
            let sum: f64 = values[start..end].iter().sum();
            result.push(sum / (end - start) as f64);
        }

        result
    }
}

// ============================================================================
// Kinematic Monitor
// ============================================================================

/// High-fidelity kinematic energy monitor
pub struct KinematicMonitor {
    config: KinematicConfig,
    reader: Box<dyn EnergyReader>,
    samples: VecDeque<Sample>,
    start_time: Option<Instant>,
    start_energy: f64,
    total_ops: u64,
    total_bytes: u64,
    running: Arc<AtomicBool>,
    /// Reserved for future background sampling implementation
    #[allow(dead_code)]
    sample_thread: Option<JoinHandle<Vec<Sample>>>,
}

impl KinematicMonitor {
    /// Create a new kinematic monitor
    pub fn new(config: KinematicConfig) -> Result<Self> {
        let reader = create_reader()?;
        let start_energy = reader.read_energy()?;

        Ok(Self {
            config,
            reader,
            samples: VecDeque::new(),
            start_time: None,
            start_energy,
            total_ops: 0,
            total_bytes: 0,
            running: Arc::new(AtomicBool::new(false)),
            sample_thread: None,
        })
    }

    /// Start background sampling
    pub fn start(&mut self) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(Error::AlreadyRunning);
        }

        self.start_time = Some(Instant::now());
        self.start_energy = self.reader.read_energy()?;
        self.samples.clear();
        self.running.store(true, Ordering::SeqCst);

        // For now, we'll do synchronous sampling
        // In a full implementation, this would spawn a background thread

        Ok(())
    }

    /// Record a sample (call periodically or use auto-sampling)
    pub fn sample(&mut self) -> Result<()> {
        let start_time = self.start_time.ok_or(Error::NotStarted)?;
        let elapsed = start_time.elapsed();

        let energy = self.reader.read_energy()?;
        let temp = self.reader.read_temperature().ok();

        let sample = Sample {
            time_s: elapsed.as_secs_f64(),
            energy_j: energy - self.start_energy,
            temp_c: temp,
            ops: if self.total_ops > 0 {
                Some(self.total_ops)
            } else {
                None
            },
            bytes: if self.total_bytes > 0 {
                Some(self.total_bytes)
            } else {
                None
            },
        };

        self.samples.push_back(sample);

        // Maintain buffer size
        while self.samples.len() > self.config.buffer_size {
            self.samples.pop_front();
        }

        Ok(())
    }

    /// Record operation count (for efficiency metrics)
    pub fn record_ops(&mut self, ops: u64) {
        self.total_ops = self.total_ops.saturating_add(ops);
    }

    /// Record bytes processed (for efficiency metrics)
    pub fn record_bytes(&mut self, bytes: u64) {
        self.total_bytes = self.total_bytes.saturating_add(bytes);
    }

    /// Stop monitoring and compute full telemetry
    pub fn stop(&mut self) -> Result<KinematicTelemetry> {
        self.running.store(false, Ordering::SeqCst);

        let start_time = self.start_time.ok_or(Error::NotStarted)?;
        let duration = start_time.elapsed();
        let duration_s = duration.as_secs_f64();

        // Take one final sample
        let _ = self.sample();

        // Convert samples to vec
        let samples: Vec<Sample> = self.samples.iter().copied().collect();
        let sample_count = samples.len();

        if sample_count < 2 {
            return Ok(KinematicTelemetry {
                duration_s,
                sample_count,
                ..Default::default()
            });
        }

        // Compute derivatives
        let computer =
            DerivativeComputer::new(self.config.derivative_order, self.config.smoothing_window);
        let (energy_chain, temp_chain) = computer.compute_derivatives(&samples);

        let energy = EnergyDerivatives::from_chain(&energy_chain);
        let thermal = ThermalDerivatives::from_chain(&temp_chain);

        // Compute thermodynamic coupling
        let thermodynamics = ThermodynamicCoupling::compute(&energy, &thermal);

        // Compute efficiency metrics
        let efficiency = EfficiencyMetrics::compute(
            energy.energy_j,
            duration_s,
            self.total_ops,
            self.total_bytes,
            energy.power_w,
        );

        // Compute statistics
        let stats = self.compute_stats(&samples, &energy_chain, &temp_chain);

        Ok(KinematicTelemetry {
            duration_s,
            sample_count,
            actual_sample_rate_hz: sample_count as f64 / duration_s,
            energy,
            thermal,
            thermodynamics,
            efficiency,
            samples,
            stats,
        })
    }

    /// Compute statistical summary
    fn compute_stats(
        &self,
        samples: &[Sample],
        energy_chain: &DerivativeChain,
        temp_chain: &DerivativeChain,
    ) -> TelemetryStats {
        if samples.len() < 2 {
            return TelemetryStats::default();
        }

        // Compute power from consecutive samples
        let mut powers = Vec::with_capacity(samples.len() - 1);
        for i in 1..samples.len() {
            let dt = samples[i].time_s - samples[i - 1].time_s;
            if dt > 0.0 {
                let de = samples[i].energy_j - samples[i - 1].energy_j;
                powers.push(de / dt);
            }
        }

        // Power statistics
        let power_min = powers.iter().cloned().fold(f64::INFINITY, f64::min);
        let power_max = powers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let power_mean = powers.iter().sum::<f64>() / powers.len() as f64;
        let power_variance =
            powers.iter().map(|p| (p - power_mean).powi(2)).sum::<f64>() / powers.len() as f64;
        let power_stddev = power_variance.sqrt();

        // Temperature statistics
        let temps: Vec<f64> = samples.iter().filter_map(|s| s.temp_c).collect();
        let (temp_min, temp_max, temp_mean) = if !temps.is_empty() {
            let min = temps.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = temps.iter().sum::<f64>() / temps.len() as f64;
            (min, max, mean)
        } else {
            (0.0, 0.0, 0.0)
        };

        TelemetryStats {
            power_min_w: power_min,
            power_max_w: power_max,
            power_mean_w: power_mean,
            power_stddev_w: power_stddev,
            temp_min_c: temp_min,
            temp_max_c: temp_max,
            temp_mean_c: temp_mean,
            peak_jerk_w_per_s2: energy_chain.jerk.abs(),
            peak_heating_rate_c_per_s: temp_chain.velocity.abs(),
        }
    }
}

// ============================================================================
// Display Implementations
// ============================================================================

impl std::fmt::Display for EnergyDerivatives {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Energy Derivatives:")?;
        writeln!(f, "  Energy:        {:>12.6} J", self.energy_j)?;
        writeln!(f, "  Power:         {:>12.6} W", self.power_w)?;
        writeln!(f, "  Power Rate:    {:>12.6} W/s", self.power_rate_w_per_s)?;
        writeln!(
            f,
            "  Power Jerk:    {:>12.6} W/s²",
            self.power_jerk_w_per_s2
        )?;
        writeln!(
            f,
            "  Power Snap:    {:>12.6} W/s³",
            self.power_snap_w_per_s3
        )?;
        writeln!(
            f,
            "  Power Crackle: {:>12.6} W/s⁴",
            self.power_crackle_w_per_s4
        )?;
        writeln!(f, "  Power Pop:     {:>12.6} W/s⁵", self.power_pop_w_per_s5)
    }
}

impl std::fmt::Display for ThermalDerivatives {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Thermal Derivatives:")?;
        writeln!(f, "  Temperature:      {:>12.3} °C", self.temp_c)?;
        writeln!(
            f,
            "  Heating Rate:     {:>12.6} °C/s",
            self.heating_rate_c_per_s
        )?;
        writeln!(
            f,
            "  Thermal Accel:    {:>12.6} °C/s²",
            self.thermal_accel_c_per_s2
        )?;
        writeln!(
            f,
            "  Thermal Jerk:     {:>12.6} °C/s³",
            self.thermal_jerk_c_per_s3
        )?;
        writeln!(
            f,
            "  Thermal Snap:     {:>12.6} °C/s⁴",
            self.thermal_snap_c_per_s4
        )?;
        writeln!(
            f,
            "  Thermal Crackle:  {:>12.6} °C/s⁵",
            self.thermal_crackle_c_per_s5
        )?;
        writeln!(
            f,
            "  Thermal Pop:      {:>12.6} °C/s⁶",
            self.thermal_pop_c_per_s6
        )
    }
}

impl std::fmt::Display for ThermodynamicCoupling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Thermodynamic Coupling:")?;
        writeln!(
            f,
            "  Heat Capacity:       {:>12.3} J/°C",
            self.heat_capacity_j_per_c
        )?;
        writeln!(
            f,
            "  Thermal Resistance:  {:>12.6} °C/W",
            self.thermal_resistance_c_per_w
        )?;
        writeln!(
            f,
            "  Time Constant:       {:>12.3} s",
            self.thermal_time_constant_s
        )?;
        writeln!(
            f,
            "  Entropy Rate:        {:>12.6} J/(K·s)",
            self.entropy_rate_j_per_c_s
        )?;
        writeln!(
            f,
            "  COP:                 {:>12.3}",
            self.coefficient_of_performance
        )
    }
}

impl std::fmt::Display for EfficiencyMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Efficiency Metrics:")?;
        writeln!(
            f,
            "  Ops/Joule:           {:>12.3e} ops/J",
            self.ops_per_joule
        )?;
        writeln!(
            f,
            "  FLOPS/Watt:          {:>12.3e} FLOP/W",
            self.flops_per_watt
        )?;
        writeln!(
            f,
            "  Bytes/Joule:         {:>12.3e} B/J",
            self.bytes_per_joule
        )?;
        writeln!(
            f,
            "  Bandwidth/Watt:      {:>12.3e} B/s/W",
            self.bandwidth_per_watt
        )?;
        writeln!(
            f,
            "  Energy-Delay:        {:>12.6e} J·s",
            self.energy_delay_product_j_s
        )?;
        writeln!(
            f,
            "  Energy-Delay²:       {:>12.6e} J·s²",
            self.energy_delay_squared_j_s2
        )?;
        writeln!(
            f,
            "  Specific Energy:     {:>12.6e} J/op",
            self.specific_energy_j_per_op
        )
    }
}

impl std::fmt::Display for KinematicTelemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "╔══════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║              KINEMATIC ENERGY TELEMETRY                      ║"
        )?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║  Duration: {:.3} s | Samples: {} | Rate: {:.1} Hz",
            self.duration_s, self.sample_count, self.actual_sample_rate_hz
        )?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════╣"
        )?;
        write!(f, "{}", self.energy)?;
        writeln!(
            f,
            "╠──────────────────────────────────────────────────────────────╣"
        )?;
        write!(f, "{}", self.thermal)?;
        writeln!(
            f,
            "╠──────────────────────────────────────────────────────────────╣"
        )?;
        write!(f, "{}", self.thermodynamics)?;
        writeln!(
            f,
            "╠──────────────────────────────────────────────────────────────╣"
        )?;
        write!(f, "{}", self.efficiency)?;
        writeln!(
            f,
            "╚══════════════════════════════════════════════════════════════╝"
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_chain() {
        let mut chain = DerivativeChain::new(100.0);
        chain.velocity = 10.0;
        chain.acceleration = 1.0;

        assert_eq!(chain.get(0), 100.0);
        assert_eq!(chain.get(1), 10.0);
        assert_eq!(chain.get(2), 1.0);
        assert_eq!(DerivativeChain::name(3), "jerk");
    }

    #[test]
    fn test_derivative_computation() {
        let computer = DerivativeComputer::new(3, 3);

        // Linear increase: y = t, so dy/dt = 1
        let samples: Vec<Sample> = (0..10)
            .map(|i| Sample {
                time_s: i as f64 * 0.1,
                energy_j: i as f64 * 0.1,
                temp_c: Some(50.0 + i as f64),
                ops: None,
                bytes: None,
            })
            .collect();

        let (energy_chain, temp_chain) = computer.compute_derivatives(&samples);

        // Energy velocity (power) should be ~1.0 W
        assert!((energy_chain.velocity - 1.0).abs() < 0.1);

        // Temp velocity should be ~10.0 °C/s
        assert!((temp_chain.velocity - 10.0).abs() < 1.0);

        // Acceleration should be ~0 for linear
        assert!(energy_chain.acceleration.abs() < 0.1);
    }

    #[test]
    fn test_efficiency_metrics() {
        let metrics = EfficiencyMetrics::compute(
            1.0,       // 1 Joule
            1.0,       // 1 second
            1_000_000, // 1M ops
            1_000_000, // 1MB
            1.0,       // 1 Watt
        );

        assert_eq!(metrics.ops_per_joule, 1_000_000.0);
        assert_eq!(metrics.bytes_per_joule, 1_000_000.0);
        assert_eq!(metrics.energy_delay_product_j_s, 1.0);
    }

    #[test]
    fn test_thermodynamic_coupling() {
        let energy = EnergyDerivatives {
            energy_j: 10.0,
            power_w: 100.0,
            power_rate_w_per_s: 10.0,
            ..Default::default()
        };

        let thermal = ThermalDerivatives {
            temp_c: 60.0,
            heating_rate_c_per_s: 0.5,
            ..Default::default()
        };

        let coupling = ThermodynamicCoupling::compute(&energy, &thermal);

        // Heat capacity: 100W / 0.5°C/s = 200 J/°C
        assert!((coupling.heat_capacity_j_per_c - 200.0).abs() < 0.1);
    }

    #[test]
    fn test_config_builder() {
        let config = KinematicConfig::default()
            .with_sample_rate_hz(500.0)
            .with_derivative_order(4)
            .with_smoothing(7);

        assert_eq!(config.sample_rate_hz, 500.0);
        assert_eq!(config.derivative_order, 4);
        assert_eq!(config.smoothing_window, 7);
    }
}
