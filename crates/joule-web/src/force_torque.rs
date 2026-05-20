//! Force-Torque Sensing — 6-axis F/T data processing (Fx, Fy, Fz, Tx, Ty, Tz),
//! gravity compensation, contact detection thresholding, wrench-space analysis,
//! and low-pass filtering for noisy sensor data.
//!
//! Pure-Rust force-torque processing using `f64` math; no external crates.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Force-torque processing errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FtError {
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Calibration required.
    CalibrationRequired,
}

impl fmt::Display for FtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::CalibrationRequired => write!(f, "sensor calibration required"),
        }
    }
}

impl std::error::Error for FtError {}

// ── Wrench (6D Force-Torque Vector) ────────────────────────────

/// A 6-axis wrench: (Fx, Fy, Fz, Tx, Ty, Tz).
///
/// Forces in Newtons, torques in Newton-meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wrench {
    /// Force along X axis (N).
    pub fx: f64,
    /// Force along Y axis (N).
    pub fy: f64,
    /// Force along Z axis (N).
    pub fz: f64,
    /// Torque about X axis (N·m).
    pub tx: f64,
    /// Torque about Y axis (N·m).
    pub ty: f64,
    /// Torque about Z axis (N·m).
    pub tz: f64,
}

impl Wrench {
    /// Zero wrench.
    pub fn zero() -> Self {
        Self {
            fx: 0.0, fy: 0.0, fz: 0.0,
            tx: 0.0, ty: 0.0, tz: 0.0,
        }
    }

    /// Create from force and torque vectors.
    pub fn new(fx: f64, fy: f64, fz: f64, tx: f64, ty: f64, tz: f64) -> Self {
        Self { fx, fy, fz, tx, ty, tz }
    }

    /// Create from arrays.
    pub fn from_arrays(force: [f64; 3], torque: [f64; 3]) -> Self {
        Self {
            fx: force[0], fy: force[1], fz: force[2],
            tx: torque[0], ty: torque[1], tz: torque[2],
        }
    }

    /// Force magnitude (N).
    pub fn force_magnitude(&self) -> f64 {
        (self.fx * self.fx + self.fy * self.fy + self.fz * self.fz).sqrt()
    }

    /// Torque magnitude (N·m).
    pub fn torque_magnitude(&self) -> f64 {
        (self.tx * self.tx + self.ty * self.ty + self.tz * self.tz).sqrt()
    }

    /// Force vector as array.
    pub fn force_vec(&self) -> [f64; 3] {
        [self.fx, self.fy, self.fz]
    }

    /// Torque vector as array.
    pub fn torque_vec(&self) -> [f64; 3] {
        [self.tx, self.ty, self.tz]
    }

    /// Full 6D vector.
    pub fn as_array(&self) -> [f64; 6] {
        [self.fx, self.fy, self.fz, self.tx, self.ty, self.tz]
    }

    /// Wrench addition.
    pub fn add(&self, other: &Wrench) -> Wrench {
        Wrench {
            fx: self.fx + other.fx,
            fy: self.fy + other.fy,
            fz: self.fz + other.fz,
            tx: self.tx + other.tx,
            ty: self.ty + other.ty,
            tz: self.tz + other.tz,
        }
    }

    /// Wrench subtraction.
    pub fn sub(&self, other: &Wrench) -> Wrench {
        Wrench {
            fx: self.fx - other.fx,
            fy: self.fy - other.fy,
            fz: self.fz - other.fz,
            tx: self.tx - other.tx,
            ty: self.ty - other.ty,
            tz: self.tz - other.tz,
        }
    }

    /// Scale wrench by a scalar.
    pub fn scale(&self, s: f64) -> Wrench {
        Wrench {
            fx: self.fx * s, fy: self.fy * s, fz: self.fz * s,
            tx: self.tx * s, ty: self.ty * s, tz: self.tz * s,
        }
    }

    /// Dot product in wrench space (force·force + torque·torque).
    pub fn dot(&self, other: &Wrench) -> f64 {
        self.fx * other.fx + self.fy * other.fy + self.fz * other.fz
            + self.tx * other.tx + self.ty * other.ty + self.tz * other.tz
    }

    /// L2 norm in 6D wrench space.
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }
}

impl fmt::Display for Wrench {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Wrench(F=[{:.3},{:.3},{:.3}]N, T=[{:.4},{:.4},{:.4}]Nm)",
            self.fx, self.fy, self.fz, self.tx, self.ty, self.tz
        )
    }
}

// ── Gravity Compensation ───────────────────────────────────────

/// Gravity compensation for a tool/payload mounted at the F/T sensor.
///
/// Removes the gravity wrench due to known tool mass and center of gravity.
#[derive(Debug, Clone, PartialEq)]
pub struct GravityCompensator {
    /// Tool mass (kg).
    pub mass: f64,
    /// Center of gravity in sensor frame [x, y, z] (m).
    pub cog: [f64; 3],
    /// Gravity vector in sensor frame [gx, gy, gz] (m/s²).
    /// Updated by orientation input.
    pub gravity_vec: [f64; 3],
}

impl GravityCompensator {
    /// Create a gravity compensator.
    pub fn new(mass: f64, cog: [f64; 3]) -> Result<Self, FtError> {
        if mass < 0.0 {
            return Err(FtError::InvalidParameter("mass must be >= 0".into()));
        }
        Ok(Self {
            mass,
            cog,
            gravity_vec: [0.0, 0.0, -9.81],
        })
    }

    /// Builder: set initial gravity vector.
    pub fn with_gravity(mut self, grav: [f64; 3]) -> Self {
        self.gravity_vec = grav;
        self
    }

    /// Update the gravity vector based on sensor orientation.
    ///
    /// `rotation` is a 3x3 rotation matrix (row-major) from world to sensor frame.
    pub fn update_orientation(&mut self, rotation: &[f64; 9]) {
        // g_sensor = R * g_world, where g_world = [0, 0, -9.81].
        let gw = [0.0, 0.0, -9.81];
        self.gravity_vec = [
            rotation[0] * gw[0] + rotation[1] * gw[1] + rotation[2] * gw[2],
            rotation[3] * gw[0] + rotation[4] * gw[1] + rotation[5] * gw[2],
            rotation[6] * gw[0] + rotation[7] * gw[1] + rotation[8] * gw[2],
        ];
    }

    /// Compute the gravity wrench at the sensor origin.
    pub fn gravity_wrench(&self) -> Wrench {
        let f = [
            self.mass * self.gravity_vec[0],
            self.mass * self.gravity_vec[1],
            self.mass * self.gravity_vec[2],
        ];
        // Torque = cog × force.
        let t = [
            self.cog[1] * f[2] - self.cog[2] * f[1],
            self.cog[2] * f[0] - self.cog[0] * f[2],
            self.cog[0] * f[1] - self.cog[1] * f[0],
        ];
        Wrench::from_arrays(f, t)
    }

    /// Compensate a raw wrench reading by subtracting gravity wrench.
    pub fn compensate(&self, raw: &Wrench) -> Wrench {
        raw.sub(&self.gravity_wrench())
    }
}

impl fmt::Display for GravityCompensator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GravComp(mass={:.3}kg, cog=[{:.4},{:.4},{:.4}])",
            self.mass, self.cog[0], self.cog[1], self.cog[2]
        )
    }
}

// ── Contact Detector ───────────────────────────────────────────

/// Threshold-based contact detection from F/T data.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactDetector {
    /// Force threshold for contact (N).
    pub force_threshold: f64,
    /// Torque threshold for contact (N·m).
    pub torque_threshold: f64,
    /// Consecutive samples above threshold to confirm contact.
    pub confirmation_count: u32,
    /// Current consecutive count.
    counter: u32,
    /// Contact state.
    pub in_contact: bool,
    /// Direction of contact force (unit vector).
    pub contact_direction: [f64; 3],
}

impl ContactDetector {
    /// Create a contact detector.
    pub fn new(force_threshold: f64, torque_threshold: f64) -> Result<Self, FtError> {
        if force_threshold <= 0.0 {
            return Err(FtError::InvalidParameter(
                "force threshold must be > 0".into(),
            ));
        }
        Ok(Self {
            force_threshold,
            torque_threshold,
            confirmation_count: 3,
            counter: 0,
            in_contact: false,
            contact_direction: [0.0; 3],
        })
    }

    /// Builder: set confirmation count.
    pub fn with_confirmation(mut self, count: u32) -> Self {
        self.confirmation_count = count.max(1);
        self
    }

    /// Process a wrench sample; returns true if contact is detected.
    pub fn update(&mut self, wrench: &Wrench) -> bool {
        let f_mag = wrench.force_magnitude();
        let t_mag = wrench.torque_magnitude();

        let above_threshold =
            f_mag > self.force_threshold || t_mag > self.torque_threshold;

        if above_threshold {
            self.counter = self.counter.saturating_add(1);
            if self.counter >= self.confirmation_count {
                self.in_contact = true;
                // Update contact direction.
                if f_mag > 1e-9 {
                    self.contact_direction = [
                        wrench.fx / f_mag,
                        wrench.fy / f_mag,
                        wrench.fz / f_mag,
                    ];
                }
            }
        } else {
            self.counter = 0;
            self.in_contact = false;
        }

        self.in_contact
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.counter = 0;
        self.in_contact = false;
        self.contact_direction = [0.0; 3];
    }
}

impl fmt::Display for ContactDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ContactDet(thresh={:.2}N, contact={}, dir=[{:.2},{:.2},{:.2}])",
            self.force_threshold,
            self.in_contact,
            self.contact_direction[0],
            self.contact_direction[1],
            self.contact_direction[2]
        )
    }
}

// ── Low-Pass Filter ────────────────────────────────────────────

/// First-order low-pass filter for 6-axis wrench data.
///
/// Implements exponential moving average: y[k] = α·y[k-1] + (1-α)·x[k].
#[derive(Debug, Clone, PartialEq)]
pub struct WrenchFilter {
    /// Filter coefficient [0, 1); higher = more smoothing.
    pub alpha: f64,
    /// Filtered output.
    pub filtered: Wrench,
    /// Whether the filter has been initialized.
    initialized: bool,
}

impl WrenchFilter {
    /// Create a wrench filter from cutoff frequency and sample rate.
    pub fn from_cutoff(cutoff_hz: f64, sample_rate_hz: f64) -> Result<Self, FtError> {
        if cutoff_hz <= 0.0 || sample_rate_hz <= 0.0 {
            return Err(FtError::InvalidParameter(
                "frequencies must be > 0".into(),
            ));
        }
        let dt = 1.0 / sample_rate_hz;
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
        let alpha = rc / (rc + dt);
        Ok(Self {
            alpha,
            filtered: Wrench::zero(),
            initialized: false,
        })
    }

    /// Create from raw alpha coefficient.
    pub fn from_alpha(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 0.9999),
            filtered: Wrench::zero(),
            initialized: false,
        }
    }

    /// Filter a wrench sample.
    pub fn filter(&mut self, raw: &Wrench) -> Wrench {
        if !self.initialized {
            self.filtered = *raw;
            self.initialized = true;
            return self.filtered;
        }
        self.filtered = Wrench {
            fx: self.alpha * self.filtered.fx + (1.0 - self.alpha) * raw.fx,
            fy: self.alpha * self.filtered.fy + (1.0 - self.alpha) * raw.fy,
            fz: self.alpha * self.filtered.fz + (1.0 - self.alpha) * raw.fz,
            tx: self.alpha * self.filtered.tx + (1.0 - self.alpha) * raw.tx,
            ty: self.alpha * self.filtered.ty + (1.0 - self.alpha) * raw.ty,
            tz: self.alpha * self.filtered.tz + (1.0 - self.alpha) * raw.tz,
        };
        self.filtered
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.filtered = Wrench::zero();
        self.initialized = false;
    }
}

impl fmt::Display for WrenchFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WrenchFilter(α={:.4})", self.alpha)
    }
}

// ── Wrench Space Analysis ──────────────────────────────────────

/// Wrench-space statistics over a sliding window.
#[derive(Debug, Clone)]
pub struct WrenchAnalyzer {
    /// Sliding window of wrench samples.
    window: VecDeque<Wrench>,
    /// Window size.
    pub window_size: usize,
    /// Running sum for mean.
    sum: Wrench,
}

impl WrenchAnalyzer {
    /// Create a wrench analyzer.
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size: window_size.max(1),
            sum: Wrench::zero(),
        }
    }

    /// Push a new sample.
    pub fn push(&mut self, w: Wrench) {
        if self.window.len() >= self.window_size {
            if let Some(old) = self.window.pop_front() {
                self.sum = self.sum.sub(&old);
            }
        }
        self.sum = self.sum.add(&w);
        self.window.push_back(w);
    }

    /// Mean wrench over the window.
    pub fn mean(&self) -> Wrench {
        if self.window.is_empty() {
            return Wrench::zero();
        }
        self.sum.scale(1.0 / self.window.len() as f64)
    }

    /// Variance of force magnitude over the window.
    pub fn force_variance(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let n = self.window.len() as f64;
        let mean_mag = self.window.iter().map(|w| w.force_magnitude()).sum::<f64>() / n;
        self.window
            .iter()
            .map(|w| {
                let d = w.force_magnitude() - mean_mag;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0)
    }

    /// Peak force magnitude in the window.
    pub fn peak_force(&self) -> f64 {
        self.window
            .iter()
            .map(|w| w.force_magnitude())
            .fold(0.0_f64, f64::max)
    }

    /// Peak torque magnitude in the window.
    pub fn peak_torque(&self) -> f64 {
        self.window
            .iter()
            .map(|w| w.torque_magnitude())
            .fold(0.0_f64, f64::max)
    }

    /// Number of samples in the window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Whether the analyzer has no samples.
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.window.clear();
        self.sum = Wrench::zero();
    }
}

impl fmt::Display for WrenchAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WrenchAnalyzer(n={}/{}, peak_F={:.2}N, peak_T={:.3}Nm)",
            self.window.len(),
            self.window_size,
            self.peak_force(),
            self.peak_torque()
        )
    }
}

// ── Sensor Bias Calibration ────────────────────────────────────

/// Bias calibration: accumulates samples at rest to compute sensor offset.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasCalibrator {
    /// Accumulated sum.
    sum: Wrench,
    /// Sample count.
    count: u64,
    /// Calibrated bias.
    pub bias: Option<Wrench>,
    /// Minimum samples required.
    pub min_samples: u64,
}

impl BiasCalibrator {
    /// Create a bias calibrator.
    pub fn new(min_samples: u64) -> Self {
        Self {
            sum: Wrench::zero(),
            count: 0,
            bias: None,
            min_samples: min_samples.max(1),
        }
    }

    /// Add a calibration sample.
    pub fn add_sample(&mut self, w: &Wrench) {
        self.sum = self.sum.add(w);
        self.count += 1;
    }

    /// Finalize calibration.
    pub fn finalize(&mut self) -> Result<Wrench, FtError> {
        if self.count < self.min_samples {
            return Err(FtError::CalibrationRequired);
        }
        let bias = self.sum.scale(1.0 / self.count as f64);
        self.bias = Some(bias);
        Ok(bias)
    }

    /// Apply bias correction to a raw reading.
    pub fn correct(&self, raw: &Wrench) -> Result<Wrench, FtError> {
        match &self.bias {
            Some(b) => Ok(raw.sub(b)),
            None => Err(FtError::CalibrationRequired),
        }
    }

    /// Reset calibration.
    pub fn reset(&mut self) {
        self.sum = Wrench::zero();
        self.count = 0;
        self.bias = None;
    }
}

impl fmt::Display for BiasCalibrator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.bias {
            Some(b) => write!(f, "BiasCal(done, bias_F={:.4}N)", b.force_magnitude()),
            None => write!(f, "BiasCal(pending, n={})", self.count),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrench_zero() {
        let w = Wrench::zero();
        assert!((w.force_magnitude()).abs() < 1e-15);
        assert!((w.torque_magnitude()).abs() < 1e-15);
    }

    #[test]
    fn test_wrench_force_magnitude() {
        let w = Wrench::new(3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
        assert!((w.force_magnitude() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_wrench_torque_magnitude() {
        let w = Wrench::new(0.0, 0.0, 0.0, 1.0, 2.0, 2.0);
        assert!((w.torque_magnitude() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_wrench_add_sub() {
        let a = Wrench::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let b = Wrench::new(4.0, 5.0, 6.0, 0.4, 0.5, 0.6);
        let c = a.add(&b);
        assert!((c.fx - 5.0).abs() < 1e-12);
        let d = c.sub(&b);
        assert!((d.fx - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_wrench_scale() {
        let w = Wrench::new(2.0, 4.0, 6.0, 1.0, 2.0, 3.0);
        let s = w.scale(0.5);
        assert!((s.fx - 1.0).abs() < 1e-12);
        assert!((s.tz - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_wrench_dot() {
        let a = Wrench::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = Wrench::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((a.dot(&b) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_wrench_norm() {
        let w = Wrench::new(3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
        assert!((w.norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_gravity_compensator_downward() {
        let gc = GravityCompensator::new(1.0, [0.0, 0.0, 0.1]).unwrap();
        let gw = gc.gravity_wrench();
        // Force = mass * g = 1.0 * [0, 0, -9.81] = [0, 0, -9.81].
        assert!((gw.fz - (-9.81)).abs() < 1e-6);
    }

    #[test]
    fn test_gravity_compensator_torque() {
        let gc = GravityCompensator::new(1.0, [0.1, 0.0, 0.0]).unwrap();
        let gw = gc.gravity_wrench();
        // cog × F = [0.1,0,0] × [0,0,-9.81] = [0*(-9.81)-0*0, 0*0-0.1*(-9.81), 0.1*0-0*0]
        //         = [0, 0.981, 0]
        assert!((gw.ty - 0.981).abs() < 1e-6);
    }

    #[test]
    fn test_gravity_compensate_cancels() {
        let gc = GravityCompensator::new(2.0, [0.0, 0.0, 0.05]).unwrap();
        let gravity = gc.gravity_wrench();
        let compensated = gc.compensate(&gravity);
        assert!(compensated.force_magnitude() < 1e-9);
    }

    #[test]
    fn test_contact_detector_below_threshold() {
        let mut cd = ContactDetector::new(5.0, 1.0).unwrap();
        let w = Wrench::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        assert!(!cd.update(&w));
    }

    #[test]
    fn test_contact_detector_above_threshold() {
        let mut cd = ContactDetector::new(5.0, 1.0).unwrap()
            .with_confirmation(1);
        let w = Wrench::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(cd.update(&w));
    }

    #[test]
    fn test_contact_detector_confirmation() {
        let mut cd = ContactDetector::new(5.0, 1.0).unwrap()
            .with_confirmation(3);
        let w = Wrench::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(!cd.update(&w)); // count=1
        assert!(!cd.update(&w)); // count=2
        assert!(cd.update(&w));  // count=3
    }

    #[test]
    fn test_contact_direction() {
        let mut cd = ContactDetector::new(1.0, 10.0).unwrap()
            .with_confirmation(1);
        let w = Wrench::new(0.0, 0.0, -5.0, 0.0, 0.0, 0.0);
        cd.update(&w);
        assert!((cd.contact_direction[2] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_wrench_filter_passthrough() {
        let mut filt = WrenchFilter::from_alpha(0.0); // no smoothing
        let w = Wrench::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let out = filt.filter(&w);
        assert!((out.fx - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_wrench_filter_smoothing() {
        let mut filt = WrenchFilter::from_alpha(0.9);
        let w1 = Wrench::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        filt.filter(&w1); // initialization
        let w2 = Wrench::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let out = filt.filter(&w2);
        // 0.9*10 + 0.1*0 = 9.0
        assert!((out.fx - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_wrench_analyzer_mean() {
        let mut wa = WrenchAnalyzer::new(10);
        wa.push(Wrench::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        wa.push(Wrench::new(4.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        let mean = wa.mean();
        assert!((mean.fx - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_wrench_analyzer_peak() {
        let mut wa = WrenchAnalyzer::new(10);
        wa.push(Wrench::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        wa.push(Wrench::new(5.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        wa.push(Wrench::new(3.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        assert!((wa.peak_force() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_bias_calibrator() {
        let mut cal = BiasCalibrator::new(3);
        cal.add_sample(&Wrench::new(0.1, 0.2, 0.3, 0.0, 0.0, 0.0));
        cal.add_sample(&Wrench::new(0.1, 0.2, 0.3, 0.0, 0.0, 0.0));
        assert!(cal.finalize().is_err()); // only 2 samples
        cal.add_sample(&Wrench::new(0.1, 0.2, 0.3, 0.0, 0.0, 0.0));
        let bias = cal.finalize().unwrap();
        assert!((bias.fx - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_bias_correction() {
        let mut cal = BiasCalibrator::new(1);
        cal.add_sample(&Wrench::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.0));
        cal.finalize().unwrap();
        let raw = Wrench::new(5.5, 0.0, 0.0, 0.0, 0.0, 0.0);
        let corrected = cal.correct(&raw).unwrap();
        assert!((corrected.fx - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_display_wrench() {
        let w = Wrench::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let s = format!("{w}");
        assert!(s.contains("Wrench"));
    }

    #[test]
    fn test_display_contact_detector() {
        let cd = ContactDetector::new(5.0, 1.0).unwrap();
        let s = format!("{cd}");
        assert!(s.contains("ContactDet"));
    }
}
