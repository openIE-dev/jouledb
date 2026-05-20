//! Admittance Control — Force-input to motion-output controller, virtual
//! dynamics model, interaction stability analysis, force filtering, and
//! multi-DOF admittance regulation for safe human-robot interaction.
//!
//! Pure-Rust admittance controller using `f64` math; no external crates.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Admittance control errors.
#[derive(Debug, Clone, PartialEq)]
pub enum AdmittanceError {
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Velocity limit exceeded.
    VelocityLimit { velocity: f64, limit: f64 },
}

impl fmt::Display for AdmittanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::VelocityLimit { velocity, limit } => {
                write!(f, "velocity {velocity:.3} exceeds limit {limit:.3}")
            }
        }
    }
}

impl std::error::Error for AdmittanceError {}

// ── 1-DOF Admittance Controller ────────────────────────────────

/// Single-DOF admittance controller: maps force input to velocity output.
///
/// Virtual dynamics: M_d * x_ddot + B_d * x_dot + K_d * x = F_ext
///
/// Given measured force, the controller integrates the virtual dynamics
/// to produce a desired velocity (or position) command for the inner
/// position/velocity loop.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittanceController1D {
    /// Virtual inertia (kg).
    pub inertia: f64,
    /// Virtual damping (N·s/m).
    pub damping: f64,
    /// Virtual stiffness (N/m).
    pub stiffness: f64,
    /// Desired velocity output (m/s).
    pub velocity_cmd: f64,
    /// Desired position offset (m).
    pub position_cmd: f64,
    /// Internal acceleration.
    acceleration: f64,
    /// Maximum velocity limit (m/s).
    pub max_velocity: f64,
    /// Maximum position offset limit (m).
    pub max_position: f64,
}

impl AdmittanceController1D {
    /// Create a 1-DOF admittance controller.
    pub fn new(inertia: f64, damping: f64, stiffness: f64) -> Result<Self, AdmittanceError> {
        if inertia <= 0.0 {
            return Err(AdmittanceError::InvalidParameter(
                "inertia must be > 0".into(),
            ));
        }
        if damping < 0.0 {
            return Err(AdmittanceError::InvalidParameter(
                "damping must be >= 0".into(),
            ));
        }
        Ok(Self {
            inertia,
            damping,
            stiffness,
            velocity_cmd: 0.0,
            position_cmd: 0.0,
            acceleration: 0.0,
            max_velocity: f64::INFINITY,
            max_position: f64::INFINITY,
        })
    }

    /// Builder: set velocity limit.
    pub fn with_velocity_limit(mut self, limit: f64) -> Self {
        self.max_velocity = limit.abs();
        self
    }

    /// Builder: set position limit.
    pub fn with_position_limit(mut self, limit: f64) -> Self {
        self.max_position = limit.abs();
        self
    }

    /// Update the admittance controller with measured external force.
    ///
    /// Returns (velocity_cmd, position_cmd).
    pub fn update(&mut self, force_ext: f64, dt: f64) -> (f64, f64) {
        if dt <= 0.0 {
            return (self.velocity_cmd, self.position_cmd);
        }

        // Virtual dynamics: M*a + B*v + K*x = F_ext
        // a = (F_ext - B*v - K*x) / M
        self.acceleration = (force_ext
            - self.damping * self.velocity_cmd
            - self.stiffness * self.position_cmd)
            / self.inertia;

        self.velocity_cmd += self.acceleration * dt;
        self.velocity_cmd = self.velocity_cmd.clamp(-self.max_velocity, self.max_velocity);

        self.position_cmd += self.velocity_cmd * dt;
        self.position_cmd = self.position_cmd.clamp(-self.max_position, self.max_position);

        (self.velocity_cmd, self.position_cmd)
    }

    /// Steady-state displacement for a constant force: x_ss = F / K.
    pub fn steady_state_displacement(&self, force: f64) -> f64 {
        if self.stiffness.abs() < 1e-12 {
            return f64::INFINITY * force.signum();
        }
        force / self.stiffness
    }

    /// Bandwidth of the admittance (rad/s): ω_n = sqrt(K/M).
    pub fn bandwidth(&self) -> f64 {
        if self.stiffness < 0.0 {
            return 0.0;
        }
        (self.stiffness / self.inertia).sqrt()
    }

    /// Damping ratio.
    pub fn damping_ratio(&self) -> f64 {
        if self.stiffness <= 0.0 {
            return 0.0;
        }
        self.damping / (2.0 * (self.stiffness * self.inertia).sqrt())
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.velocity_cmd = 0.0;
        self.position_cmd = 0.0;
        self.acceleration = 0.0;
    }
}

impl fmt::Display for AdmittanceController1D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Admittance1D(M={:.2}, B={:.2}, K={:.2}, v_cmd={:.4})",
            self.inertia, self.damping, self.stiffness, self.velocity_cmd
        )
    }
}

// ── Multi-DOF Admittance Controller ────────────────────────────

/// Multi-DOF admittance controller with independent axis dynamics.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittanceControllerND {
    /// Number of DOFs.
    pub ndof: usize,
    /// Per-axis controllers.
    pub axes: Vec<AdmittanceController1D>,
}

impl AdmittanceControllerND {
    /// Create an N-DOF admittance controller with uniform parameters.
    pub fn new_uniform(
        ndof: usize,
        inertia: f64,
        damping: f64,
        stiffness: f64,
    ) -> Result<Self, AdmittanceError> {
        if ndof == 0 {
            return Err(AdmittanceError::InvalidParameter("ndof must be > 0".into()));
        }
        let axes: Result<Vec<_>, _> = (0..ndof)
            .map(|_| AdmittanceController1D::new(inertia, damping, stiffness))
            .collect();
        Ok(Self { ndof, axes: axes? })
    }

    /// Create from per-axis parameter vectors.
    pub fn new(
        inertias: &[f64],
        dampings: &[f64],
        stiffnesses: &[f64],
    ) -> Result<Self, AdmittanceError> {
        let n = inertias.len();
        if n == 0 {
            return Err(AdmittanceError::InvalidParameter("empty parameters".into()));
        }
        if dampings.len() != n || stiffnesses.len() != n {
            return Err(AdmittanceError::DimensionMismatch {
                expected: n,
                got: dampings.len().min(stiffnesses.len()),
            });
        }
        let axes: Result<Vec<_>, _> = (0..n)
            .map(|i| AdmittanceController1D::new(inertias[i], dampings[i], stiffnesses[i]))
            .collect();
        Ok(Self { ndof: n, axes: axes? })
    }

    /// Builder: set velocity limits for all axes.
    pub fn with_velocity_limits(mut self, limits: &[f64]) -> Self {
        for (i, ax) in self.axes.iter_mut().enumerate() {
            if i < limits.len() {
                ax.max_velocity = limits[i].abs();
            }
        }
        self
    }

    /// Update all axes with measured forces. Returns velocity commands.
    pub fn update(
        &mut self,
        forces: &[f64],
        dt: f64,
    ) -> Result<Vec<f64>, AdmittanceError> {
        if forces.len() != self.ndof {
            return Err(AdmittanceError::DimensionMismatch {
                expected: self.ndof,
                got: forces.len(),
            });
        }

        let velocities: Vec<f64> = self
            .axes
            .iter_mut()
            .zip(forces.iter())
            .map(|(ax, &f)| ax.update(f, dt).0)
            .collect();

        Ok(velocities)
    }

    /// Get position commands for all axes.
    pub fn position_commands(&self) -> Vec<f64> {
        self.axes.iter().map(|a| a.position_cmd).collect()
    }

    /// Get velocity commands for all axes.
    pub fn velocity_commands(&self) -> Vec<f64> {
        self.axes.iter().map(|a| a.velocity_cmd).collect()
    }

    /// Reset all axes.
    pub fn reset(&mut self) {
        for ax in &mut self.axes {
            ax.reset();
        }
    }
}

impl fmt::Display for AdmittanceControllerND {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AdmittanceND(ndof={})", self.ndof)
    }
}

// ── Force Filter ───────────────────────────────────────────────

/// Second-order Butterworth low-pass filter for force signals.
///
/// Implements the discrete-time transfer function derived from bilinear
/// transform of a continuous 2nd-order Butterworth filter.
#[derive(Debug, Clone, PartialEq)]
pub struct ForceFilter {
    /// Cutoff frequency (Hz).
    pub cutoff_hz: f64,
    /// Sample rate (Hz).
    pub sample_rate: f64,
    /// Filter coefficients [a1, a2, b0, b1, b2].
    coeffs: [f64; 5],
    /// Input history [x[n-1], x[n-2]].
    x_hist: [f64; 2],
    /// Output history [y[n-1], y[n-2]].
    y_hist: [f64; 2],
}

impl ForceFilter {
    /// Create a 2nd-order Butterworth filter.
    pub fn new(cutoff_hz: f64, sample_rate: f64) -> Result<Self, AdmittanceError> {
        if cutoff_hz <= 0.0 || sample_rate <= 0.0 {
            return Err(AdmittanceError::InvalidParameter(
                "frequencies must be > 0".into(),
            ));
        }
        if cutoff_hz >= sample_rate / 2.0 {
            return Err(AdmittanceError::InvalidParameter(
                "cutoff must be below Nyquist".into(),
            ));
        }

        let wc = (std::f64::consts::PI * cutoff_hz / sample_rate).tan();
        let wc2 = wc * wc;
        let sqrt2 = std::f64::consts::SQRT_2;
        let denom = 1.0 + sqrt2 * wc + wc2;

        let b0 = wc2 / denom;
        let b1 = 2.0 * wc2 / denom;
        let b2 = wc2 / denom;
        let a1 = 2.0 * (wc2 - 1.0) / denom;
        let a2 = (1.0 - sqrt2 * wc + wc2) / denom;

        Ok(Self {
            cutoff_hz,
            sample_rate,
            coeffs: [a1, a2, b0, b1, b2],
            x_hist: [0.0; 2],
            y_hist: [0.0; 2],
        })
    }

    /// Filter a single sample.
    pub fn filter(&mut self, input: f64) -> f64 {
        let [a1, a2, b0, b1, b2] = self.coeffs;

        let output = b0 * input + b1 * self.x_hist[0] + b2 * self.x_hist[1]
            - a1 * self.y_hist[0]
            - a2 * self.y_hist[1];

        self.x_hist[1] = self.x_hist[0];
        self.x_hist[0] = input;
        self.y_hist[1] = self.y_hist[0];
        self.y_hist[0] = output;

        output
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.x_hist = [0.0; 2];
        self.y_hist = [0.0; 2];
    }
}

impl fmt::Display for ForceFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ForceFilter(fc={:.1}Hz, fs={:.0}Hz)",
            self.cutoff_hz, self.sample_rate
        )
    }
}

// ── Interaction Stability Analyzer ─────────────────────────────

/// Analyzes interaction stability between admittance controller and environment.
///
/// Uses passivity-based criteria: the system is passive (and hence stable
/// with any passive environment) if the admittance is positive-real.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityAnalyzer {
    /// Energy balance: should remain non-negative for passivity.
    pub energy_balance: f64,
    /// Power history for monitoring.
    power_history: VecDeque<f64>,
    /// Window size.
    pub window_size: usize,
    /// Passivity violation counter.
    pub violation_count: u64,
    /// Total samples processed.
    pub sample_count: u64,
}

impl StabilityAnalyzer {
    /// Create a stability analyzer.
    pub fn new(window_size: usize) -> Self {
        Self {
            energy_balance: 0.0,
            power_history: VecDeque::with_capacity(window_size),
            window_size: window_size.max(1),
            violation_count: 0,
            sample_count: 0,
        }
    }

    /// Update with measured force and velocity.
    ///
    /// Power = force * velocity; positive power means energy flows into the
    /// controller (passive). Returns true if currently passive.
    pub fn update(&mut self, force: f64, velocity: f64, dt: f64) -> bool {
        let power = force * velocity;

        self.power_history.push_back(power);
        if self.power_history.len() > self.window_size {
            self.power_history.pop_front();
        }

        // Accumulate energy.
        self.energy_balance += power * dt;
        self.sample_count += 1;

        if self.energy_balance < -1e-6 {
            self.violation_count += 1;
            false
        } else {
            true
        }
    }

    /// Mean power over the window.
    pub fn mean_power(&self) -> f64 {
        if self.power_history.is_empty() {
            return 0.0;
        }
        self.power_history.iter().sum::<f64>() / self.power_history.len() as f64
    }

    /// Peak absolute power.
    pub fn peak_power(&self) -> f64 {
        self.power_history
            .iter()
            .map(|p| p.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Whether the system has been passive throughout.
    pub fn is_passive(&self) -> bool {
        self.violation_count == 0
    }

    /// Passivity index: ratio of non-violating samples.
    pub fn passivity_index(&self) -> f64 {
        if self.sample_count == 0 {
            return 1.0;
        }
        1.0 - self.violation_count as f64 / self.sample_count as f64
    }

    /// Reset analyzer.
    pub fn reset(&mut self) {
        self.energy_balance = 0.0;
        self.power_history.clear();
        self.violation_count = 0;
        self.sample_count = 0;
    }
}

impl fmt::Display for StabilityAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StabilityAnalyzer(E={:.4}, passive={}, index={:.3})",
            self.energy_balance,
            self.is_passive(),
            self.passivity_index()
        )
    }
}

// ── Dead-Zone Compensator ──────────────────────────────────────

/// Dead-zone compensator for force sensors: suppresses small force noise
/// while preserving larger signals.
#[derive(Debug, Clone, PartialEq)]
pub struct DeadZone {
    /// Threshold below which force is zeroed.
    pub threshold: f64,
    /// Smooth transition width (0 = hard cutoff).
    pub transition: f64,
}

impl DeadZone {
    /// Create a dead zone with hard threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.abs(),
            transition: 0.0,
        }
    }

    /// Builder: set smooth transition width.
    pub fn with_transition(mut self, width: f64) -> Self {
        self.transition = width.abs();
        self
    }

    /// Apply the dead zone to a force value.
    pub fn apply(&self, force: f64) -> f64 {
        let mag = force.abs();
        if self.transition < 1e-12 {
            // Hard cutoff.
            if mag <= self.threshold {
                0.0
            } else {
                force
            }
        } else {
            // Smooth transition.
            if mag <= self.threshold {
                0.0
            } else if mag <= self.threshold + self.transition {
                let t = (mag - self.threshold) / self.transition;
                // Cubic smoothstep.
                let smooth = t * t * (3.0 - 2.0 * t);
                force.signum() * mag * smooth
            } else {
                force
            }
        }
    }

    /// Apply dead zone to a 6D wrench.
    pub fn apply_wrench(&self, force: &[f64; 6]) -> [f64; 6] {
        [
            self.apply(force[0]),
            self.apply(force[1]),
            self.apply(force[2]),
            self.apply(force[3]),
            self.apply(force[4]),
            self.apply(force[5]),
        ]
    }
}

impl fmt::Display for DeadZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeadZone(thresh={:.3}, transition={:.3})",
            self.threshold, self.transition
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admittance_1d_creation() {
        let c = AdmittanceController1D::new(1.0, 10.0, 50.0);
        assert!(c.is_ok());
    }

    #[test]
    fn test_admittance_1d_invalid_inertia() {
        assert!(AdmittanceController1D::new(0.0, 10.0, 50.0).is_err());
    }

    #[test]
    fn test_admittance_1d_responds_to_force() {
        let mut c = AdmittanceController1D::new(1.0, 10.0, 0.0).unwrap();
        let (v, _) = c.update(10.0, 0.01);
        // a = 10/1 = 10 m/s², v = 10*0.01 = 0.1 m/s
        assert!((v - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_admittance_1d_damping_resists() {
        let mut c = AdmittanceController1D::new(1.0, 100.0, 0.0).unwrap();
        // Apply force, let velocity build up.
        for _ in 0..1000 {
            c.update(10.0, 0.001);
        }
        // Steady-state: B*v = F → v = F/B = 10/100 = 0.1
        assert!((c.velocity_cmd - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_admittance_1d_stiffness_limits_displacement() {
        let mut c = AdmittanceController1D::new(1.0, 20.0, 100.0).unwrap();
        // Apply constant force.
        for _ in 0..20_000 {
            c.update(10.0, 0.001);
        }
        // Steady-state: K*x = F → x = F/K = 10/100 = 0.1
        assert!((c.position_cmd - 0.1).abs() < 0.02, "x={}", c.position_cmd);
    }

    #[test]
    fn test_admittance_1d_velocity_limit() {
        let mut c = AdmittanceController1D::new(1.0, 0.1, 0.0)
            .unwrap()
            .with_velocity_limit(0.5);
        for _ in 0..1000 {
            c.update(100.0, 0.01);
        }
        assert!(c.velocity_cmd <= 0.5 + 1e-9);
    }

    #[test]
    fn test_admittance_1d_reset() {
        let mut c = AdmittanceController1D::new(1.0, 10.0, 50.0).unwrap();
        c.update(10.0, 0.01);
        c.reset();
        assert!(c.velocity_cmd.abs() < 1e-15);
        assert!(c.position_cmd.abs() < 1e-15);
    }

    #[test]
    fn test_admittance_nd_uniform() {
        let c = AdmittanceControllerND::new_uniform(3, 1.0, 10.0, 50.0);
        assert!(c.is_ok());
        assert_eq!(c.unwrap().ndof, 3);
    }

    #[test]
    fn test_admittance_nd_update() {
        let mut c = AdmittanceControllerND::new_uniform(2, 1.0, 10.0, 0.0).unwrap();
        let vels = c.update(&[5.0, -3.0], 0.01).unwrap();
        assert_eq!(vels.len(), 2);
        assert!(vels[0] > 0.0);
        assert!(vels[1] < 0.0);
    }

    #[test]
    fn test_admittance_nd_dim_mismatch() {
        let mut c = AdmittanceControllerND::new_uniform(3, 1.0, 10.0, 0.0).unwrap();
        assert!(c.update(&[1.0, 2.0], 0.01).is_err());
    }

    #[test]
    fn test_force_filter_creation() {
        let f = ForceFilter::new(10.0, 1000.0);
        assert!(f.is_ok());
    }

    #[test]
    fn test_force_filter_nyquist() {
        let f = ForceFilter::new(600.0, 1000.0);
        assert!(f.is_err());
    }

    #[test]
    fn test_force_filter_attenuates_high_freq() {
        let mut filt = ForceFilter::new(10.0, 1000.0).unwrap();
        // Feed a high-frequency signal (200 Hz).
        let mut max_output = 0.0_f64;
        for i in 0..1000 {
            let t = i as f64 / 1000.0;
            let input = (2.0 * std::f64::consts::PI * 200.0 * t).sin();
            let out = filt.filter(input);
            if i > 100 {
                // Skip transient.
                max_output = max_output.max(out.abs());
            }
        }
        assert!(max_output < 0.1, "high-freq should be attenuated: {max_output}");
    }

    #[test]
    fn test_force_filter_passes_low_freq() {
        let mut filt = ForceFilter::new(50.0, 1000.0).unwrap();
        // Feed DC (constant force).
        for _ in 0..500 {
            filt.filter(1.0);
        }
        let out = filt.filter(1.0);
        assert!((out - 1.0).abs() < 0.01, "DC should pass through: {out}");
    }

    #[test]
    fn test_stability_passive() {
        let mut sa = StabilityAnalyzer::new(100);
        // Positive power (energy into controller).
        for _ in 0..100 {
            sa.update(1.0, 1.0, 0.01);
        }
        assert!(sa.is_passive());
        assert!((sa.passivity_index() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stability_violation() {
        let mut sa = StabilityAnalyzer::new(100);
        // Negative power (energy out of controller).
        sa.update(-10.0, 1.0, 0.1);
        assert!(!sa.is_passive());
    }

    #[test]
    fn test_dead_zone_hard() {
        let dz = DeadZone::new(1.0);
        assert!((dz.apply(0.5)).abs() < 1e-15);
        assert!((dz.apply(2.0) - 2.0).abs() < 1e-15);
        assert!((dz.apply(-0.5)).abs() < 1e-15);
        assert!((dz.apply(-2.0) - (-2.0)).abs() < 1e-15);
    }

    #[test]
    fn test_dead_zone_smooth() {
        let dz = DeadZone::new(1.0).with_transition(0.5);
        assert!((dz.apply(0.5)).abs() < 1e-15);
        let v = dz.apply(1.25);
        assert!(v > 0.0 && v < 1.25, "smooth transition: {v}");
        let v2 = dz.apply(2.0);
        assert!((v2 - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_dead_zone_wrench() {
        let dz = DeadZone::new(0.5);
        let result = dz.apply_wrench(&[0.1, -0.3, 1.0, 0.0, -2.0, 0.4]);
        assert!(result[0].abs() < 1e-15);
        assert!(result[1].abs() < 1e-15);
        assert!((result[2] - 1.0).abs() < 1e-15);
        assert!((result[4] - (-2.0)).abs() < 1e-15);
    }

    #[test]
    fn test_display_admittance_1d() {
        let c = AdmittanceController1D::new(1.0, 10.0, 50.0).unwrap();
        let s = format!("{c}");
        assert!(s.contains("Admittance1D"));
    }

    #[test]
    fn test_display_stability() {
        let sa = StabilityAnalyzer::new(10);
        let s = format!("{sa}");
        assert!(s.contains("StabilityAnalyzer"));
    }

    #[test]
    fn test_display_dead_zone() {
        let dz = DeadZone::new(0.5);
        let s = format!("{dz}");
        assert!(s.contains("DeadZone"));
    }
}
