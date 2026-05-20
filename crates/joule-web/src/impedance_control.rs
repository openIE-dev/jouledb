//! Impedance Control — Virtual spring-damper-inertia model, desired impedance
//! specification, force/position balance, compliance shaping, and multi-DOF
//! impedance regulation for robot interaction control.
//!
//! Pure-Rust impedance controller using `f64` math; no external crates.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Impedance control errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ImpedanceError {
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Instability detected.
    Instability(String),
}

impl fmt::Display for ImpedanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::Instability(msg) => write!(f, "instability: {msg}"),
        }
    }
}

impl std::error::Error for ImpedanceError {}

// ── 1-DOF Impedance Model ──────────────────────────────────────

/// Single-DOF virtual spring-damper-inertia impedance model.
///
/// The desired impedance relation:
///   M_d * x_ddot + B_d * x_dot + K_d * (x - x_eq) = F_ext
///
/// where F_ext is the external force, and the controller computes the
/// required actuator force to achieve this impedance behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpedanceModel1D {
    /// Desired virtual inertia (kg).
    pub inertia: f64,
    /// Desired virtual damping (N·s/m).
    pub damping: f64,
    /// Desired virtual stiffness (N/m).
    pub stiffness: f64,
    /// Equilibrium (rest) position.
    pub x_eq: f64,
    /// Current position.
    pub position: f64,
    /// Current velocity.
    pub velocity: f64,
    /// Current acceleration.
    pub acceleration: f64,
}

impl ImpedanceModel1D {
    /// Create a 1-DOF impedance model.
    pub fn new(inertia: f64, damping: f64, stiffness: f64) -> Result<Self, ImpedanceError> {
        if inertia <= 0.0 {
            return Err(ImpedanceError::InvalidParameter(
                "inertia must be > 0".into(),
            ));
        }
        if damping < 0.0 {
            return Err(ImpedanceError::InvalidParameter(
                "damping must be >= 0".into(),
            ));
        }
        if stiffness < 0.0 {
            return Err(ImpedanceError::InvalidParameter(
                "stiffness must be >= 0".into(),
            ));
        }
        Ok(Self {
            inertia,
            damping,
            stiffness,
            x_eq: 0.0,
            position: 0.0,
            velocity: 0.0,
            acceleration: 0.0,
        })
    }

    /// Builder: set equilibrium position.
    pub fn with_equilibrium(mut self, x_eq: f64) -> Self {
        self.x_eq = x_eq;
        self
    }

    /// Builder: set initial state.
    pub fn with_initial_state(mut self, pos: f64, vel: f64) -> Self {
        self.position = pos;
        self.velocity = vel;
        self
    }

    /// Compute the force required to achieve desired impedance behavior,
    /// given measured position, velocity, and external force.
    ///
    /// Returns the actuator force command.
    pub fn compute_force(
        &self,
        measured_pos: f64,
        measured_vel: f64,
        external_force: f64,
    ) -> f64 {
        // Desired behavior: F_ext = M_d*a + B_d*v + K_d*(x - x_eq)
        // Rearranged: a_desired = (F_ext - B_d*v - K_d*(x-x_eq)) / M_d
        // Actuator force = actual robot dynamics cancel + desired impedance force
        //
        // In impedance control, the actuator force is:
        // F_act = K_d*(x_eq - x) + B_d*(0 - v) + F_ext
        // which creates the illusion of a spring-damper to the environment.

        let spring_force = self.stiffness * (self.x_eq - measured_pos);
        let damping_force = -self.damping * measured_vel;

        spring_force + damping_force + external_force
    }

    /// Simulate the impedance model forward by dt given external force.
    /// Updates internal state.
    pub fn step(&mut self, external_force: f64, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        // M_d * a = F_ext - B_d * v - K_d * (x - x_eq)
        self.acceleration = (external_force
            - self.damping * self.velocity
            - self.stiffness * (self.position - self.x_eq))
            / self.inertia;

        self.velocity += self.acceleration * dt;
        self.position += self.velocity * dt;
    }

    /// Natural frequency (rad/s).
    pub fn natural_frequency(&self) -> f64 {
        (self.stiffness / self.inertia).sqrt()
    }

    /// Damping ratio (dimensionless).
    pub fn damping_ratio(&self) -> f64 {
        self.damping / (2.0 * (self.stiffness * self.inertia).sqrt())
    }

    /// Whether the system is critically or over-damped.
    pub fn is_stable(&self) -> bool {
        self.damping_ratio() >= 0.0 && self.stiffness >= 0.0
    }

    /// Reset state to equilibrium.
    pub fn reset(&mut self) {
        self.position = self.x_eq;
        self.velocity = 0.0;
        self.acceleration = 0.0;
    }
}

impl fmt::Display for ImpedanceModel1D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Impedance1D(M={:.2}, B={:.2}, K={:.2}, ζ={:.3})",
            self.inertia,
            self.damping,
            self.stiffness,
            self.damping_ratio()
        )
    }
}

// ── Multi-DOF Impedance Model ──────────────────────────────────

/// Multi-DOF impedance model with diagonal inertia, damping, and stiffness.
///
/// Each DOF has independent impedance parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpedanceModelND {
    /// Number of DOFs.
    pub ndof: usize,
    /// Desired inertias (one per DOF).
    pub inertias: Vec<f64>,
    /// Desired damping coefficients.
    pub dampings: Vec<f64>,
    /// Desired stiffness coefficients.
    pub stiffnesses: Vec<f64>,
    /// Equilibrium positions.
    pub x_eq: Vec<f64>,
    /// Current positions.
    pub positions: Vec<f64>,
    /// Current velocities.
    pub velocities: Vec<f64>,
}

impl ImpedanceModelND {
    /// Create an N-DOF impedance model with uniform parameters.
    pub fn new_uniform(
        ndof: usize,
        inertia: f64,
        damping: f64,
        stiffness: f64,
    ) -> Result<Self, ImpedanceError> {
        if ndof == 0 {
            return Err(ImpedanceError::InvalidParameter("ndof must be > 0".into()));
        }
        if inertia <= 0.0 {
            return Err(ImpedanceError::InvalidParameter(
                "inertia must be > 0".into(),
            ));
        }
        Ok(Self {
            ndof,
            inertias: vec![inertia; ndof],
            dampings: vec![damping; ndof],
            stiffnesses: vec![stiffness; ndof],
            x_eq: vec![0.0; ndof],
            positions: vec![0.0; ndof],
            velocities: vec![0.0; ndof],
        })
    }

    /// Create from per-DOF parameter vectors.
    pub fn new(
        inertias: Vec<f64>,
        dampings: Vec<f64>,
        stiffnesses: Vec<f64>,
    ) -> Result<Self, ImpedanceError> {
        let n = inertias.len();
        if n == 0 {
            return Err(ImpedanceError::InvalidParameter("empty parameters".into()));
        }
        if dampings.len() != n || stiffnesses.len() != n {
            return Err(ImpedanceError::DimensionMismatch {
                expected: n,
                got: dampings.len().min(stiffnesses.len()),
            });
        }
        for (i, &m) in inertias.iter().enumerate() {
            if m <= 0.0 {
                return Err(ImpedanceError::InvalidParameter(
                    format!("inertia[{i}] must be > 0"),
                ));
            }
        }
        Ok(Self {
            ndof: n,
            inertias,
            dampings,
            stiffnesses,
            x_eq: vec![0.0; n],
            positions: vec![0.0; n],
            velocities: vec![0.0; n],
        })
    }

    /// Builder: set equilibrium positions.
    pub fn with_equilibrium(mut self, x_eq: Vec<f64>) -> Self {
        if x_eq.len() == self.ndof {
            self.x_eq = x_eq;
        }
        self
    }

    /// Compute actuator forces for all DOFs.
    pub fn compute_forces(
        &self,
        measured_pos: &[f64],
        measured_vel: &[f64],
        ext_forces: &[f64],
    ) -> Result<Vec<f64>, ImpedanceError> {
        if measured_pos.len() != self.ndof
            || measured_vel.len() != self.ndof
            || ext_forces.len() != self.ndof
        {
            return Err(ImpedanceError::DimensionMismatch {
                expected: self.ndof,
                got: measured_pos.len().min(measured_vel.len()),
            });
        }

        let forces: Vec<f64> = (0..self.ndof)
            .map(|i| {
                let spring = self.stiffnesses[i] * (self.x_eq[i] - measured_pos[i]);
                let damp = -self.dampings[i] * measured_vel[i];
                spring + damp + ext_forces[i]
            })
            .collect();

        Ok(forces)
    }

    /// Simulate forward by dt with external forces.
    pub fn step(&mut self, ext_forces: &[f64], dt: f64) -> Result<(), ImpedanceError> {
        if ext_forces.len() != self.ndof {
            return Err(ImpedanceError::DimensionMismatch {
                expected: self.ndof,
                got: ext_forces.len(),
            });
        }
        if dt <= 0.0 {
            return Ok(());
        }

        for i in 0..self.ndof {
            let accel = (ext_forces[i]
                - self.dampings[i] * self.velocities[i]
                - self.stiffnesses[i] * (self.positions[i] - self.x_eq[i]))
                / self.inertias[i];
            self.velocities[i] += accel * dt;
            self.positions[i] += self.velocities[i] * dt;
        }

        Ok(())
    }

    /// Natural frequencies for each DOF (rad/s).
    pub fn natural_frequencies(&self) -> Vec<f64> {
        (0..self.ndof)
            .map(|i| (self.stiffnesses[i] / self.inertias[i]).sqrt())
            .collect()
    }

    /// Damping ratios for each DOF.
    pub fn damping_ratios(&self) -> Vec<f64> {
        (0..self.ndof)
            .map(|i| {
                self.dampings[i] / (2.0 * (self.stiffnesses[i] * self.inertias[i]).sqrt())
            })
            .collect()
    }

    /// Reset all DOFs to equilibrium.
    pub fn reset(&mut self) {
        self.positions = self.x_eq.clone();
        self.velocities = vec![0.0; self.ndof];
    }
}

impl fmt::Display for ImpedanceModelND {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImpedanceND(ndof={}", self.ndof)?;
        if self.ndof <= 3 {
            let ratios = self.damping_ratios();
            write!(f, ", ζ={:?}", ratios)?;
        }
        write!(f, ")")
    }
}

// ── Compliance Profile ─────────────────────────────────────────

/// Compliance profile: maps task-space directions to impedance parameters.
///
/// Allows different stiffness along different Cartesian axes (e.g., stiff
/// along approach axis, compliant along contact surface).
#[derive(Debug, Clone, PartialEq)]
pub struct ComplianceProfile {
    /// Stiffness values for [x, y, z, rx, ry, rz].
    pub stiffness: [f64; 6],
    /// Damping values for [x, y, z, rx, ry, rz].
    pub damping: [f64; 6],
    /// Profile name.
    pub name: String,
}

impl ComplianceProfile {
    /// Uniform compliance (isotropic).
    pub fn uniform(stiffness: f64, damping: f64) -> Self {
        Self {
            stiffness: [stiffness; 6],
            damping: [damping; 6],
            name: "uniform".to_string(),
        }
    }

    /// Compliant in one axis, stiff in others.
    pub fn compliant_axis(
        axis: usize,
        soft_k: f64,
        stiff_k: f64,
        damping: f64,
    ) -> Self {
        let mut stiffness = [stiff_k; 6];
        if axis < 6 {
            stiffness[axis] = soft_k;
        }
        Self {
            stiffness,
            damping: [damping; 6],
            name: format!("compliant-axis-{axis}"),
        }
    }

    /// Builder: set per-axis stiffness.
    pub fn with_stiffness(mut self, stiffness: [f64; 6]) -> Self {
        self.stiffness = stiffness;
        self
    }

    /// Builder: set per-axis damping.
    pub fn with_damping(mut self, damping: [f64; 6]) -> Self {
        self.damping = damping;
        self
    }

    /// Builder: set name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Interpolate between two profiles (alpha in [0, 1]).
    pub fn interpolate(a: &ComplianceProfile, b: &ComplianceProfile, alpha: f64) -> Self {
        let t = alpha.clamp(0.0, 1.0);
        let mut stiffness = [0.0; 6];
        let mut damping = [0.0; 6];
        for i in 0..6 {
            stiffness[i] = a.stiffness[i] + t * (b.stiffness[i] - a.stiffness[i]);
            damping[i] = a.damping[i] + t * (b.damping[i] - a.damping[i]);
        }
        Self {
            stiffness,
            damping,
            name: format!("interp({:.2})", t),
        }
    }

    /// Maximum stiffness across all axes.
    pub fn max_stiffness(&self) -> f64 {
        self.stiffness.iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Minimum stiffness across all axes.
    pub fn min_stiffness(&self) -> f64 {
        self.stiffness.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Condition number (max_K / min_K) — measures anisotropy.
    pub fn condition_number(&self) -> f64 {
        let min_k = self.min_stiffness();
        if min_k < 1e-12 {
            return f64::INFINITY;
        }
        self.max_stiffness() / min_k
    }
}

impl fmt::Display for ComplianceProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Compliance('{}', K=[{:.1},{:.1},{:.1},{:.2},{:.2},{:.2}])",
            self.name,
            self.stiffness[0], self.stiffness[1], self.stiffness[2],
            self.stiffness[3], self.stiffness[4], self.stiffness[5]
        )
    }
}

// ── Variable Impedance Controller ──────────────────────────────

/// Variable impedance controller that adjusts stiffness/damping online.
///
/// Implements a simple energy-based adaptation: reduce stiffness when
/// interaction energy exceeds a threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableImpedanceController {
    /// Base impedance model (1-DOF for simplicity).
    pub model: ImpedanceModel1D,
    /// Minimum stiffness.
    pub min_stiffness: f64,
    /// Maximum stiffness.
    pub max_stiffness: f64,
    /// Interaction energy accumulator.
    pub energy: f64,
    /// Energy threshold for adaptation.
    pub energy_threshold: f64,
    /// Adaptation rate.
    pub adaptation_rate: f64,
}

impl VariableImpedanceController {
    /// Create a variable impedance controller.
    pub fn new(
        model: ImpedanceModel1D,
        min_stiffness: f64,
        max_stiffness: f64,
    ) -> Self {
        Self {
            model,
            min_stiffness,
            max_stiffness,
            energy: 0.0,
            energy_threshold: 1.0,
            adaptation_rate: 0.1,
        }
    }

    /// Builder: set energy threshold.
    pub fn with_energy_threshold(mut self, threshold: f64) -> Self {
        self.energy_threshold = threshold.abs();
        self
    }

    /// Builder: set adaptation rate.
    pub fn with_adaptation_rate(mut self, rate: f64) -> Self {
        self.adaptation_rate = rate.abs();
        self
    }

    /// Update the controller given measured state and external force.
    /// Returns the actuator force command.
    pub fn update(
        &mut self,
        measured_pos: f64,
        measured_vel: f64,
        external_force: f64,
        dt: f64,
    ) -> f64 {
        // Accumulate interaction energy: E = integral(F_ext * v * dt).
        self.energy += (external_force * measured_vel * dt).abs();
        // Exponential decay.
        self.energy *= (-dt * 0.5_f64).exp();

        // Adapt stiffness based on energy.
        if self.energy > self.energy_threshold {
            // Reduce stiffness (become more compliant).
            let ratio = self.energy / self.energy_threshold;
            let new_k = self.model.stiffness / (1.0 + self.adaptation_rate * ratio);
            self.model.stiffness = new_k.max(self.min_stiffness);
        } else {
            // Gradually restore stiffness.
            let new_k = self.model.stiffness + self.adaptation_rate * dt * self.max_stiffness;
            self.model.stiffness = new_k.min(self.max_stiffness);
        }

        // Adjust damping to maintain critical damping ratio.
        self.model.damping =
            2.0 * (self.model.stiffness * self.model.inertia).sqrt();

        self.model.compute_force(measured_pos, measured_vel, external_force)
    }

    /// Current stiffness.
    pub fn current_stiffness(&self) -> f64 {
        self.model.stiffness
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.energy = 0.0;
        self.model.stiffness = self.max_stiffness;
        self.model.reset();
    }
}

impl fmt::Display for VariableImpedanceController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VarImpedance(K={:.2}, E={:.4}, range=[{:.1},{:.1}])",
            self.model.stiffness, self.energy, self.min_stiffness, self.max_stiffness
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impedance_1d_creation() {
        let m = ImpedanceModel1D::new(1.0, 10.0, 100.0);
        assert!(m.is_ok());
    }

    #[test]
    fn test_impedance_1d_invalid_inertia() {
        assert!(ImpedanceModel1D::new(0.0, 10.0, 100.0).is_err());
        assert!(ImpedanceModel1D::new(-1.0, 10.0, 100.0).is_err());
    }

    #[test]
    fn test_impedance_1d_natural_frequency() {
        let m = ImpedanceModel1D::new(1.0, 0.0, 100.0).unwrap();
        assert!((m.natural_frequency() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_impedance_1d_damping_ratio() {
        // Critical damping: B = 2*sqrt(K*M)
        let m = ImpedanceModel1D::new(1.0, 20.0, 100.0).unwrap();
        assert!((m.damping_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_impedance_1d_spring_force() {
        let m = ImpedanceModel1D::new(1.0, 0.0, 100.0).unwrap()
            .with_equilibrium(1.0);
        // Displaced 0.1 m from equilibrium.
        let f = m.compute_force(0.9, 0.0, 0.0);
        // K*(x_eq - x) = 100*(1.0 - 0.9) = 10.0
        assert!((f - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_impedance_1d_damping_force() {
        let m = ImpedanceModel1D::new(1.0, 10.0, 0.0).unwrap();
        let f = m.compute_force(0.0, 2.0, 0.0);
        // -B*v = -10*2 = -20
        assert!((f - (-20.0)).abs() < 1e-9);
    }

    #[test]
    fn test_impedance_1d_step_convergence() {
        // Critically damped system should converge to equilibrium.
        let mut m = ImpedanceModel1D::new(1.0, 20.0, 100.0).unwrap()
            .with_equilibrium(0.0)
            .with_initial_state(1.0, 0.0);

        for _ in 0..10_000 {
            m.step(0.0, 0.001);
        }
        assert!(m.position.abs() < 0.01, "should converge: x={}", m.position);
    }

    #[test]
    fn test_impedance_nd_uniform() {
        let m = ImpedanceModelND::new_uniform(3, 1.0, 10.0, 100.0);
        assert!(m.is_ok());
        assert_eq!(m.unwrap().ndof, 3);
    }

    #[test]
    fn test_impedance_nd_dim_mismatch() {
        let m = ImpedanceModelND::new(
            vec![1.0, 1.0],
            vec![10.0],
            vec![100.0, 100.0],
        );
        assert!(m.is_err());
    }

    #[test]
    fn test_impedance_nd_compute_forces() {
        let m = ImpedanceModelND::new_uniform(2, 1.0, 0.0, 100.0).unwrap()
            .with_equilibrium(vec![1.0, 2.0]);

        let forces = m.compute_forces(&[0.9, 1.8], &[0.0, 0.0], &[0.0, 0.0]).unwrap();
        // K*(x_eq - x) = 100*0.1 = 10.0
        assert!((forces[0] - 10.0).abs() < 1e-9);
        assert!((forces[1] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_impedance_nd_step() {
        let mut m = ImpedanceModelND::new_uniform(2, 1.0, 20.0, 100.0).unwrap();
        m.positions = vec![1.0, -1.0];
        for _ in 0..10_000 {
            m.step(&[0.0, 0.0], 0.001).unwrap();
        }
        assert!(m.positions[0].abs() < 0.01);
        assert!(m.positions[1].abs() < 0.01);
    }

    #[test]
    fn test_compliance_uniform() {
        let cp = ComplianceProfile::uniform(500.0, 50.0);
        assert!((cp.condition_number() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compliance_anisotropic() {
        let cp = ComplianceProfile::compliant_axis(2, 10.0, 1000.0, 50.0);
        assert!((cp.stiffness[2] - 10.0).abs() < 1e-9);
        assert!((cp.condition_number() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_compliance_interpolation() {
        let a = ComplianceProfile::uniform(100.0, 10.0);
        let b = ComplianceProfile::uniform(200.0, 20.0);
        let c = ComplianceProfile::interpolate(&a, &b, 0.5);
        assert!((c.stiffness[0] - 150.0).abs() < 1e-9);
        assert!((c.damping[0] - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_variable_impedance_reduces_stiffness() {
        let model = ImpedanceModel1D::new(1.0, 20.0, 500.0).unwrap();
        let mut vic = VariableImpedanceController::new(model, 10.0, 500.0)
            .with_energy_threshold(0.1);

        // Apply large external force repeatedly.
        for _ in 0..100 {
            vic.update(0.0, 1.0, 50.0, 0.01);
        }
        assert!(
            vic.current_stiffness() < 500.0,
            "stiffness should decrease: K={}",
            vic.current_stiffness()
        );
    }

    #[test]
    fn test_variable_impedance_reset() {
        let model = ImpedanceModel1D::new(1.0, 20.0, 500.0).unwrap();
        let mut vic = VariableImpedanceController::new(model, 10.0, 500.0);
        vic.update(0.0, 1.0, 50.0, 0.01);
        vic.reset();
        assert!((vic.current_stiffness() - 500.0).abs() < 1e-9);
        assert!(vic.energy.abs() < 1e-15);
    }

    #[test]
    fn test_display_impedance_1d() {
        let m = ImpedanceModel1D::new(1.0, 20.0, 100.0).unwrap();
        let s = format!("{m}");
        assert!(s.contains("Impedance1D"));
    }

    #[test]
    fn test_display_compliance() {
        let cp = ComplianceProfile::uniform(100.0, 10.0);
        let s = format!("{cp}");
        assert!(s.contains("Compliance"));
    }

    #[test]
    fn test_display_variable_impedance() {
        let model = ImpedanceModel1D::new(1.0, 20.0, 100.0).unwrap();
        let vic = VariableImpedanceController::new(model, 10.0, 100.0);
        let s = format!("{vic}");
        assert!(s.contains("VarImpedance"));
    }
}
