//! Adaptive Control — Model Reference Adaptive Control (MRAC), MIT rule,
//! Lyapunov-based adaptation, recursive least squares parameter estimation,
//! gain scheduling, and self-tuning regulators.
//!
//! Pure-Rust adaptive controllers for systems with uncertain or time-varying
//! parameters, suitable for embedded and server-side control workloads.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// Adaptive control errors.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveError {
    /// Invalid parameter.
    InvalidParameter(String),
    /// Dimension mismatch.
    DimensionMismatch(String),
    /// Singular matrix.
    SingularMatrix,
    /// Adaptation diverged.
    Diverged(String),
    /// Insufficient data for estimation.
    InsufficientData(String),
}

impl std::fmt::Display for AdaptiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::Diverged(msg) => write!(f, "adaptation diverged: {msg}"),
            Self::InsufficientData(msg) => write!(f, "insufficient data: {msg}"),
        }
    }
}

impl std::error::Error for AdaptiveError {}

// ── Reference Model ─────────────────────────────────────────────

/// First-order reference model: y_m[k+1] = a_m * y_m[k] + b_m * r[k].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceModel {
    /// Model pole (should be stable, |a_m| < 1 for discrete).
    pub a_m: f64,
    /// Model input gain.
    pub b_m: f64,
    /// Current model output.
    pub y_m: f64,
}

impl ReferenceModel {
    /// Create a first-order reference model.
    pub fn new(a_m: f64, b_m: f64) -> Result<Self, AdaptiveError> {
        if a_m.abs() >= 1.0 {
            return Err(AdaptiveError::InvalidParameter(
                "reference model pole must be stable (|a_m| < 1)".into(),
            ));
        }
        Ok(Self { a_m, b_m, y_m: 0.0 })
    }

    /// Step the reference model with a reference command.
    pub fn step(&mut self, r: f64) -> f64 {
        self.y_m = self.a_m * self.y_m + self.b_m * r;
        self.y_m
    }

    /// Steady-state gain.
    pub fn dc_gain(&self) -> f64 {
        self.b_m / (1.0 - self.a_m)
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.y_m = 0.0;
    }
}

// ── MIT Rule MRAC ───────────────────────────────────────────────

/// Model Reference Adaptive Control using the MIT rule.
/// Plant: y[k+1] = a_p * y[k] + b_p * u[k]  (a_p, b_p unknown).
/// Controller: u = theta_r * r + theta_y * y.
/// Adaptation: theta_dot = -gamma * e * (de/dtheta) (gradient descent on e^2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MitRuleMrac {
    /// Reference model.
    pub model: ReferenceModel,
    /// Feedforward parameter estimate.
    pub theta_r: f64,
    /// Feedback parameter estimate.
    pub theta_y: f64,
    /// Adaptation rate for theta_r.
    pub gamma_r: f64,
    /// Adaptation rate for theta_y.
    pub gamma_y: f64,
    /// Current plant output.
    pub y: f64,
    /// Normalization signal (prevents bursting).
    pub normalization: f64,
    /// Previous reference command.
    pub prev_r: f64,
    /// Previous plant output.
    pub prev_y: f64,
}

impl MitRuleMrac {
    /// Create MIT-rule MRAC.
    pub fn new(
        model: ReferenceModel,
        gamma_r: f64,
        gamma_y: f64,
    ) -> Result<Self, AdaptiveError> {
        if gamma_r < 0.0 || gamma_y < 0.0 {
            return Err(AdaptiveError::InvalidParameter("gamma must be >= 0".into()));
        }
        Ok(Self {
            model,
            theta_r: 1.0,
            theta_y: 0.0,
            gamma_r,
            gamma_y,
            y: 0.0,
            normalization: 1.0,
            prev_r: 0.0,
            prev_y: 0.0,
        })
    }

    /// Compute control input and adapt parameters.
    /// Returns (control_input, tracking_error).
    pub fn step(&mut self, r: f64, y_plant: f64) -> (f64, f64) {
        // Step reference model.
        let y_m = self.model.step(r);

        // Tracking error.
        let e = y_plant - y_m;

        // Normalization to prevent bursting.
        self.normalization = 1.0 + self.prev_r * self.prev_r + self.prev_y * self.prev_y;

        // MIT rule: sensitivity is approximated as partial of e w.r.t. theta.
        // de/dtheta_r ≈ b_p * r (approximate with model gain * r).
        // de/dtheta_y ≈ b_p * y (approximate with model gain * y).
        // Use reference model gain as proxy for plant gain.
        let sensitivity_r = self.model.b_m * self.prev_r / self.normalization;
        let sensitivity_y = self.model.b_m * self.prev_y / self.normalization;

        // Parameter adaptation.
        self.theta_r -= self.gamma_r * e * sensitivity_r;
        self.theta_y -= self.gamma_y * e * sensitivity_y;

        // Compute control input.
        let u = self.theta_r * r + self.theta_y * y_plant;

        self.prev_r = r;
        self.prev_y = y_plant;
        self.y = y_plant;

        (u, e)
    }

    /// Reset adaptation.
    pub fn reset(&mut self) {
        self.theta_r = 1.0;
        self.theta_y = 0.0;
        self.y = 0.0;
        self.model.reset();
        self.prev_r = 0.0;
        self.prev_y = 0.0;
    }
}

// ── Lyapunov MRAC ───────────────────────────────────────────────

/// Lyapunov-based MRAC with guaranteed stability.
/// Uses Lyapunov function V = e^2/2 + (theta - theta*)^2 / (2*gamma).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LyapunovMrac {
    /// Reference model.
    pub model: ReferenceModel,
    /// Adaptive gain.
    pub theta: f64,
    /// Adaptation rate.
    pub gamma: f64,
    /// Sigma modification for robustness (leakage).
    pub sigma: f64,
    /// Current plant output.
    pub y: f64,
    /// Previous reference.
    pub prev_r: f64,
}

impl LyapunovMrac {
    /// Create Lyapunov-based MRAC.
    pub fn new(model: ReferenceModel, gamma: f64, sigma: f64) -> Result<Self, AdaptiveError> {
        if gamma <= 0.0 {
            return Err(AdaptiveError::InvalidParameter("gamma must be > 0".into()));
        }
        if sigma < 0.0 {
            return Err(AdaptiveError::InvalidParameter("sigma must be >= 0".into()));
        }
        Ok(Self {
            model,
            theta: 1.0,
            gamma,
            sigma,
            y: 0.0,
            prev_r: 0.0,
        })
    }

    /// Compute control and adapt.
    pub fn step(&mut self, r: f64, y_plant: f64) -> (f64, f64) {
        let y_m = self.model.step(r);
        let e = y_plant - y_m;

        // Lyapunov adaptation law with sigma modification.
        // theta_dot = -gamma * e * r - sigma * gamma * theta
        self.theta -= self.gamma * e * self.prev_r - self.sigma * self.gamma * self.theta;

        let u = self.theta * r;
        self.prev_r = r;
        self.y = y_plant;

        (u, e)
    }

    /// Compute Lyapunov function value.
    pub fn lyapunov_value(&self, e: f64, theta_star: f64) -> f64 {
        0.5 * e * e + (self.theta - theta_star).powi(2) / (2.0 * self.gamma)
    }
}

// ── Recursive Least Squares ─────────────────────────────────────

/// Online parameter estimation using Recursive Least Squares (RLS).
/// Estimates theta in: y = phi' * theta + noise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecursiveLeastSquares {
    /// Parameter estimate vector (n_params).
    pub theta: Vec<f64>,
    /// Covariance matrix (n x n, row-major).
    pub p: Vec<f64>,
    /// Number of parameters.
    pub n: usize,
    /// Forgetting factor (0 < lambda <= 1). 1 = no forgetting.
    pub lambda: f64,
    /// Number of updates performed.
    pub count: usize,
}

impl RecursiveLeastSquares {
    /// Create RLS estimator for n parameters.
    pub fn new(n: usize, lambda: f64) -> Result<Self, AdaptiveError> {
        if n == 0 {
            return Err(AdaptiveError::InvalidParameter("n must be > 0".into()));
        }
        if lambda <= 0.0 || lambda > 1.0 {
            return Err(AdaptiveError::InvalidParameter(
                "lambda must be in (0, 1]".into(),
            ));
        }
        // Initialize P = large * I (diffuse prior).
        let mut p = vec![0.0; n * n];
        for i in 0..n {
            p[i * n + i] = 1000.0;
        }
        Ok(Self {
            theta: vec![0.0; n],
            p,
            n,
            lambda,
            count: 0,
        })
    }

    /// Update with new data: y = phi' * theta.
    pub fn update(&mut self, phi: &[f64], y: f64) -> Result<f64, AdaptiveError> {
        if phi.len() != self.n {
            return Err(AdaptiveError::DimensionMismatch(format!(
                "expected {} regressors, got {}", self.n, phi.len()
            )));
        }

        let n = self.n;

        // Prediction error.
        let y_hat: f64 = phi.iter().zip(&self.theta).map(|(p, t)| p * t).sum();
        let e = y - y_hat;

        // P * phi.
        let mut p_phi = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                p_phi[i] += self.p[i * n + j] * phi[j];
            }
        }

        // phi' * P * phi (scalar).
        let phi_p_phi: f64 = phi.iter().zip(&p_phi).map(|(a, b)| a * b).sum();

        // Gain: k = P * phi / (lambda + phi' * P * phi).
        let denom = self.lambda + phi_p_phi;
        if denom.abs() < 1e-14 {
            return Err(AdaptiveError::SingularMatrix);
        }
        let k: Vec<f64> = p_phi.iter().map(|v| v / denom).collect();

        // Theta update.
        for i in 0..n {
            self.theta[i] += k[i] * e;
        }

        // P update: P = (P - k * phi' * P) / lambda.
        // More stable: P = (I - k * phi') * P / lambda.
        let mut new_p = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let ij_val = if i == j { 1.0 } else { 0.0 };
                let k_phi = k[i] * phi[j];
                new_p[i * n + j] = (ij_val - k_phi) * self.p[i * n + j];
            }
        }
        // Actually need full matrix multiply (I - k*phi') * P, let's do it properly.
        let mut final_p = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for m in 0..n {
                    let i_kphi = if i == m { 1.0 } else { 0.0 } - k[i] * phi[m];
                    sum += i_kphi * self.p[m * n + j];
                }
                final_p[i * n + j] = sum / self.lambda;
            }
        }
        self.p = final_p;
        self.count += 1;

        Ok(e)
    }

    /// Get estimated parameters.
    pub fn parameters(&self) -> &[f64] {
        &self.theta
    }

    /// Reset estimator.
    pub fn reset(&mut self) {
        self.theta = vec![0.0; self.n];
        self.p = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            self.p[i * self.n + i] = 1000.0;
        }
        self.count = 0;
    }
}

// ── Gain Scheduling ─────────────────────────────────────────────

/// Gain schedule entry: operating point + controller gains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GainScheduleEntry {
    /// Operating point parameter.
    pub operating_point: f64,
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
    /// Derivative gain.
    pub kd: f64,
}

/// Gain scheduler: interpolates PID gains based on operating point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GainScheduler {
    /// Schedule entries (sorted by operating_point).
    pub entries: Vec<GainScheduleEntry>,
}

impl GainScheduler {
    /// Create from entries (will sort by operating point).
    pub fn new(mut entries: Vec<GainScheduleEntry>) -> Result<Self, AdaptiveError> {
        if entries.len() < 2 {
            return Err(AdaptiveError::InsufficientData(
                "need at least 2 schedule entries".into(),
            ));
        }
        entries.sort_by(|a, b| a.operating_point.partial_cmp(&b.operating_point).unwrap());
        Ok(Self { entries })
    }

    /// Interpolate gains at a given operating point.
    pub fn gains(&self, op: f64) -> (f64, f64, f64) {
        let n = self.entries.len();

        // Clamp to range.
        if op <= self.entries[0].operating_point {
            return (self.entries[0].kp, self.entries[0].ki, self.entries[0].kd);
        }
        if op >= self.entries[n - 1].operating_point {
            let e = &self.entries[n - 1];
            return (e.kp, e.ki, e.kd);
        }

        // Find segment.
        let mut lo = 0;
        for i in 0..n - 1 {
            if op >= self.entries[i].operating_point && op <= self.entries[i + 1].operating_point {
                lo = i;
                break;
            }
        }
        let hi = lo + 1;

        let t = (op - self.entries[lo].operating_point)
            / (self.entries[hi].operating_point - self.entries[lo].operating_point);

        let kp = self.entries[lo].kp + t * (self.entries[hi].kp - self.entries[lo].kp);
        let ki = self.entries[lo].ki + t * (self.entries[hi].ki - self.entries[lo].ki);
        let kd = self.entries[lo].kd + t * (self.entries[hi].kd - self.entries[lo].kd);

        (kp, ki, kd)
    }
}

// ── Self-Tuning Regulator ───────────────────────────────────────

/// Self-tuning regulator: estimates plant online (RLS), designs controller.
/// Plant model: y[k] = a1*y[k-1] + b0*u[k-1]  (first-order ARX).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelfTuningRegulator {
    /// Online parameter estimator.
    pub rls: RecursiveLeastSquares,
    /// Previous output.
    pub y_prev: f64,
    /// Previous input.
    pub u_prev: f64,
    /// Desired closed-loop pole.
    pub desired_pole: f64,
    /// Control output limits.
    pub u_min: f64,
    pub u_max: f64,
}

impl SelfTuningRegulator {
    /// Create a self-tuning regulator.
    pub fn new(desired_pole: f64, u_min: f64, u_max: f64) -> Result<Self, AdaptiveError> {
        if desired_pole.abs() >= 1.0 {
            return Err(AdaptiveError::InvalidParameter(
                "desired pole must be stable (|p| < 1)".into(),
            ));
        }
        // 2 parameters: a1 and b0.
        let rls = RecursiveLeastSquares::new(2, 0.99)?;
        Ok(Self {
            rls,
            y_prev: 0.0,
            u_prev: 0.0,
            desired_pole,
            u_min,
            u_max,
        })
    }

    /// Compute control for setpoint tracking.
    pub fn control(&mut self, setpoint: f64, y: f64) -> Result<f64, AdaptiveError> {
        // Form regressor: phi = [y_prev, u_prev].
        let phi = vec![self.y_prev, self.u_prev];

        // Update parameter estimates.
        self.rls.update(&phi, y)?;

        let a1_hat = self.rls.theta[0];
        let b0_hat = self.rls.theta[1];

        // Design pole placement controller.
        // Closed-loop: y[k+1] = desired_pole * y[k] + (1 - desired_pole) * setpoint.
        // u = (1/b0_hat) * ((desired_pole - a1_hat) * y + (1 - desired_pole) * setpoint)
        let u = if b0_hat.abs() > 1e-6 {
            ((self.desired_pole - a1_hat) * y + (1.0 - self.desired_pole) * setpoint)
                / b0_hat
        } else {
            // b0 estimate too small — use simple proportional.
            setpoint - y
        };

        let u_clamped = u.clamp(self.u_min, self.u_max);

        self.y_prev = y;
        self.u_prev = u_clamped;

        Ok(u_clamped)
    }

    /// Get estimated plant parameters (a1, b0).
    pub fn plant_parameters(&self) -> (f64, f64) {
        (self.rls.theta[0], self.rls.theta[1])
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.rls.reset();
        self.y_prev = 0.0;
        self.u_prev = 0.0;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_reference_model_step() {
        let mut rm = ReferenceModel::new(0.9, 0.1).unwrap();
        rm.step(1.0);
        assert!(approx(rm.y_m, 0.1, 1e-10));
        rm.step(1.0);
        assert!(approx(rm.y_m, 0.19, 1e-10));
    }

    #[test]
    fn test_reference_model_dc_gain() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        assert!(approx(rm.dc_gain(), 1.0, 1e-10));
    }

    #[test]
    fn test_reference_model_converges() {
        let mut rm = ReferenceModel::new(0.9, 0.1).unwrap();
        for _ in 0..200 {
            rm.step(1.0);
        }
        // Should converge to DC gain * r = 1.0 * 1.0 = 1.0.
        assert!(approx(rm.y_m, 1.0, 0.01));
    }

    #[test]
    fn test_reference_model_unstable_rejected() {
        assert!(ReferenceModel::new(1.0, 0.1).is_err());
        assert!(ReferenceModel::new(-1.0, 0.1).is_err());
    }

    #[test]
    fn test_mit_rule_creation() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        let mrac = MitRuleMrac::new(rm, 0.01, 0.01).unwrap();
        assert!(approx(mrac.theta_r, 1.0, 1e-10));
    }

    #[test]
    fn test_mit_rule_negative_gamma_rejected() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        assert!(MitRuleMrac::new(rm, -0.01, 0.01).is_err());
    }

    #[test]
    fn test_mit_rule_step() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        let mut mrac = MitRuleMrac::new(rm, 0.01, 0.01).unwrap();
        let (u, e) = mrac.step(1.0, 0.0);
        assert!(u.is_finite());
        assert!(e.is_finite());
    }

    #[test]
    fn test_mit_rule_reset() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        let mut mrac = MitRuleMrac::new(rm, 0.01, 0.01).unwrap();
        mrac.step(1.0, 0.5);
        mrac.reset();
        assert!(approx(mrac.theta_r, 1.0, 1e-10));
        assert!(approx(mrac.theta_y, 0.0, 1e-10));
    }

    #[test]
    fn test_lyapunov_mrac_creation() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        let lmrac = LyapunovMrac::new(rm, 0.1, 0.01).unwrap();
        assert!(approx(lmrac.theta, 1.0, 1e-10));
    }

    #[test]
    fn test_lyapunov_mrac_invalid_gamma() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        assert!(LyapunovMrac::new(rm, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_lyapunov_function_positive() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        let lmrac = LyapunovMrac::new(rm, 0.1, 0.0).unwrap();
        let v = lmrac.lyapunov_value(1.0, 2.0);
        assert!(v > 0.0);
    }

    #[test]
    fn test_lyapunov_function_zero_at_equilibrium() {
        let rm = ReferenceModel::new(0.9, 0.1).unwrap();
        let lmrac = LyapunovMrac::new(rm, 0.1, 0.0).unwrap();
        let v = lmrac.lyapunov_value(0.0, lmrac.theta);
        assert!(approx(v, 0.0, 1e-10));
    }

    #[test]
    fn test_rls_creation() {
        let rls = RecursiveLeastSquares::new(3, 0.99).unwrap();
        assert_eq!(rls.n, 3);
        assert_eq!(rls.theta.len(), 3);
    }

    #[test]
    fn test_rls_invalid_params() {
        assert!(RecursiveLeastSquares::new(0, 0.99).is_err());
        assert!(RecursiveLeastSquares::new(1, 0.0).is_err());
        assert!(RecursiveLeastSquares::new(1, 1.1).is_err());
    }

    #[test]
    fn test_rls_estimates_linear() {
        // y = 2*x + 3 => theta = [2, 3], phi = [x, 1]
        let mut rls = RecursiveLeastSquares::new(2, 1.0).unwrap();
        for i in 0..100 {
            let x = i as f64 * 0.1;
            let y = 2.0 * x + 3.0;
            let phi = vec![x, 1.0];
            rls.update(&phi, y).unwrap();
        }
        assert!(approx(rls.theta[0], 2.0, 0.1));
        assert!(approx(rls.theta[1], 3.0, 0.1));
    }

    #[test]
    fn test_rls_dimension_mismatch() {
        let mut rls = RecursiveLeastSquares::new(2, 1.0).unwrap();
        assert!(rls.update(&[1.0], 1.0).is_err());
    }

    #[test]
    fn test_rls_reset() {
        let mut rls = RecursiveLeastSquares::new(2, 1.0).unwrap();
        rls.update(&[1.0, 1.0], 5.0).unwrap();
        rls.reset();
        assert_eq!(rls.count, 0);
        assert!(approx(rls.theta[0], 0.0, 1e-10));
    }

    #[test]
    fn test_gain_scheduler_creation() {
        let entries = vec![
            GainScheduleEntry { operating_point: 0.0, kp: 1.0, ki: 0.1, kd: 0.01 },
            GainScheduleEntry { operating_point: 10.0, kp: 5.0, ki: 0.5, kd: 0.05 },
        ];
        let gs = GainScheduler::new(entries).unwrap();
        assert_eq!(gs.entries.len(), 2);
    }

    #[test]
    fn test_gain_scheduler_insufficient_entries() {
        let entries = vec![
            GainScheduleEntry { operating_point: 0.0, kp: 1.0, ki: 0.1, kd: 0.01 },
        ];
        assert!(GainScheduler::new(entries).is_err());
    }

    #[test]
    fn test_gain_scheduler_interpolation() {
        let entries = vec![
            GainScheduleEntry { operating_point: 0.0, kp: 1.0, ki: 0.1, kd: 0.01 },
            GainScheduleEntry { operating_point: 10.0, kp: 5.0, ki: 0.5, kd: 0.05 },
        ];
        let gs = GainScheduler::new(entries).unwrap();
        let (kp, ki, kd) = gs.gains(5.0);
        assert!(approx(kp, 3.0, 1e-4));
        assert!(approx(ki, 0.3, 1e-4));
        assert!(approx(kd, 0.03, 1e-4));
    }

    #[test]
    fn test_gain_scheduler_clamp_low() {
        let entries = vec![
            GainScheduleEntry { operating_point: 0.0, kp: 1.0, ki: 0.1, kd: 0.01 },
            GainScheduleEntry { operating_point: 10.0, kp: 5.0, ki: 0.5, kd: 0.05 },
        ];
        let gs = GainScheduler::new(entries).unwrap();
        let (kp, _, _) = gs.gains(-5.0);
        assert!(approx(kp, 1.0, 1e-4));
    }

    #[test]
    fn test_gain_scheduler_clamp_high() {
        let entries = vec![
            GainScheduleEntry { operating_point: 0.0, kp: 1.0, ki: 0.1, kd: 0.01 },
            GainScheduleEntry { operating_point: 10.0, kp: 5.0, ki: 0.5, kd: 0.05 },
        ];
        let gs = GainScheduler::new(entries).unwrap();
        let (kp, _, _) = gs.gains(20.0);
        assert!(approx(kp, 5.0, 1e-4));
    }

    #[test]
    fn test_self_tuning_regulator_creation() {
        let str_ctrl = SelfTuningRegulator::new(0.5, -10.0, 10.0).unwrap();
        assert_eq!(str_ctrl.rls.n, 2);
    }

    #[test]
    fn test_self_tuning_regulator_unstable_pole_rejected() {
        assert!(SelfTuningRegulator::new(1.0, -10.0, 10.0).is_err());
    }

    #[test]
    fn test_self_tuning_regulator_step() {
        let mut str_ctrl = SelfTuningRegulator::new(0.5, -10.0, 10.0).unwrap();
        let u = str_ctrl.control(1.0, 0.5).unwrap();
        assert!(u.is_finite());
        assert!(u >= -10.0 && u <= 10.0);
    }

    #[test]
    fn test_self_tuning_regulator_reset() {
        let mut str_ctrl = SelfTuningRegulator::new(0.5, -10.0, 10.0).unwrap();
        str_ctrl.control(1.0, 0.5).unwrap();
        str_ctrl.reset();
        assert!(approx(str_ctrl.y_prev, 0.0, 1e-10));
        assert!(approx(str_ctrl.u_prev, 0.0, 1e-10));
    }

    #[test]
    fn test_self_tuning_regulator_tracks_setpoint() {
        let mut str_ctrl = SelfTuningRegulator::new(0.5, -100.0, 100.0).unwrap();
        // Simple plant: y[k] = 0.8*y[k-1] + 0.5*u[k-1]
        let a_true = 0.8;
        let b_true = 0.5;
        let mut y = 0.0;
        let setpoint = 5.0;

        for _ in 0..200 {
            let u = str_ctrl.control(setpoint, y).unwrap();
            y = a_true * y + b_true * u;
        }

        // Should be near setpoint.
        assert!(approx(y, setpoint, 2.0));
    }

    #[test]
    fn test_rls_tracks_changing_params() {
        let mut rls = RecursiveLeastSquares::new(1, 0.95).unwrap(); // forgetting factor
        // First 50: y = 2*x
        for i in 0..50 {
            let x = i as f64 * 0.1;
            rls.update(&[x], 2.0 * x).unwrap();
        }
        assert!(approx(rls.theta[0], 2.0, 0.5));

        // Next 50: y = 5*x (parameter changed)
        for i in 50..150 {
            let x = i as f64 * 0.1;
            rls.update(&[x], 5.0 * x).unwrap();
        }
        assert!(approx(rls.theta[0], 5.0, 0.5));
    }

    #[test]
    fn test_gain_scheduler_three_points() {
        let entries = vec![
            GainScheduleEntry { operating_point: 0.0, kp: 1.0, ki: 0.0, kd: 0.0 },
            GainScheduleEntry { operating_point: 5.0, kp: 3.0, ki: 0.0, kd: 0.0 },
            GainScheduleEntry { operating_point: 10.0, kp: 10.0, ki: 0.0, kd: 0.0 },
        ];
        let gs = GainScheduler::new(entries).unwrap();
        let (kp, _, _) = gs.gains(7.5);
        assert!(approx(kp, 6.5, 1e-4));
    }

    #[test]
    fn test_reference_model_reset() {
        let mut rm = ReferenceModel::new(0.9, 0.1).unwrap();
        rm.step(1.0);
        rm.reset();
        assert!(approx(rm.y_m, 0.0, 1e-10));
    }
}
