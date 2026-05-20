//! Keplerian orbital mechanics — orbital elements, state vectors, transfers.
//!
//! Replaces poliastro / orb.js with pure Rust.
//! Orbital elements to/from state vectors, Kepler's equation solver,
//! orbit propagation, Hohmann transfers, vis-viva equation.

use std::f64::consts::PI;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for orbital mechanics.
#[derive(Debug, Clone, PartialEq)]
pub enum OrbitalError {
    /// Semi-major axis must be positive.
    NonPositiveSemiMajor(f64),
    /// Eccentricity must be in [0, 1) for elliptical orbits.
    InvalidEccentricity(f64),
    /// Gravitational parameter must be positive.
    NonPositiveMu(f64),
    /// Kepler's equation did not converge.
    KeplerNoConverge { max_iter: u32 },
    /// Transfer orbit is not valid.
    InvalidTransfer(String),
    /// Negative time of flight.
    NegativeTime(f64),
}

impl fmt::Display for OrbitalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveSemiMajor(a) => write!(f, "semi-major axis must be positive, got {a}"),
            Self::InvalidEccentricity(e) => write!(f, "eccentricity invalid: {e}"),
            Self::NonPositiveMu(mu) => write!(f, "mu must be positive, got {mu}"),
            Self::KeplerNoConverge { max_iter } => {
                write!(f, "Kepler equation did not converge in {max_iter} iterations")
            }
            Self::InvalidTransfer(s) => write!(f, "invalid transfer: {s}"),
            Self::NegativeTime(t) => write!(f, "time must be non-negative, got {t}"),
        }
    }
}

impl std::error::Error for OrbitalError {}

// ── Vec3 ────────────────────────────────────────────────────────

/// 3D vector for position/velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}

// ── Orbital Elements ────────────────────────────────────────────

/// Classical Keplerian orbital elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitalElements {
    /// Semi-major axis (length units).
    pub semi_major_axis: f64,
    /// Eccentricity [0, 1) for ellipses.
    pub eccentricity: f64,
    /// Inclination (radians).
    pub inclination: f64,
    /// Right ascension of ascending node (radians).
    pub raan: f64,
    /// Argument of periapsis (radians).
    pub arg_periapsis: f64,
    /// True anomaly (radians).
    pub true_anomaly: f64,
}

impl OrbitalElements {
    pub fn new(
        semi_major_axis: f64,
        eccentricity: f64,
        inclination: f64,
        raan: f64,
        arg_periapsis: f64,
        true_anomaly: f64,
    ) -> Result<Self, OrbitalError> {
        if semi_major_axis <= 0.0 {
            return Err(OrbitalError::NonPositiveSemiMajor(semi_major_axis));
        }
        if eccentricity < 0.0 || eccentricity >= 1.0 {
            return Err(OrbitalError::InvalidEccentricity(eccentricity));
        }
        Ok(Self { semi_major_axis, eccentricity, inclination, raan, arg_periapsis, true_anomaly })
    }

    /// Orbit period: T = 2*pi*sqrt(a^3/mu).
    pub fn period(&self, mu: f64) -> Result<f64, OrbitalError> {
        if mu <= 0.0 {
            return Err(OrbitalError::NonPositiveMu(mu));
        }
        Ok(2.0 * PI * (self.semi_major_axis.powi(3) / mu).sqrt())
    }

    /// Periapsis distance: a*(1-e).
    pub fn periapsis(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity)
    }

    /// Apoapsis distance: a*(1+e).
    pub fn apoapsis(&self) -> f64 {
        self.semi_major_axis * (1.0 + self.eccentricity)
    }

    /// Semi-latus rectum: p = a*(1-e^2).
    pub fn semi_latus_rectum(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity * self.eccentricity)
    }

    /// Specific orbital energy: -mu/(2a).
    pub fn specific_energy(&self, mu: f64) -> f64 {
        -mu / (2.0 * self.semi_major_axis)
    }

    /// Mean motion: n = sqrt(mu/a^3).
    pub fn mean_motion(&self, mu: f64) -> f64 {
        (mu / self.semi_major_axis.powi(3)).sqrt()
    }

    /// Convert true anomaly to eccentric anomaly.
    pub fn true_to_eccentric_anomaly(&self) -> f64 {
        let e = self.eccentricity;
        let nu = self.true_anomaly;
        let ea = ((1.0 - e) / (1.0 + e)).sqrt() * (nu / 2.0).tan();
        2.0 * ea.atan()
    }

    /// Convert true anomaly to mean anomaly.
    pub fn true_to_mean_anomaly(&self) -> f64 {
        let ea = self.true_to_eccentric_anomaly();
        ea - self.eccentricity * ea.sin()
    }
}

// ── State Vector ────────────────────────────────────────────────

/// Position and velocity state vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateVector {
    pub position: Vec3,
    pub velocity: Vec3,
}

// ── Conversions ─────────────────────────────────────────────────

/// Convert orbital elements to state vector.
pub fn elements_to_state(oe: &OrbitalElements, mu: f64) -> Result<StateVector, OrbitalError> {
    if mu <= 0.0 {
        return Err(OrbitalError::NonPositiveMu(mu));
    }
    let p = oe.semi_latus_rectum();
    let nu = oe.true_anomaly;
    let r_mag = p / (1.0 + oe.eccentricity * nu.cos());

    // Position and velocity in perifocal frame.
    let r_pqw = Vec3::new(r_mag * nu.cos(), r_mag * nu.sin(), 0.0);
    let coeff = (mu / p).sqrt();
    let v_pqw = Vec3::new(-coeff * nu.sin(), coeff * (oe.eccentricity + nu.cos()), 0.0);

    // Rotation from perifocal to inertial.
    let co = oe.arg_periapsis.cos();
    let so = oe.arg_periapsis.sin();
    let cO = oe.raan.cos();
    let sO = oe.raan.sin();
    let ci = oe.inclination.cos();
    let si = oe.inclination.sin();

    let rot = |v: Vec3| -> Vec3 {
        Vec3::new(
            (cO * co - sO * so * ci) * v.x + (-cO * so - sO * co * ci) * v.y,
            (sO * co + cO * so * ci) * v.x + (-sO * so + cO * co * ci) * v.y,
            (so * si) * v.x + (co * si) * v.y,
        )
    };

    Ok(StateVector { position: rot(r_pqw), velocity: rot(v_pqw) })
}

/// Convert state vector to orbital elements.
pub fn state_to_elements(sv: &StateVector, mu: f64) -> Result<OrbitalElements, OrbitalError> {
    if mu <= 0.0 {
        return Err(OrbitalError::NonPositiveMu(mu));
    }
    let r = sv.position;
    let v = sv.velocity;
    let r_mag = r.magnitude();
    let v_mag = v.magnitude();

    let h = r.cross(v); // specific angular momentum
    let h_mag = h.magnitude();

    let n = Vec3::new(-h.y, h.x, 0.0); // node vector
    let n_mag = n.magnitude();

    // eccentricity vector
    let e_vec = Vec3::new(
        (v_mag * v_mag - mu / r_mag) * r.x / mu - r.dot(v) * v.x / mu,
        (v_mag * v_mag - mu / r_mag) * r.y / mu - r.dot(v) * v.y / mu,
        (v_mag * v_mag - mu / r_mag) * r.z / mu - r.dot(v) * v.z / mu,
    );
    let e = e_vec.magnitude();

    let energy = v_mag * v_mag / 2.0 - mu / r_mag;
    let a = -mu / (2.0 * energy);

    let i = (h.z / h_mag).acos();

    let mut raan = 0.0;
    if n_mag > 1e-12 {
        raan = (n.x / n_mag).acos();
        if n.y < 0.0 {
            raan = 2.0 * PI - raan;
        }
    }

    let mut omega = 0.0;
    if n_mag > 1e-12 && e > 1e-12 {
        let cos_omega = n.dot(e_vec) / (n_mag * e);
        omega = cos_omega.clamp(-1.0, 1.0).acos();
        if e_vec.z < 0.0 {
            omega = 2.0 * PI - omega;
        }
    }

    let mut nu = 0.0;
    if e > 1e-12 {
        let cos_nu = e_vec.dot(r) / (e * r_mag);
        nu = cos_nu.clamp(-1.0, 1.0).acos();
        if r.dot(v) < 0.0 {
            nu = 2.0 * PI - nu;
        }
    }

    Ok(OrbitalElements {
        semi_major_axis: a,
        eccentricity: e.min(0.9999999),
        inclination: i,
        raan,
        arg_periapsis: omega,
        true_anomaly: nu,
    })
}

// ── Kepler's Equation ───────────────────────────────────────────

/// Solve Kepler's equation: M = E - e*sin(E) for E given M and e.
/// Uses Newton-Raphson iteration.
pub fn solve_kepler(mean_anomaly: f64, eccentricity: f64, tol: f64, max_iter: u32) -> Result<f64, OrbitalError> {
    let m = mean_anomaly % (2.0 * PI);
    let mut ea = if eccentricity < 0.8 { m } else { PI };
    for _ in 0..max_iter {
        let f = ea - eccentricity * ea.sin() - m;
        let fp = 1.0 - eccentricity * ea.cos();
        if fp.abs() < 1e-30 {
            break;
        }
        let delta = f / fp;
        ea -= delta;
        if delta.abs() < tol {
            return Ok(ea);
        }
    }
    Err(OrbitalError::KeplerNoConverge { max_iter })
}

/// Convert eccentric anomaly to true anomaly.
pub fn eccentric_to_true(ea: f64, eccentricity: f64) -> f64 {
    let beta = eccentricity / (1.0 + (1.0 - eccentricity * eccentricity).sqrt());
    ea + 2.0 * (beta * ea.sin() / (1.0 - beta * ea.cos())).atan()
}

// ── Vis-Viva ────────────────────────────────────────────────────

/// Vis-viva equation: v = sqrt(mu * (2/r - 1/a)).
pub fn vis_viva(mu: f64, r: f64, a: f64) -> f64 {
    (mu * (2.0 / r - 1.0 / a)).sqrt()
}

// ── Hohmann Transfer ────────────────────────────────────────────

/// Delta-v for a Hohmann transfer between two circular orbits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HohmannTransfer {
    /// Delta-v for departure burn (at r1).
    pub dv1: f64,
    /// Delta-v for arrival burn (at r2).
    pub dv2: f64,
    /// Total delta-v.
    pub total_dv: f64,
    /// Transfer time (half the transfer orbit period).
    pub transfer_time: f64,
    /// Semi-major axis of the transfer orbit.
    pub transfer_sma: f64,
}

/// Compute Hohmann transfer parameters.
pub fn hohmann_transfer(r1: f64, r2: f64, mu: f64) -> Result<HohmannTransfer, OrbitalError> {
    if r1 <= 0.0 || r2 <= 0.0 {
        return Err(OrbitalError::InvalidTransfer("radii must be positive".to_string()));
    }
    if mu <= 0.0 {
        return Err(OrbitalError::NonPositiveMu(mu));
    }
    let a_transfer = (r1 + r2) / 2.0;
    let v_circ1 = (mu / r1).sqrt();
    let v_circ2 = (mu / r2).sqrt();
    let v_transfer_peri = vis_viva(mu, r1, a_transfer);
    let v_transfer_apo = vis_viva(mu, r2, a_transfer);

    let dv1 = (v_transfer_peri - v_circ1).abs();
    let dv2 = (v_circ2 - v_transfer_apo).abs();
    let transfer_time = PI * (a_transfer.powi(3) / mu).sqrt();

    Ok(HohmannTransfer { dv1, dv2, total_dv: dv1 + dv2, transfer_time, transfer_sma: a_transfer })
}

// ── Orbit Propagation ───────────────────────────────────────────

/// Propagate orbital elements forward by a given time.
pub fn propagate(oe: &OrbitalElements, mu: f64, dt: f64) -> Result<OrbitalElements, OrbitalError> {
    if mu <= 0.0 {
        return Err(OrbitalError::NonPositiveMu(mu));
    }
    if dt < 0.0 {
        return Err(OrbitalError::NegativeTime(dt));
    }
    let n = oe.mean_motion(mu);
    let ma0 = oe.true_to_mean_anomaly();
    let ma1 = (ma0 + n * dt) % (2.0 * PI);
    let ea1 = solve_kepler(ma1, oe.eccentricity, 1e-12, 100)?;
    let nu1 = eccentric_to_true(ea1, oe.eccentricity);
    let mut result = *oe;
    result.true_anomaly = nu1;
    Ok(result)
}

// ── Escape Velocity ─────────────────────────────────────────────

/// Escape velocity at distance r from a body with parameter mu.
pub fn escape_velocity(mu: f64, r: f64) -> f64 {
    (2.0 * mu / r).sqrt()
}

/// Circular orbit velocity at distance r.
pub fn circular_velocity(mu: f64, r: f64) -> f64 {
    (mu / r).sqrt()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3, eps: f64) -> bool {
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
    }

    #[test]
    fn elements_validation() {
        assert!(OrbitalElements::new(1.0, 0.5, 0.0, 0.0, 0.0, 0.0).is_ok());
        assert!(OrbitalElements::new(-1.0, 0.5, 0.0, 0.0, 0.0, 0.0).is_err());
        assert!(OrbitalElements::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0).is_err());
        assert!(OrbitalElements::new(1.0, -0.1, 0.0, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn periapsis_apoapsis() {
        let oe = OrbitalElements::new(10.0, 0.5, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert!(approx_eq(oe.periapsis(), 5.0, 1e-10));
        assert!(approx_eq(oe.apoapsis(), 15.0, 1e-10));
    }

    #[test]
    fn period_computation() {
        // mu=1, a=1 => T = 2*pi
        let oe = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert!(approx_eq(oe.period(1.0).unwrap(), 2.0 * PI, 1e-10));
    }

    #[test]
    fn period_negative_mu() {
        let oe = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert!(oe.period(-1.0).is_err());
    }

    #[test]
    fn kepler_equation_circular() {
        // For e=0, E = M.
        let ea = solve_kepler(1.0, 0.0, 1e-12, 100).unwrap();
        assert!(approx_eq(ea, 1.0, 1e-10));
    }

    #[test]
    fn kepler_equation_eccentric() {
        // Check M = E - e*sin(E)
        let e = 0.6;
        let m = 1.5;
        let ea = solve_kepler(m, e, 1e-12, 100).unwrap();
        let m_check = ea - e * ea.sin();
        assert!(approx_eq(m_check, m, 1e-10));
    }

    #[test]
    fn kepler_high_eccentricity() {
        let e = 0.95;
        let m = 0.1;
        let ea = solve_kepler(m, e, 1e-12, 200).unwrap();
        let m_check = ea - e * ea.sin();
        assert!(approx_eq(m_check, m, 1e-8));
    }

    #[test]
    fn circular_orbit_state_vector() {
        // Circular orbit: e=0, a=1, mu=1, at nu=0
        let oe = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        let sv = elements_to_state(&oe, 1.0).unwrap();
        assert!(approx_eq(sv.position.magnitude(), 1.0, 1e-10));
        assert!(approx_eq(sv.velocity.magnitude(), 1.0, 1e-10));
    }

    #[test]
    fn roundtrip_elements_state() {
        let oe = OrbitalElements::new(2.0, 0.3, 0.5, 1.0, 0.7, 0.8).unwrap();
        let mu = 1.0;
        let sv = elements_to_state(&oe, mu).unwrap();
        let oe2 = state_to_elements(&sv, mu).unwrap();
        assert!(approx_eq(oe.semi_major_axis, oe2.semi_major_axis, 1e-6));
        assert!(approx_eq(oe.eccentricity, oe2.eccentricity, 1e-6));
        assert!(approx_eq(oe.inclination, oe2.inclination, 1e-6));
    }

    #[test]
    fn vis_viva_circular() {
        // v = sqrt(mu/r) for circular (a = r)
        let v = vis_viva(1.0, 1.0, 1.0);
        assert!(approx_eq(v, 1.0, 1e-10));
    }

    #[test]
    fn vis_viva_escape() {
        // At r=a, v = sqrt(mu/a). At r=a for parabola (a->inf), v = sqrt(2*mu/r).
        let r = 1.0;
        let v_esc = escape_velocity(1.0, r);
        let v_circ = circular_velocity(1.0, r);
        assert!(approx_eq(v_esc, v_circ * 2.0_f64.sqrt(), 1e-10));
    }

    #[test]
    fn hohmann_same_orbit() {
        let h = hohmann_transfer(1.0, 1.0, 1.0).unwrap();
        assert!(approx_eq(h.total_dv, 0.0, 1e-10));
    }

    #[test]
    fn hohmann_leo_to_geo() {
        // Approximate: r1=6578, r2=42164, mu=398600.4 (Earth, km/s units)
        let h = hohmann_transfer(6578.0, 42164.0, 398600.4).unwrap();
        assert!(h.dv1 > 2.0 && h.dv1 < 3.0); // ~2.46 km/s
        assert!(h.dv2 > 1.0 && h.dv2 < 2.0); // ~1.47 km/s
        assert!(h.total_dv > 3.0 && h.total_dv < 5.0);
    }

    #[test]
    fn hohmann_invalid() {
        assert!(hohmann_transfer(-1.0, 1.0, 1.0).is_err());
        assert!(hohmann_transfer(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn propagation_full_period() {
        let oe = OrbitalElements::new(1.0, 0.3, 0.0, 0.0, 0.0, 0.0).unwrap();
        let mu = 1.0;
        let period = oe.period(mu).unwrap();
        let oe2 = propagate(&oe, mu, period).unwrap();
        // After one full period, true anomaly should return near original.
        let diff = (oe2.true_anomaly - oe.true_anomaly).abs();
        let diff_mod = diff.min((2.0 * PI - diff).abs());
        assert!(diff_mod < 0.01, "propagation drift: {diff_mod}");
    }

    #[test]
    fn propagation_quarter_period() {
        let oe = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        let mu = 1.0;
        let period = oe.period(mu).unwrap();
        let oe2 = propagate(&oe, mu, period / 4.0).unwrap();
        // For circular orbit, quarter period => 90 degrees.
        assert!(approx_eq(oe2.true_anomaly, PI / 2.0, 0.01));
    }

    #[test]
    fn propagation_negative_time() {
        let oe = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert!(propagate(&oe, 1.0, -1.0).is_err());
    }

    #[test]
    fn specific_energy() {
        let oe = OrbitalElements::new(2.0, 0.5, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert!(approx_eq(oe.specific_energy(1.0), -0.25, 1e-10));
    }

    #[test]
    fn mean_motion() {
        let oe = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        // n = sqrt(1/1) = 1.0
        assert!(approx_eq(oe.mean_motion(1.0), 1.0, 1e-10));
    }

    #[test]
    fn eccentric_anomaly_roundtrip() {
        let oe = OrbitalElements::new(1.0, 0.5, 0.0, 0.0, 0.0, 1.2).unwrap();
        let ea = oe.true_to_eccentric_anomaly();
        let nu = eccentric_to_true(ea, 0.5);
        assert!(approx_eq(nu, 1.2, 1e-6));
    }

    #[test]
    fn semi_latus_rectum() {
        let oe = OrbitalElements::new(4.0, 0.5, 0.0, 0.0, 0.0, 0.0).unwrap();
        // p = 4 * (1 - 0.25) = 3.0
        assert!(approx_eq(oe.semi_latus_rectum(), 3.0, 1e-10));
    }

    #[test]
    fn state_to_elements_negative_mu() {
        let sv = StateVector { position: Vec3::new(1.0, 0.0, 0.0), velocity: Vec3::new(0.0, 1.0, 0.0) };
        assert!(state_to_elements(&sv, -1.0).is_err());
    }

    #[test]
    fn escape_and_circular_velocity() {
        let v_esc = escape_velocity(398600.4, 6578.0);
        let v_circ = circular_velocity(398600.4, 6578.0);
        assert!(approx_eq(v_esc / v_circ, 2.0_f64.sqrt(), 1e-6));
    }

    #[test]
    fn hohmann_transfer_sma() {
        let h = hohmann_transfer(1.0, 3.0, 1.0).unwrap();
        assert!(approx_eq(h.transfer_sma, 2.0, 1e-10));
    }
}
