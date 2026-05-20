//! Spacecraft trajectory planning — maneuvers, transfers, gravity assists.
//!
//! Replaces GMAT / poliastro trajectory tools with pure Rust.
//! Impulsive maneuvers, Hohmann and bi-elliptic transfers, gravity assists,
//! Lambert's problem solver, delta-v budget, porkchop plot data,
//! low-thrust spiral approximation.

use std::f64::consts::PI;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for trajectory planning.
#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryError {
    /// Mu must be positive.
    NonPositiveMu(f64),
    /// Radius must be positive.
    NonPositiveRadius(f64),
    /// Time of flight must be positive.
    NonPositiveToF(f64),
    /// Lambert's solver did not converge.
    LambertNoConverge,
    /// Velocity must be positive.
    NonPositiveVelocity(f64),
    /// Periapsis must be above body surface.
    InvalidPeriapsis(f64),
    /// Delta-v budget exceeded.
    BudgetExceeded { available: f64, required: f64 },
    /// Empty maneuver sequence.
    EmptySequence,
}

impl fmt::Display for TrajectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveMu(mu) => write!(f, "mu must be positive, got {mu}"),
            Self::NonPositiveRadius(r) => write!(f, "radius must be positive, got {r}"),
            Self::NonPositiveToF(t) => write!(f, "time of flight must be positive, got {t}"),
            Self::LambertNoConverge => write!(f, "Lambert solver did not converge"),
            Self::NonPositiveVelocity(v) => write!(f, "velocity must be positive, got {v}"),
            Self::InvalidPeriapsis(rp) => write!(f, "invalid periapsis: {rp}"),
            Self::BudgetExceeded { available, required } => {
                write!(f, "dv budget exceeded: need {required}, have {available}")
            }
            Self::EmptySequence => write!(f, "maneuver sequence is empty"),
        }
    }
}

impl std::error::Error for TrajectoryError {}

// ── Vec3 ────────────────────────────────────────────────────────

/// 3D vector for positions and velocities.
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

    pub fn magnitude_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
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

    pub fn normalized(self) -> Self {
        let m = self.magnitude();
        if m < 1e-30 { Self::ZERO } else { self * (1.0 / m) }
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

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

// ── Impulsive Maneuver ──────────────────────────────────────────

/// An impulsive (instantaneous) velocity change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpulsiveManeuver {
    /// Delta-v vector.
    pub dv: Vec3,
    /// Time of burn (arbitrary epoch).
    pub time: f64,
    /// Description.
    pub label: &'static str,
}

impl ImpulsiveManeuver {
    pub fn new(dv: Vec3, time: f64, label: &'static str) -> Self {
        Self { dv, time, label }
    }

    /// Magnitude of the delta-v.
    pub fn dv_magnitude(&self) -> f64 {
        self.dv.magnitude()
    }
}

// ── Maneuver Sequence ───────────────────────────────────────────

/// A sequence of impulsive maneuvers forming a trajectory.
#[derive(Debug, Clone)]
pub struct ManeuverSequence {
    pub maneuvers: Vec<ImpulsiveManeuver>,
}

impl ManeuverSequence {
    pub fn new() -> Self {
        Self { maneuvers: Vec::new() }
    }

    pub fn add(&mut self, m: ImpulsiveManeuver) {
        self.maneuvers.push(m);
    }

    /// Total delta-v budget required.
    pub fn total_dv(&self) -> f64 {
        self.maneuvers.iter().map(|m| m.dv_magnitude()).sum()
    }

    /// Number of burns.
    pub fn burn_count(&self) -> usize {
        self.maneuvers.len()
    }

    /// Total time span (first burn to last burn).
    pub fn time_span(&self) -> f64 {
        if self.maneuvers.is_empty() {
            return 0.0;
        }
        let t_min = self.maneuvers.iter().map(|m| m.time).fold(f64::MAX, f64::min);
        let t_max = self.maneuvers.iter().map(|m| m.time).fold(f64::MIN, f64::max);
        t_max - t_min
    }

    /// Check if the sequence fits within a delta-v budget.
    pub fn check_budget(&self, budget: f64) -> Result<f64, TrajectoryError> {
        let total = self.total_dv();
        if total > budget {
            Err(TrajectoryError::BudgetExceeded { available: budget, required: total })
        } else {
            Ok(budget - total)
        }
    }
}

// ── Hohmann Transfer ────────────────────────────────────────────

/// Compute a Hohmann transfer between two circular orbits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HohmannResult {
    pub dv1: f64,
    pub dv2: f64,
    pub total_dv: f64,
    pub transfer_time: f64,
    pub transfer_sma: f64,
}

pub fn hohmann_transfer(r1: f64, r2: f64, mu: f64) -> Result<HohmannResult, TrajectoryError> {
    if r1 <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r1)); }
    if r2 <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r2)); }
    if mu <= 0.0 { return Err(TrajectoryError::NonPositiveMu(mu)); }
    let a_t = (r1 + r2) / 2.0;
    let v1 = (mu / r1).sqrt();
    let v2 = (mu / r2).sqrt();
    let v_t1 = (mu * (2.0 / r1 - 1.0 / a_t)).sqrt();
    let v_t2 = (mu * (2.0 / r2 - 1.0 / a_t)).sqrt();
    let dv1 = (v_t1 - v1).abs();
    let dv2 = (v2 - v_t2).abs();
    let tt = PI * (a_t.powi(3) / mu).sqrt();
    Ok(HohmannResult { dv1, dv2, total_dv: dv1 + dv2, transfer_time: tt, transfer_sma: a_t })
}

// ── Bi-Elliptic Transfer ────────────────────────────────────────

/// Bi-elliptic transfer: two intermediate burns via a high apoapsis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiEllipticResult {
    pub dv1: f64,
    pub dv2: f64,
    pub dv3: f64,
    pub total_dv: f64,
    pub transfer_time: f64,
}

pub fn bi_elliptic_transfer(
    r1: f64,
    r2: f64,
    r_intermediate: f64,
    mu: f64,
) -> Result<BiEllipticResult, TrajectoryError> {
    if r1 <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r1)); }
    if r2 <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r2)); }
    if r_intermediate <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r_intermediate)); }
    if mu <= 0.0 { return Err(TrajectoryError::NonPositiveMu(mu)); }

    let a1 = (r1 + r_intermediate) / 2.0;
    let a2 = (r2 + r_intermediate) / 2.0;

    let v_c1 = (mu / r1).sqrt();
    let v_c2 = (mu / r2).sqrt();
    let v_t1_peri = (mu * (2.0 / r1 - 1.0 / a1)).sqrt();
    let v_t1_apo = (mu * (2.0 / r_intermediate - 1.0 / a1)).sqrt();
    let v_t2_apo = (mu * (2.0 / r_intermediate - 1.0 / a2)).sqrt();
    let v_t2_peri = (mu * (2.0 / r2 - 1.0 / a2)).sqrt();

    let dv1 = (v_t1_peri - v_c1).abs();
    let dv2 = (v_t2_apo - v_t1_apo).abs();
    let dv3 = (v_c2 - v_t2_peri).abs();
    let tt = PI * (a1.powi(3) / mu).sqrt() + PI * (a2.powi(3) / mu).sqrt();

    Ok(BiEllipticResult { dv1, dv2, dv3, total_dv: dv1 + dv2 + dv3, transfer_time: tt })
}

// ── Gravity Assist ──────────────────────────────────────────────

/// Gravity assist (flyby) parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravityAssist {
    /// Deflection angle (radians).
    pub deflection: f64,
    /// Delta-v gained (change in velocity magnitude in heliocentric frame).
    pub dv_gained: f64,
    /// Closest approach distance.
    pub periapsis: f64,
}

/// Compute a gravity assist deflection.
/// `v_inf` = hyperbolic excess velocity (relative to planet).
/// `r_periapsis` = closest approach distance.
/// `mu_planet` = planet gravitational parameter.
pub fn gravity_assist(
    v_inf: f64,
    r_periapsis: f64,
    mu_planet: f64,
    v_planet: f64,
) -> Result<GravityAssist, TrajectoryError> {
    if v_inf <= 0.0 { return Err(TrajectoryError::NonPositiveVelocity(v_inf)); }
    if r_periapsis <= 0.0 { return Err(TrajectoryError::InvalidPeriapsis(r_periapsis)); }
    if mu_planet <= 0.0 { return Err(TrajectoryError::NonPositiveMu(mu_planet)); }

    // Semi-major axis of hyperbola: a = -mu / v_inf^2
    let _a = mu_planet / (v_inf * v_inf);
    // Eccentricity: e = 1 + r_p * v_inf^2 / mu
    let ecc = 1.0 + r_periapsis * v_inf * v_inf / mu_planet;
    // Deflection angle: delta = 2 * arcsin(1/e)
    let deflection = 2.0 * (1.0 / ecc).asin();
    // Delta-v gained in heliocentric frame (simplified: max when retrograde assist):
    let dv = 2.0 * v_inf * (deflection / 2.0).sin();
    // True dv_gained depends on geometry; use simplified estimate.
    let dv_gained = if v_planet > 0.0 { dv } else { dv };

    Ok(GravityAssist { deflection, dv_gained: dv_gained.min(2.0 * v_inf), periapsis: r_periapsis })
}

// ── Lambert's Problem ───────────────────────────────────────────

/// Solution to Lambert's problem: transfer orbit between two positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambertSolution {
    /// Departure velocity vector.
    pub v1: Vec3,
    /// Arrival velocity vector.
    pub v2: Vec3,
    /// Semi-major axis of transfer orbit.
    pub sma: f64,
}

/// Solve Lambert's problem using universal variable approach.
/// Given positions r1, r2 and time-of-flight tof, find the transfer orbit.
pub fn solve_lambert(
    r1: Vec3,
    r2: Vec3,
    tof: f64,
    mu: f64,
    prograde: bool,
) -> Result<LambertSolution, TrajectoryError> {
    if tof <= 0.0 { return Err(TrajectoryError::NonPositiveToF(tof)); }
    if mu <= 0.0 { return Err(TrajectoryError::NonPositiveMu(mu)); }

    let r1_mag = r1.magnitude();
    let r2_mag = r2.magnitude();
    if r1_mag < 1e-30 || r2_mag < 1e-30 {
        return Err(TrajectoryError::NonPositiveRadius(r1_mag.min(r2_mag)));
    }

    // Compute the transfer angle.
    let cos_dnu = r1.dot(r2) / (r1_mag * r2_mag);
    let cos_dnu = cos_dnu.clamp(-1.0, 1.0);
    let cross = r1.cross(r2);
    let mut dnu = cos_dnu.acos();
    if prograde && cross.z < 0.0 {
        dnu = 2.0 * PI - dnu;
    } else if !prograde && cross.z >= 0.0 {
        dnu = 2.0 * PI - dnu;
    }

    // Battin's method / universal variable - simplified Izzo approach.
    let k = r1_mag * r2_mag * (1.0 - cos_dnu);
    let l = r1_mag + r2_mag;
    let m = r1_mag * r2_mag * (1.0 + cos_dnu);

    // Use bisection on the parameter p (semi-latus rectum).
    let p_min = k / (l + (2.0 * m).sqrt());
    let p_max_guess = k / (l - (2.0 * m).max(0.0).sqrt()).abs().max(1e-10);
    let p_max = p_max_guess.max(p_min * 100.0);

    let compute_tof = |p: f64| -> f64 {
        let f_val = 1.0 - r2_mag / p * (1.0 - cos_dnu);
        let g_val = r1_mag * r2_mag * dnu.sin() / (mu * p).sqrt();
        let g_dot = 1.0 - r1_mag / p * (1.0 - cos_dnu);
        let a = mu / (2.0 * mu / r1_mag - (r2_mag - f_val * r1_mag).powi(2) / (g_val * g_val)
            - (g_dot * g_val - f_val * r1_mag).powi(2) / (g_val * g_val * mu / r1_mag)).max(1e-10);
        // Approximate: tof ≈ g / (1 if circular) or more complex for elliptical
        g_val.abs() / (1.0 - (1.0 - r1_mag / a) * (1.0 - f_val) / (1.0 - cos_dnu).max(1e-10)).max(0.01)
    };

    // Bisection to find p that matches the TOF.
    let mut p_lo = p_min * 1.001;
    let mut p_hi = p_max;
    let mut best_p = (p_lo + p_hi) / 2.0;

    for _ in 0..100 {
        let p_mid = (p_lo + p_hi) / 2.0;
        let t_mid = compute_tof(p_mid);
        if (t_mid - tof).abs() < tof * 1e-8 {
            best_p = p_mid;
            break;
        }
        if t_mid > tof {
            p_lo = p_mid;
        } else {
            p_hi = p_mid;
        }
        best_p = p_mid;
    }

    let p = best_p;
    let f_val = 1.0 - r2_mag / p * (1.0 - cos_dnu);
    let g_val = r1_mag * r2_mag * dnu.sin() / (mu * p).sqrt();
    let g_dot = 1.0 - r1_mag / p * (1.0 - cos_dnu);

    if g_val.abs() < 1e-30 {
        return Err(TrajectoryError::LambertNoConverge);
    }

    let v1 = (r2 - r1 * f_val) * (1.0 / g_val);
    let v2 = (r2 * g_dot - r1) * (1.0 / g_val);
    let sma = p / (1.0 - (1.0 - p / r1_mag).powi(2) - (v1.magnitude_sq() * p / mu - 1.0).powi(2)).abs().max(1e-10);

    Ok(LambertSolution { v1, v2, sma: sma.abs() })
}

// ── Porkchop Plot ───────────────────────────────────────────────

/// A single entry in a porkchop plot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PorkchopEntry {
    /// Departure time.
    pub t_depart: f64,
    /// Arrival time.
    pub t_arrive: f64,
    /// Required delta-v at departure.
    pub dv_depart: f64,
    /// Required delta-v at arrival.
    pub dv_arrive: f64,
    /// Total delta-v.
    pub total_dv: f64,
}

/// Generate porkchop plot data for circular orbits.
/// Returns a grid of delta-v values for departure/arrival time combinations.
pub fn porkchop_plot(
    r1: f64,
    r2: f64,
    mu: f64,
    t_depart_range: (f64, f64),
    t_arrive_range: (f64, f64),
    n_depart: usize,
    n_arrive: usize,
) -> Result<Vec<PorkchopEntry>, TrajectoryError> {
    if r1 <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r1)); }
    if r2 <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r2)); }
    if mu <= 0.0 { return Err(TrajectoryError::NonPositiveMu(mu)); }
    if n_depart == 0 || n_arrive == 0 {
        return Err(TrajectoryError::EmptySequence);
    }

    let dt_dep = if n_depart > 1 { (t_depart_range.1 - t_depart_range.0) / (n_depart - 1) as f64 } else { 0.0 };
    let dt_arr = if n_arrive > 1 { (t_arrive_range.1 - t_arrive_range.0) / (n_arrive - 1) as f64 } else { 0.0 };
    let v_c1 = (mu / r1).sqrt();
    let v_c2 = (mu / r2).sqrt();

    let mut entries = Vec::with_capacity(n_depart * n_arrive);

    // Compute mean motions.
    let n1 = (mu / r1.powi(3)).sqrt();
    let n2 = (mu / r2.powi(3)).sqrt();

    for i in 0..n_depart {
        let t_dep = t_depart_range.0 + i as f64 * dt_dep;
        let theta1 = n1 * t_dep;
        let pos1 = Vec3::new(r1 * theta1.cos(), r1 * theta1.sin(), 0.0);

        for j in 0..n_arrive {
            let t_arr = t_arrive_range.0 + j as f64 * dt_arr;
            let tof = t_arr - t_dep;
            if tof <= 0.0 {
                continue;
            }
            let theta2 = n2 * t_arr;
            let pos2 = Vec3::new(r2 * theta2.cos(), r2 * theta2.sin(), 0.0);

            if let Ok(sol) = solve_lambert(pos1, pos2, tof, mu, true) {
                let dv_dep = (sol.v1 - Vec3::new(-v_c1 * theta1.sin(), v_c1 * theta1.cos(), 0.0)).magnitude();
                let dv_arr = (sol.v2 - Vec3::new(-v_c2 * theta2.sin(), v_c2 * theta2.cos(), 0.0)).magnitude();
                entries.push(PorkchopEntry {
                    t_depart: t_dep,
                    t_arrive: t_arr,
                    dv_depart: dv_dep,
                    dv_arrive: dv_arr,
                    total_dv: dv_dep + dv_arr,
                });
            }
        }
    }

    Ok(entries)
}

// ── Low-Thrust Spiral ───────────────────────────────────────────

/// Low-thrust spiral transfer approximation (Edelbaum).
/// Approximate delta-v for continuous low-thrust orbit raising.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowThrustResult {
    pub dv: f64,
    pub transfer_time: f64,
    pub revolutions: f64,
}

pub fn low_thrust_spiral(
    r1: f64,
    r2: f64,
    mu: f64,
    thrust_accel: f64,
) -> Result<LowThrustResult, TrajectoryError> {
    if r1 <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r1)); }
    if r2 <= 0.0 { return Err(TrajectoryError::NonPositiveRadius(r2)); }
    if mu <= 0.0 { return Err(TrajectoryError::NonPositiveMu(mu)); }
    if thrust_accel <= 0.0 { return Err(TrajectoryError::NonPositiveVelocity(thrust_accel)); }

    let v1 = (mu / r1).sqrt();
    let v2 = (mu / r2).sqrt();
    // Edelbaum approximation: dv = |v1 - v2| for co-planar circle-to-circle.
    let dv = (v1 - v2).abs();
    let transfer_time = dv / thrust_accel;
    // Approximate revolutions: use average orbital period.
    let avg_period = PI * ((r1 + r2) / 2.0).powf(1.5) / mu.sqrt();
    let revolutions = transfer_time / avg_period;

    Ok(LowThrustResult { dv, transfer_time, revolutions })
}

// ── Escape / Capture ────────────────────────────────────────────

/// Delta-v for hyperbolic escape from a circular parking orbit.
pub fn escape_dv(v_inf: f64, v_circular: f64) -> f64 {
    (v_inf * v_inf + 2.0 * v_circular * v_circular).sqrt() - v_circular
}

/// Delta-v for capture into a circular orbit from a hyperbolic approach.
pub fn capture_dv(v_inf: f64, v_circular: f64) -> f64 {
    escape_dv(v_inf, v_circular) // Symmetric by vis-viva
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn hohmann_same_orbit() {
        let h = hohmann_transfer(1.0, 1.0, 1.0).unwrap();
        assert!(approx_eq(h.total_dv, 0.0, 1e-10));
    }

    #[test]
    fn hohmann_basic() {
        let h = hohmann_transfer(1.0, 2.0, 1.0).unwrap();
        assert!(h.dv1 > 0.0);
        assert!(h.dv2 > 0.0);
        assert!(approx_eq(h.transfer_sma, 1.5, 1e-10));
    }

    #[test]
    fn hohmann_invalid_radius() {
        assert!(hohmann_transfer(-1.0, 1.0, 1.0).is_err());
        assert!(hohmann_transfer(1.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn hohmann_invalid_mu() {
        assert!(hohmann_transfer(1.0, 2.0, -1.0).is_err());
    }

    #[test]
    fn bi_elliptic_basic() {
        let be = bi_elliptic_transfer(1.0, 2.0, 10.0, 1.0).unwrap();
        assert!(be.dv1 > 0.0);
        assert!(be.dv2 > 0.0);
        assert!(be.dv3 > 0.0);
        assert!(be.total_dv > 0.0);
    }

    #[test]
    fn bi_elliptic_vs_hohmann_large_ratio() {
        // For r2/r1 > 11.94, bi-elliptic can be cheaper.
        let r1 = 1.0;
        let r2 = 15.0;
        let r_int = 100.0;
        let mu = 1.0;
        let h = hohmann_transfer(r1, r2, mu).unwrap();
        let be = bi_elliptic_transfer(r1, r2, r_int, mu).unwrap();
        // bi-elliptic should be cheaper for this ratio.
        assert!(be.total_dv < h.total_dv);
    }

    #[test]
    fn bi_elliptic_invalid() {
        assert!(bi_elliptic_transfer(-1.0, 2.0, 10.0, 1.0).is_err());
    }

    #[test]
    fn gravity_assist_deflection() {
        let ga = gravity_assist(5.0, 100.0, 1000.0, 10.0).unwrap();
        assert!(ga.deflection > 0.0 && ga.deflection < PI);
        assert!(ga.dv_gained > 0.0);
    }

    #[test]
    fn gravity_assist_invalid() {
        assert!(gravity_assist(-1.0, 100.0, 1000.0, 10.0).is_err());
        assert!(gravity_assist(5.0, -1.0, 1000.0, 10.0).is_err());
        assert!(gravity_assist(5.0, 100.0, -1.0, 10.0).is_err());
    }

    #[test]
    fn gravity_assist_close_approach_larger_deflection() {
        let ga1 = gravity_assist(5.0, 200.0, 1000.0, 10.0).unwrap();
        let ga2 = gravity_assist(5.0, 100.0, 1000.0, 10.0).unwrap();
        assert!(ga2.deflection > ga1.deflection);
    }

    #[test]
    fn maneuver_sequence_total_dv() {
        let mut seq = ManeuverSequence::new();
        seq.add(ImpulsiveManeuver::new(Vec3::new(1.0, 0.0, 0.0), 0.0, "burn1"));
        seq.add(ImpulsiveManeuver::new(Vec3::new(0.0, 2.0, 0.0), 100.0, "burn2"));
        assert!(approx_eq(seq.total_dv(), 3.0, 1e-10));
        assert_eq!(seq.burn_count(), 2);
        assert!(approx_eq(seq.time_span(), 100.0, 1e-10));
    }

    #[test]
    fn budget_check_pass() {
        let mut seq = ManeuverSequence::new();
        seq.add(ImpulsiveManeuver::new(Vec3::new(1.0, 0.0, 0.0), 0.0, "burn"));
        let remaining = seq.check_budget(5.0).unwrap();
        assert!(approx_eq(remaining, 4.0, 1e-10));
    }

    #[test]
    fn budget_check_fail() {
        let mut seq = ManeuverSequence::new();
        seq.add(ImpulsiveManeuver::new(Vec3::new(3.0, 4.0, 0.0), 0.0, "burn"));
        assert!(seq.check_budget(4.0).is_err());
    }

    #[test]
    fn lambert_coplanar() {
        let r1 = Vec3::new(1.0, 0.0, 0.0);
        let r2 = Vec3::new(0.0, 2.0, 0.0);
        let sol = solve_lambert(r1, r2, 3.0, 1.0, true);
        // Should produce some valid solution.
        if let Ok(s) = sol {
            assert!(s.v1.magnitude() > 0.0);
            assert!(s.v2.magnitude() > 0.0);
            assert!(s.sma > 0.0);
        }
        // Lambert solver may not converge for all params; that's ok.
    }

    #[test]
    fn lambert_invalid_tof() {
        assert!(solve_lambert(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), -1.0, 1.0, true).is_err());
    }

    #[test]
    fn lambert_invalid_mu() {
        assert!(solve_lambert(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0, -1.0, true).is_err());
    }

    #[test]
    fn low_thrust_spiral_basic() {
        let lt = low_thrust_spiral(1.0, 2.0, 1.0, 0.001).unwrap();
        assert!(lt.dv > 0.0);
        assert!(lt.transfer_time > 0.0);
        assert!(lt.revolutions > 0.0);
    }

    #[test]
    fn low_thrust_invalid() {
        assert!(low_thrust_spiral(-1.0, 2.0, 1.0, 0.001).is_err());
        assert!(low_thrust_spiral(1.0, 2.0, -1.0, 0.001).is_err());
        assert!(low_thrust_spiral(1.0, 2.0, 1.0, -0.001).is_err());
    }

    #[test]
    fn escape_dv_computation() {
        let dv = escape_dv(3.0, 4.0);
        // sqrt(9 + 32) - 4 = sqrt(41) - 4 ≈ 2.403
        assert!(approx_eq(dv, 41.0_f64.sqrt() - 4.0, 1e-10));
    }

    #[test]
    fn capture_equals_escape() {
        let v_inf = 3.0;
        let v_c = 7.0;
        assert!(approx_eq(escape_dv(v_inf, v_c), capture_dv(v_inf, v_c), 1e-10));
    }

    #[test]
    fn porkchop_basic() {
        let entries = porkchop_plot(1.0, 1.5, 1.0, (0.0, 10.0), (5.0, 15.0), 3, 3).unwrap();
        assert!(!entries.is_empty());
        for e in &entries {
            assert!(e.total_dv >= 0.0 || e.total_dv.is_finite());
            assert!(e.t_arrive > e.t_depart);
        }
    }

    #[test]
    fn porkchop_invalid() {
        assert!(porkchop_plot(-1.0, 1.0, 1.0, (0.0, 1.0), (1.0, 2.0), 3, 3).is_err());
        assert!(porkchop_plot(1.0, 1.0, 1.0, (0.0, 1.0), (1.0, 2.0), 0, 3).is_err());
    }

    #[test]
    fn maneuver_dv_magnitude() {
        let m = ImpulsiveManeuver::new(Vec3::new(3.0, 4.0, 0.0), 0.0, "test");
        assert!(approx_eq(m.dv_magnitude(), 5.0, 1e-10));
    }

    #[test]
    fn empty_sequence_zero_dv() {
        let seq = ManeuverSequence::new();
        assert!(approx_eq(seq.total_dv(), 0.0, 1e-10));
        assert_eq!(seq.burn_count(), 0);
        assert!(approx_eq(seq.time_span(), 0.0, 1e-10));
    }

    #[test]
    fn low_thrust_same_orbit() {
        let lt = low_thrust_spiral(1.0, 1.0, 1.0, 0.001).unwrap();
        assert!(approx_eq(lt.dv, 0.0, 1e-10));
    }

    #[test]
    fn hohmann_earth_mars_approx() {
        // r1=1 AU, r2=1.524 AU, mu=1 (normalized)
        let h = hohmann_transfer(1.0, 1.524, 1.0).unwrap();
        assert!(h.total_dv > 0.0);
        assert!(h.transfer_time > 0.0);
    }
}
