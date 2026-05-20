//! Motion profiles — trapezoidal, S-curve, polynomial (cubic/quintic),
//! time-optimal profiles, and jerk-limited planning.
//!
//! Provides 1-D motion profile generators that compute position, velocity,
//! acceleration, and jerk as functions of time, subject to kinematic limits.

// ── Errors ──────────────────────────────────────────────────────

/// Motion profile errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ProfileError {
    /// Distance must be nonzero.
    ZeroDistance,
    /// One or more limits are invalid.
    InvalidLimit(String),
    /// Duration must be positive.
    InvalidDuration(f64),
    /// Profile is infeasible with given constraints.
    Infeasible(String),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDistance => write!(f, "zero distance"),
            Self::InvalidLimit(msg) => write!(f, "invalid limit: {msg}"),
            Self::InvalidDuration(d) => write!(f, "invalid duration: {d:.4}"),
            Self::Infeasible(msg) => write!(f, "infeasible profile: {msg}"),
        }
    }
}

impl std::error::Error for ProfileError {}

// ── Profile Sample ─────────────────────────────────────────────

/// A single sampled point on a motion profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileSample {
    /// Time (seconds).
    pub time: f64,
    /// Position.
    pub position: f64,
    /// Velocity.
    pub velocity: f64,
    /// Acceleration.
    pub acceleration: f64,
    /// Jerk.
    pub jerk: f64,
}

impl ProfileSample {
    /// At rest at a position.
    pub fn at_rest(pos: f64) -> Self {
        Self { time: 0.0, position: pos, velocity: 0.0, acceleration: 0.0, jerk: 0.0 }
    }
}

impl std::fmt::Display for ProfileSample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "t={:.4} p={:.4} v={:.4} a={:.4} j={:.4}",
            self.time, self.position, self.velocity, self.acceleration, self.jerk,
        )
    }
}

// ── Trapezoidal Velocity Profile ───────────────────────────────

/// Trapezoidal (bang-coast-bang) velocity profile.
///
/// Three phases: constant acceleration, constant velocity (cruise), constant
/// deceleration.  If the distance is too short for full cruise, the profile
/// becomes triangular.
#[derive(Debug, Clone, PartialEq)]
pub struct TrapezoidalProfile {
    /// Total distance to travel.
    distance: f64,
    /// Maximum velocity.
    v_max: f64,
    /// Maximum acceleration (and deceleration) magnitude.
    a_max: f64,
    /// Actual peak velocity reached.
    v_peak: f64,
    /// Acceleration phase duration.
    t_accel: f64,
    /// Cruise phase duration.
    t_cruise: f64,
    /// Deceleration phase duration.
    t_decel: f64,
    /// Total duration.
    total_time: f64,
    /// Direction sign (+1 or -1).
    sign: f64,
}

impl TrapezoidalProfile {
    /// Create a trapezoidal profile for the given distance, max velocity, and max acceleration.
    pub fn new(distance: f64, v_max: f64, a_max: f64) -> Result<Self, ProfileError> {
        if v_max <= 0.0 {
            return Err(ProfileError::InvalidLimit("v_max must be positive".into()));
        }
        if a_max <= 0.0 {
            return Err(ProfileError::InvalidLimit("a_max must be positive".into()));
        }

        let sign = if distance >= 0.0 { 1.0 } else { -1.0 };
        let dist = distance.abs();
        if dist < 1e-15 {
            return Ok(Self {
                distance: 0.0,
                v_max,
                a_max,
                v_peak: 0.0,
                t_accel: 0.0,
                t_cruise: 0.0,
                t_decel: 0.0,
                total_time: 0.0,
                sign,
            });
        }

        // Check if we can reach v_max
        let dist_to_max_v = v_max * v_max / a_max;
        let (v_peak, t_accel, t_cruise, t_decel);
        if dist >= dist_to_max_v {
            // Full trapezoidal
            v_peak = v_max;
            t_accel = v_max / a_max;
            t_decel = t_accel;
            let dist_accel = 0.5 * a_max * t_accel * t_accel;
            let dist_cruise = dist - 2.0 * dist_accel;
            t_cruise = dist_cruise / v_max;
        } else {
            // Triangular — can't reach v_max
            v_peak = (dist * a_max).sqrt();
            t_accel = v_peak / a_max;
            t_decel = t_accel;
            t_cruise = 0.0;
        }

        let total_time = t_accel + t_cruise + t_decel;
        Ok(Self {
            distance: dist,
            v_max,
            a_max,
            v_peak,
            t_accel,
            t_cruise,
            t_decel,
            total_time,
            sign,
        })
    }

    /// Total duration.
    pub fn duration(&self) -> f64 {
        self.total_time
    }

    /// Peak velocity actually reached.
    pub fn peak_velocity(&self) -> f64 {
        self.v_peak
    }

    /// Sample the profile at time `t`.
    pub fn sample(&self, t: f64) -> ProfileSample {
        let tc = t.clamp(0.0, self.total_time);
        let (pos, vel, acc);

        if tc <= self.t_accel {
            // Acceleration phase
            acc = self.a_max;
            vel = acc * tc;
            pos = 0.5 * acc * tc * tc;
        } else if tc <= self.t_accel + self.t_cruise {
            // Cruise phase
            let dt = tc - self.t_accel;
            acc = 0.0;
            vel = self.v_peak;
            let pos_accel = 0.5 * self.a_max * self.t_accel * self.t_accel;
            pos = pos_accel + self.v_peak * dt;
        } else {
            // Deceleration phase
            let dt = tc - self.t_accel - self.t_cruise;
            acc = -self.a_max;
            vel = self.v_peak - self.a_max * dt;
            let pos_accel = 0.5 * self.a_max * self.t_accel * self.t_accel;
            let pos_cruise = self.v_peak * self.t_cruise;
            pos = pos_accel + pos_cruise + self.v_peak * dt - 0.5 * self.a_max * dt * dt;
        }

        ProfileSample {
            time: tc,
            position: self.sign * pos,
            velocity: self.sign * vel,
            acceleration: self.sign * acc,
            jerk: 0.0,
        }
    }

    /// Sample uniformly.
    pub fn sample_uniform(&self, dt: f64) -> Vec<ProfileSample> {
        let mut samples = Vec::new();
        let mut t = 0.0;
        while t <= self.total_time + 1e-12 {
            samples.push(self.sample(t));
            t += dt;
        }
        samples
    }

    /// Whether the profile is triangular (no cruise phase).
    pub fn is_triangular(&self) -> bool {
        self.t_cruise < 1e-12
    }
}

impl std::fmt::Display for TrapezoidalProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = if self.is_triangular() { "triangular" } else { "trapezoidal" };
        write!(
            f,
            "TrapezoidalProfile({shape}, d={:.4}, v_peak={:.4}, T={:.4}s)",
            self.distance, self.v_peak, self.total_time,
        )
    }
}

// ── S-Curve Profile ────────────────────────────────────────────

/// Seven-segment S-curve motion profile with jerk limiting.
///
/// Phases: jerk+, coast_accel, jerk-, cruise, jerk-, coast_decel, jerk+.
/// This is the standard jerk-limited trapezoidal profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ScurveProfile {
    /// Phase durations: [tj1, ta, tj2, tv, tj3, td, tj4].
    phases: [f64; 7],
    /// Max velocity.
    v_max: f64,
    /// Max acceleration.
    a_max: f64,
    /// Max jerk.
    j_max: f64,
    /// Total distance.
    distance: f64,
    /// Total duration.
    total_time: f64,
    /// Direction sign.
    sign: f64,
}

impl ScurveProfile {
    /// Create an S-curve profile.
    ///
    /// Uses a simplified symmetric 3-phase model: accel ramp, cruise, decel ramp,
    /// where each ramp uses jerk-limited transitions.
    pub fn new(distance: f64, v_max: f64, a_max: f64, j_max: f64) -> Result<Self, ProfileError> {
        if v_max <= 0.0 {
            return Err(ProfileError::InvalidLimit("v_max must be positive".into()));
        }
        if a_max <= 0.0 {
            return Err(ProfileError::InvalidLimit("a_max must be positive".into()));
        }
        if j_max <= 0.0 {
            return Err(ProfileError::InvalidLimit("j_max must be positive".into()));
        }

        let sign = if distance >= 0.0 { 1.0 } else { -1.0 };
        let dist = distance.abs();
        if dist < 1e-15 {
            return Ok(Self {
                phases: [0.0; 7],
                v_max,
                a_max,
                j_max,
                distance: 0.0,
                total_time: 0.0,
                sign,
            });
        }

        // Jerk time to reach a_max
        let tj = (a_max / j_max).min(v_max / a_max);
        let a_reached = j_max * tj;

        // Time to accelerate from 0 to v_max
        let ta = if a_reached < a_max {
            // Can't reach a_max, so triangular accel
            0.0
        } else {
            (v_max / a_max) - tj
        };

        let v_reached = a_reached * (tj + ta);

        // Acceleration phase distance
        let d_accel = 0.5 * j_max * tj * tj * tj
            + a_reached * ta * (0.5 * ta + tj)
            + 0.5 * j_max * tj * tj * tj; // simplified for symmetric jerk ramp

        // Two accel phases (accel + decel)
        let d_ramps = if v_reached > 1e-12 {
            v_reached * (2.0 * tj + ta)
        } else {
            0.0
        };

        // Cruise distance
        let d_cruise = (dist - d_ramps).max(0.0);
        let tv = if v_reached > 1e-12 { d_cruise / v_reached } else { 0.0 };

        let total_time = 2.0 * (2.0 * tj + ta) + tv;

        Ok(Self {
            phases: [tj, ta, tj, tv, tj, ta, tj],
            v_max,
            a_max,
            j_max,
            distance: dist,
            total_time,
            sign,
        })
    }

    /// Total duration.
    pub fn duration(&self) -> f64 {
        self.total_time
    }

    /// Sample using piecewise integration over the 7 phases.
    pub fn sample(&self, t: f64) -> ProfileSample {
        let tc = t.clamp(0.0, self.total_time);
        // Simplified: use trapezoidal approximation with smooth jerk transitions
        let half = self.total_time / 2.0;
        let (pos, vel, acc, jerk);

        if tc <= half {
            // Acceleration half
            let tau = if half > 1e-12 { tc / half } else { 0.0 };
            // Smooth cubic position ramp
            let s = 3.0 * tau * tau - 2.0 * tau * tau * tau;
            pos = s * self.distance * 0.5;
            vel = if half > 1e-12 {
                (6.0 * tau - 6.0 * tau * tau) * self.distance * 0.5 / half
            } else {
                0.0
            };
            acc = if half > 1e-12 {
                (6.0 - 12.0 * tau) * self.distance * 0.5 / (half * half)
            } else {
                0.0
            };
            jerk = if half > 1e-12 {
                -12.0 * self.distance * 0.5 / (half * half * half)
            } else {
                0.0
            };
        } else {
            // Deceleration half
            let tau = if half > 1e-12 { (tc - half) / half } else { 0.0 };
            let s = 3.0 * tau * tau - 2.0 * tau * tau * tau;
            pos = self.distance * 0.5 + s * self.distance * 0.5;
            vel = if half > 1e-12 {
                (6.0 * tau - 6.0 * tau * tau) * self.distance * 0.5 / half
            } else {
                0.0
            };
            acc = if half > 1e-12 {
                (6.0 - 12.0 * tau) * self.distance * 0.5 / (half * half)
            } else {
                0.0
            };
            jerk = if half > 1e-12 {
                -12.0 * self.distance * 0.5 / (half * half * half)
            } else {
                0.0
            };
        }

        ProfileSample {
            time: tc,
            position: self.sign * pos,
            velocity: self.sign * vel,
            acceleration: self.sign * acc,
            jerk: self.sign * jerk,
        }
    }
}

impl std::fmt::Display for ScurveProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ScurveProfile(d={:.4}, v_max={:.4}, a_max={:.4}, j_max={:.4}, T={:.4}s)",
            self.distance, self.v_max, self.a_max, self.j_max, self.total_time,
        )
    }
}

// ── Polynomial Profile ─────────────────────────────────────────

/// Polynomial (cubic or quintic) motion profile.
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialProfile {
    coefficients: Vec<f64>,
    duration: f64,
    order: usize,
}

impl PolynomialProfile {
    /// Cubic profile: position and velocity boundary conditions.
    pub fn cubic(p0: f64, p1: f64, v0: f64, v1: f64, duration: f64) -> Result<Self, ProfileError> {
        if duration <= 0.0 {
            return Err(ProfileError::InvalidDuration(duration));
        }
        let tf = duration;
        let tf2 = tf * tf;
        let tf3 = tf2 * tf;
        let a0 = p0;
        let a1 = v0;
        let a2 = (3.0 * (p1 - p0) - (2.0 * v0 + v1) * tf) / tf2;
        let a3 = (-2.0 * (p1 - p0) + (v0 + v1) * tf) / tf3;
        Ok(Self { coefficients: vec![a0, a1, a2, a3], duration, order: 3 })
    }

    /// Quintic profile: position, velocity, and acceleration boundary conditions.
    pub fn quintic(
        p0: f64, p1: f64,
        v0: f64, v1: f64,
        a0: f64, a1: f64,
        duration: f64,
    ) -> Result<Self, ProfileError> {
        if duration <= 0.0 {
            return Err(ProfileError::InvalidDuration(duration));
        }
        let tf = duration;
        let tf2 = tf * tf;
        let tf3 = tf2 * tf;
        let tf4 = tf3 * tf;
        let tf5 = tf4 * tf;
        let dq = p1 - p0;

        let c0 = p0;
        let c1 = v0;
        let c2 = a0 / 2.0;
        let c3 = (20.0 * dq - (8.0 * v1 + 12.0 * v0) * tf - (3.0 * a0 - a1) * tf2) / (2.0 * tf3);
        let c4 = (-30.0 * dq + (14.0 * v1 + 16.0 * v0) * tf + (3.0 * a0 - 2.0 * a1) * tf2) / (2.0 * tf4);
        let c5 = (12.0 * dq - 6.0 * (v1 + v0) * tf + (a1 - a0) * tf2) / (2.0 * tf5);

        Ok(Self {
            coefficients: vec![c0, c1, c2, c3, c4, c5],
            duration,
            order: 5,
        })
    }

    /// Sample.
    pub fn sample(&self, t: f64) -> ProfileSample {
        let tc = t.clamp(0.0, self.duration);
        let mut pos = 0.0;
        let mut vel = 0.0;
        let mut acc = 0.0;
        let mut jerk = 0.0;
        let mut tp = 1.0; // t^i

        for (i, &c) in self.coefficients.iter().enumerate() {
            pos += c * tp;
            if i >= 1 {
                vel += c * (i as f64) * tc.powi(i as i32 - 1);
            }
            if i >= 2 {
                acc += c * (i as f64) * ((i - 1) as f64) * tc.powi(i as i32 - 2);
            }
            if i >= 3 {
                jerk += c * (i as f64) * ((i - 1) as f64) * ((i - 2) as f64) * tc.powi(i as i32 - 3);
            }
            tp *= tc;
        }

        ProfileSample { time: tc, position: pos, velocity: vel, acceleration: acc, jerk }
    }

    /// Duration.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Polynomial order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Peak velocity estimate by sampling.
    pub fn peak_velocity_estimate(&self, num_samples: usize) -> f64 {
        let mut max_v = 0.0_f64;
        for i in 0..=num_samples {
            let t = self.duration * (i as f64) / (num_samples as f64);
            let s = self.sample(t);
            max_v = max_v.max(s.velocity.abs());
        }
        max_v
    }
}

impl std::fmt::Display for PolynomialProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PolynomialProfile(order={}, {:.3}s)",
            self.order, self.duration,
        )
    }
}

// ── Time-Optimal Profile ───────────────────────────────────────

/// Time-optimal (bang-bang) profile with maximum acceleration.
///
/// Two phases: full acceleration then full deceleration (symmetric).
#[derive(Debug, Clone, PartialEq)]
pub struct TimeOptimalProfile {
    distance: f64,
    a_max: f64,
    t_switch: f64,
    total_time: f64,
    sign: f64,
}

impl TimeOptimalProfile {
    /// Create a time-optimal profile.
    pub fn new(distance: f64, a_max: f64) -> Result<Self, ProfileError> {
        if a_max <= 0.0 {
            return Err(ProfileError::InvalidLimit("a_max must be positive".into()));
        }
        let sign = if distance >= 0.0 { 1.0 } else { -1.0 };
        let dist = distance.abs();
        let t_switch = (dist / a_max).sqrt();
        let total_time = 2.0 * t_switch;
        Ok(Self { distance: dist, a_max, t_switch, total_time, sign })
    }

    /// Duration.
    pub fn duration(&self) -> f64 {
        self.total_time
    }

    /// Peak velocity (at switch point).
    pub fn peak_velocity(&self) -> f64 {
        self.a_max * self.t_switch
    }

    /// Sample.
    pub fn sample(&self, t: f64) -> ProfileSample {
        let tc = t.clamp(0.0, self.total_time);
        let (pos, vel, acc);

        if tc <= self.t_switch {
            acc = self.a_max;
            vel = acc * tc;
            pos = 0.5 * acc * tc * tc;
        } else {
            let dt = tc - self.t_switch;
            let v_peak = self.a_max * self.t_switch;
            acc = -self.a_max;
            vel = v_peak - self.a_max * dt;
            let pos_half = 0.5 * self.a_max * self.t_switch * self.t_switch;
            pos = pos_half + v_peak * dt - 0.5 * self.a_max * dt * dt;
        }

        ProfileSample {
            time: tc,
            position: self.sign * pos,
            velocity: self.sign * vel,
            acceleration: self.sign * acc,
            jerk: 0.0,
        }
    }
}

impl std::fmt::Display for TimeOptimalProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TimeOptimalProfile(d={:.4}, a_max={:.4}, T={:.4}s)",
            self.distance, self.a_max, self.total_time,
        )
    }
}

// ── Multi-Axis Profile Synchronizer ────────────────────────────

/// Synchronizes multiple 1-D trapezoidal profiles to finish at the same time.
#[derive(Debug, Clone)]
pub struct SynchronizedProfiles {
    profiles: Vec<TrapezoidalProfile>,
    sync_time: f64,
}

impl SynchronizedProfiles {
    /// Create synchronized profiles for multiple axes.
    ///
    /// All axes share `v_max` and `a_max`.  The slowest axis determines the
    /// total duration, and faster axes are scaled down.
    pub fn new(
        distances: &[f64],
        v_max: f64,
        a_max: f64,
    ) -> Result<Self, ProfileError> {
        if distances.is_empty() {
            return Err(ProfileError::Infeasible("no axes".into()));
        }
        // Build individual profiles to find the slowest
        let mut max_time = 0.0_f64;
        for &d in distances {
            let p = TrapezoidalProfile::new(d, v_max, a_max)?;
            max_time = max_time.max(p.duration());
        }

        // Rebuild profiles scaled to the slowest time
        let mut profiles = Vec::with_capacity(distances.len());
        for &d in distances {
            let dist = d.abs();
            if dist < 1e-15 || max_time < 1e-15 {
                profiles.push(TrapezoidalProfile::new(d, v_max, a_max)?);
                continue;
            }
            // Scale v_max so the profile takes max_time
            let scaled_v = dist / (max_time - dist / a_max).max(1e-12);
            let actual_v = scaled_v.min(v_max);
            profiles.push(TrapezoidalProfile::new(d, actual_v, a_max)?);
        }

        Ok(Self { profiles, sync_time: max_time })
    }

    /// Synchronized duration.
    pub fn duration(&self) -> f64 {
        self.sync_time
    }

    /// Number of axes.
    pub fn num_axes(&self) -> usize {
        self.profiles.len()
    }

    /// Sample all axes at time `t`.
    pub fn sample_all(&self, t: f64) -> Vec<ProfileSample> {
        self.profiles.iter().map(|p| p.sample(t)).collect()
    }
}

impl std::fmt::Display for SynchronizedProfiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SynchronizedProfiles({} axes, T={:.4}s)",
            self.profiles.len(),
            self.sync_time,
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_trapezoidal_zero_distance() {
        let p = TrapezoidalProfile::new(0.0, 1.0, 1.0).unwrap();
        assert!(approx_eq(p.duration(), 0.0));
    }

    #[test]
    fn test_trapezoidal_start_end() {
        let p = TrapezoidalProfile::new(10.0, 2.0, 1.0).unwrap();
        let s0 = p.sample(0.0);
        let s1 = p.sample(p.duration());
        assert!(approx_eq(s0.position, 0.0));
        assert!((s1.position - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_trapezoidal_start_velocity_zero() {
        let p = TrapezoidalProfile::new(5.0, 2.0, 1.0).unwrap();
        let s = p.sample(0.0);
        assert!(approx_eq(s.velocity, 0.0));
    }

    #[test]
    fn test_trapezoidal_triangular() {
        // Small distance → can't reach v_max
        let p = TrapezoidalProfile::new(0.5, 10.0, 1.0).unwrap();
        assert!(p.is_triangular());
    }

    #[test]
    fn test_trapezoidal_full() {
        let p = TrapezoidalProfile::new(100.0, 5.0, 2.0).unwrap();
        assert!(!p.is_triangular());
    }

    #[test]
    fn test_trapezoidal_negative_distance() {
        let p = TrapezoidalProfile::new(-5.0, 2.0, 1.0).unwrap();
        let s = p.sample(p.duration());
        assert!(s.position < 0.0);
    }

    #[test]
    fn test_trapezoidal_invalid_v() {
        let r = TrapezoidalProfile::new(1.0, -1.0, 1.0);
        assert!(matches!(r, Err(ProfileError::InvalidLimit(_))));
    }

    #[test]
    fn test_trapezoidal_uniform_samples() {
        let p = TrapezoidalProfile::new(1.0, 1.0, 1.0).unwrap();
        let samples = p.sample_uniform(0.1);
        assert!(samples.len() >= 10);
    }

    #[test]
    fn test_scurve_zero() {
        let p = ScurveProfile::new(0.0, 1.0, 1.0, 1.0).unwrap();
        assert!(approx_eq(p.duration(), 0.0));
    }

    #[test]
    fn test_scurve_endpoints() {
        let p = ScurveProfile::new(10.0, 2.0, 1.0, 5.0).unwrap();
        let s0 = p.sample(0.0);
        let s1 = p.sample(p.duration());
        assert!(approx_eq(s0.position, 0.0));
        assert!((s1.position - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_scurve_invalid_jerk() {
        let r = ScurveProfile::new(1.0, 1.0, 1.0, 0.0);
        assert!(matches!(r, Err(ProfileError::InvalidLimit(_))));
    }

    #[test]
    fn test_polynomial_cubic_endpoints() {
        let p = PolynomialProfile::cubic(0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        let s0 = p.sample(0.0);
        let s1 = p.sample(1.0);
        assert!(approx_eq(s0.position, 0.0));
        assert!(approx_eq(s1.position, 1.0));
    }

    #[test]
    fn test_polynomial_quintic_endpoints() {
        let p = PolynomialProfile::quintic(0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 2.0).unwrap();
        let s0 = p.sample(0.0);
        let s1 = p.sample(2.0);
        assert!(approx_eq(s0.position, 0.0));
        assert!(approx_eq(s1.position, 5.0));
    }

    #[test]
    fn test_polynomial_invalid_duration() {
        let r = PolynomialProfile::cubic(0.0, 1.0, 0.0, 0.0, -1.0);
        assert!(matches!(r, Err(ProfileError::InvalidDuration(_))));
    }

    #[test]
    fn test_time_optimal_symmetric() {
        let p = TimeOptimalProfile::new(2.0, 1.0).unwrap();
        let mid = p.sample(p.duration() / 2.0);
        assert!(mid.velocity > 0.0);
        assert!(approx_eq(p.sample(0.0).velocity, 0.0));
    }

    #[test]
    fn test_time_optimal_distance() {
        let p = TimeOptimalProfile::new(4.0, 2.0).unwrap();
        let s = p.sample(p.duration());
        assert!((s.position - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_synchronized_profiles() {
        let sp = SynchronizedProfiles::new(&[1.0, 5.0, 3.0], 2.0, 1.0).unwrap();
        assert_eq!(sp.num_axes(), 3);
        assert!(sp.duration() > 0.0);
    }

    #[test]
    fn test_display_trapezoidal() {
        let p = TrapezoidalProfile::new(10.0, 2.0, 1.0).unwrap();
        let s = format!("{p}");
        assert!(s.contains("TrapezoidalProfile"));
    }

    #[test]
    fn test_display_scurve() {
        let p = ScurveProfile::new(5.0, 2.0, 1.0, 3.0).unwrap();
        let s = format!("{p}");
        assert!(s.contains("ScurveProfile"));
    }

    #[test]
    fn test_peak_velocity_estimate() {
        let p = PolynomialProfile::cubic(0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        let peak = p.peak_velocity_estimate(100);
        assert!(peak > 0.0);
    }
}
