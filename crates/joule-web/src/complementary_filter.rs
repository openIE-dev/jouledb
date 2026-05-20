//! Complementary Filter — sensor fusion combining high-frequency (gyroscope)
//! and low-frequency (accelerometer) data. First-order, quaternion-based 3D,
//! adaptive alpha, magnetometer integration, and tilt estimation.
//!
//! Replaces ad-hoc sensor fusion in JS/TS with a pure-Rust filter that handles
//! IMU orientation estimation for embedded and robotics workloads.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// Complementary filter errors.
#[derive(Debug, Clone, PartialEq)]
pub enum CompFilterError {
    /// Alpha parameter out of range.
    InvalidAlpha(f64),
    /// Invalid sample time.
    InvalidDt(f64),
    /// Zero-length vector (cannot normalize).
    ZeroVector,
    /// Invalid threshold.
    InvalidThreshold(String),
}

impl std::fmt::Display for CompFilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlpha(a) => write!(f, "alpha out of [0, 1]: {a}"),
            Self::InvalidDt(dt) => write!(f, "invalid dt: {dt}"),
            Self::ZeroVector => write!(f, "zero-length vector"),
            Self::InvalidThreshold(msg) => write!(f, "invalid threshold: {msg}"),
        }
    }
}

impl std::error::Error for CompFilterError {}

// ── 3D Vector ───────────────────────────────────────────────────

/// Simple 3D vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Result<Self, CompFilterError> {
        let m = self.magnitude();
        if m < 1e-12 {
            return Err(CompFilterError::ZeroVector);
        }
        Ok(Self { x: self.x / m, y: self.y / m, z: self.z / m })
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn add(&self, other: &Vec3) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

// ── Quaternion ──────────────────────────────────────────────────

/// Unit quaternion for 3D rotation: w + xi + yj + zk.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    /// Identity quaternion (no rotation).
    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Create from axis-angle.
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Result<Self, CompFilterError> {
        let n = axis.normalize()?;
        let half = angle / 2.0;
        let s = half.sin();
        Ok(Self {
            w: half.cos(),
            x: n.x * s,
            y: n.y * s,
            z: n.z * s,
        })
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-12 {
            return Self::identity();
        }
        Self { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }

    /// Hamilton product: self * other.
    pub fn mul(&self, other: &Quaternion) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Conjugate.
    pub fn conjugate(&self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// Rotate a vector by this quaternion: q * v * q'.
    pub fn rotate_vec(&self, v: &Vec3) -> Vec3 {
        let q_v = Quaternion { w: 0.0, x: v.x, y: v.y, z: v.z };
        let result = self.mul(&q_v).mul(&self.conjugate());
        Vec3 { x: result.x, y: result.y, z: result.z }
    }

    /// Spherical linear interpolation.
    pub fn slerp(&self, other: &Quaternion, t: f64) -> Self {
        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let mut other = *other;
        if dot < 0.0 {
            other = Quaternion { w: -other.w, x: -other.x, y: -other.y, z: -other.z };
            dot = -dot;
        }
        if dot > 0.9995 {
            // Linear interpolation for very close quaternions.
            let result = Quaternion {
                w: self.w + t * (other.w - self.w),
                x: self.x + t * (other.x - self.x),
                y: self.y + t * (other.y - self.y),
                z: self.z + t * (other.z - self.z),
            };
            return result.normalize();
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;
        Quaternion {
            w: a * self.w + b * other.w,
            x: a * self.x + b * other.x,
            y: a * self.y + b * other.y,
            z: a * self.z + b * other.z,
        }.normalize()
    }

    /// Convert to Euler angles (roll, pitch, yaw) in radians.
    pub fn to_euler(&self) -> (f64, f64, f64) {
        let sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z);
        let cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.z + self.x * self.y);
        let cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Create from Euler angles (roll, pitch, yaw).
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (sr, cr) = (roll / 2.0).sin_cos();
        let (sp, cp) = (pitch / 2.0).sin_cos();
        let (sy, cy) = (yaw / 2.0).sin_cos();
        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }
}

// ── Tilt Estimation ─────────────────────────────────────────────

/// Estimate roll and pitch from accelerometer data (assuming stationary or slow motion).
pub fn tilt_from_accel(accel: &Vec3) -> Result<(f64, f64), CompFilterError> {
    let _ = accel.normalize()?; // check non-zero
    let roll = accel.y.atan2(accel.z);
    let pitch = (-accel.x).atan2((accel.y * accel.y + accel.z * accel.z).sqrt());
    Ok((roll, pitch))
}

// ── First-Order Complementary Filter (1D) ───────────────────────

/// Simple first-order complementary filter for a single axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompFilter1D {
    /// Gyro trust weight (0..1).
    pub alpha: f64,
    /// Current angle estimate (radians).
    pub angle: f64,
    /// Sample time.
    pub dt: f64,
}

impl CompFilter1D {
    /// Create with specified alpha and dt.
    pub fn new(alpha: f64, dt: f64) -> Result<Self, CompFilterError> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(CompFilterError::InvalidAlpha(alpha));
        }
        if dt <= 0.0 {
            return Err(CompFilterError::InvalidDt(dt));
        }
        Ok(Self { alpha, angle: 0.0, dt })
    }

    /// Update: fuse gyro rate (rad/s) with accelerometer-derived angle (rad).
    pub fn update(&mut self, gyro_rate: f64, accel_angle: f64) -> f64 {
        self.angle = self.alpha * (self.angle + gyro_rate * self.dt)
            + (1.0 - self.alpha) * accel_angle;
        self.angle
    }

    /// Reset to specific angle.
    pub fn reset(&mut self, angle: f64) {
        self.angle = angle;
    }
}

// ── 3D Quaternion Complementary Filter ──────────────────────────

/// Configuration for the quaternion complementary filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuatCompConfig {
    /// Base alpha for gyro trust.
    pub alpha: f64,
    /// Sample time in seconds.
    pub dt: f64,
    /// Enable adaptive alpha.
    pub adaptive: bool,
    /// Threshold angular rate (rad/s) for full gyro trust.
    pub gyro_threshold: f64,
    /// Enable magnetometer fusion for yaw.
    pub use_magnetometer: bool,
    /// Magnetometer trust weight.
    pub mag_alpha: f64,
}

impl QuatCompConfig {
    /// Sensible defaults for 100 Hz IMU.
    pub fn default_100hz() -> Self {
        Self {
            alpha: 0.98,
            dt: 0.01,
            adaptive: false,
            gyro_threshold: 1.0,
            use_magnetometer: false,
            mag_alpha: 0.95,
        }
    }
}

/// 3D complementary filter using quaternions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuatCompFilter {
    pub config: QuatCompConfig,
    pub orientation: Quaternion,
}

impl QuatCompFilter {
    pub fn new(config: QuatCompConfig) -> Result<Self, CompFilterError> {
        if !(0.0..=1.0).contains(&config.alpha) {
            return Err(CompFilterError::InvalidAlpha(config.alpha));
        }
        if config.dt <= 0.0 {
            return Err(CompFilterError::InvalidDt(config.dt));
        }
        Ok(Self {
            config,
            orientation: Quaternion::identity(),
        })
    }

    /// Compute adaptive alpha based on angular rate magnitude.
    fn effective_alpha(&self, gyro: &Vec3) -> f64 {
        if !self.config.adaptive {
            return self.config.alpha;
        }
        let rate = gyro.magnitude();
        // Increase alpha (more gyro trust) at high angular rates.
        let ratio = (rate / self.config.gyro_threshold).min(1.0);
        self.config.alpha + (1.0 - self.config.alpha) * ratio
    }

    /// Integrate gyroscope to get delta quaternion.
    fn gyro_integration(&self, gyro: &Vec3) -> Quaternion {
        let angle = gyro.magnitude() * self.config.dt;
        if angle < 1e-12 {
            return Quaternion::identity();
        }
        let axis = gyro.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        Quaternion::from_axis_angle(&axis, angle).unwrap_or(Quaternion::identity())
    }

    /// Update with gyroscope and accelerometer data.
    pub fn update(&mut self, gyro: &Vec3, accel: &Vec3) -> Quaternion {
        let alpha = self.effective_alpha(gyro);

        // Gyro prediction.
        let dq = self.gyro_integration(gyro);
        let q_gyro = self.orientation.mul(&dq).normalize();

        // Accelerometer correction: compute tilt quaternion from accel.
        let q_accel = self.accel_quaternion(accel);

        // Fuse via SLERP.
        self.orientation = q_gyro.slerp(&q_accel, 1.0 - alpha);
        self.orientation
    }

    /// Update with gyroscope, accelerometer, and magnetometer.
    pub fn update_with_mag(
        &mut self,
        gyro: &Vec3,
        accel: &Vec3,
        mag: &Vec3,
    ) -> Quaternion {
        // First do gyro+accel fusion.
        self.update(gyro, accel);

        if !self.config.use_magnetometer {
            return self.orientation;
        }

        // Compute yaw from magnetometer.
        if let Ok(yaw) = self.mag_yaw(accel, mag) {
            let (roll, pitch, current_yaw) = self.orientation.to_euler();
            let fused_yaw = self.config.mag_alpha * current_yaw
                + (1.0 - self.config.mag_alpha) * yaw;
            self.orientation = Quaternion::from_euler(roll, pitch, fused_yaw);
        }

        self.orientation
    }

    /// Estimate orientation quaternion from accelerometer alone (roll+pitch, no yaw).
    fn accel_quaternion(&self, accel: &Vec3) -> Quaternion {
        if let Ok((roll, pitch)) = tilt_from_accel(accel) {
            let (_, _, yaw) = self.orientation.to_euler();
            Quaternion::from_euler(roll, pitch, yaw)
        } else {
            self.orientation
        }
    }

    /// Compute yaw from tilt-compensated magnetometer.
    fn mag_yaw(&self, accel: &Vec3, mag: &Vec3) -> Result<f64, CompFilterError> {
        let (roll, pitch) = tilt_from_accel(accel)?;
        let cr = roll.cos();
        let sr = roll.sin();
        let cp = pitch.cos();
        let sp = pitch.sin();

        // Tilt-compensate magnetometer.
        let mx = mag.x * cp + mag.y * sp * sr + mag.z * sp * cr;
        let my = mag.y * cr - mag.z * sr;
        Ok((-my).atan2(mx))
    }

    /// Get current Euler angles (roll, pitch, yaw).
    pub fn euler(&self) -> (f64, f64, f64) {
        self.orientation.to_euler()
    }

    /// Reset to identity.
    pub fn reset(&mut self) {
        self.orientation = Quaternion::identity();
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
    fn test_vec3_magnitude() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(v.magnitude(), 5.0, 1e-10));
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(0.0, 3.0, 4.0);
        let n = v.normalize().unwrap();
        assert!(approx(n.magnitude(), 1.0, 1e-10));
    }

    #[test]
    fn test_vec3_zero_normalize_fails() {
        let v = Vec3::zero();
        assert!(v.normalize().is_err());
    }

    #[test]
    fn test_vec3_cross_product() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!(approx(z.x, 0.0, 1e-10));
        assert!(approx(z.y, 0.0, 1e-10));
        assert!(approx(z.z, 1.0, 1e-10));
    }

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = q.rotate_vec(&v);
        assert!(approx(rotated.x, 1.0, 1e-10));
        assert!(approx(rotated.y, 2.0, 1e-10));
        assert!(approx(rotated.z, 3.0, 1e-10));
    }

    #[test]
    fn test_quaternion_90_deg_rotation() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let q = Quaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2).unwrap();
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vec(&v);
        // 90 degrees around z: (1,0,0) -> (0,1,0)
        assert!(approx(rotated.x, 0.0, 1e-8));
        assert!(approx(rotated.y, 1.0, 1e-8));
        assert!(approx(rotated.z, 0.0, 1e-8));
    }

    #[test]
    fn test_quaternion_euler_roundtrip() {
        let roll = 0.3;
        let pitch = 0.2;
        let yaw = 0.5;
        let q = Quaternion::from_euler(roll, pitch, yaw);
        let (r, p, y) = q.to_euler();
        assert!(approx(r, roll, 1e-8));
        assert!(approx(p, pitch, 1e-8));
        assert!(approx(y, yaw, 1e-8));
    }

    #[test]
    fn test_quaternion_slerp_endpoints() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_euler(0.5, 0.0, 0.0);
        let s0 = q1.slerp(&q2, 0.0);
        assert!(approx(s0.w, q1.w, 1e-4));
        let s1 = q1.slerp(&q2, 1.0);
        assert!(approx(s1.w, q2.w, 1e-4));
    }

    #[test]
    fn test_quaternion_slerp_midpoint() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_euler(1.0, 0.0, 0.0);
        let mid = q1.slerp(&q2, 0.5);
        let (r, _, _) = mid.to_euler();
        assert!(approx(r, 0.5, 0.05));
    }

    #[test]
    fn test_tilt_from_accel_level() {
        // Sensor pointing up: accel = (0, 0, 9.81)
        let accel = Vec3::new(0.0, 0.0, 9.81);
        let (roll, pitch) = tilt_from_accel(&accel).unwrap();
        assert!(approx(roll, 0.0, 1e-4));
        assert!(approx(pitch, 0.0, 1e-4));
    }

    #[test]
    fn test_tilt_from_accel_tilted() {
        // Tilted ~45 degrees in pitch.
        let accel = Vec3::new(-6.94, 0.0, 6.94);
        let (roll, pitch) = tilt_from_accel(&accel).unwrap();
        assert!(approx(roll, 0.0, 1e-4));
        assert!(approx(pitch, std::f64::consts::FRAC_PI_4, 0.01));
    }

    #[test]
    fn test_comp_filter_1d_pure_gyro() {
        let mut cf = CompFilter1D::new(1.0, 0.01).unwrap();
        // alpha=1 => only gyro.
        cf.update(10.0, 0.0); // gyro: 10 rad/s => angle += 0.1
        assert!(approx(cf.angle, 0.1, 1e-4));
    }

    #[test]
    fn test_comp_filter_1d_pure_accel() {
        let mut cf = CompFilter1D::new(0.0, 0.01).unwrap();
        // alpha=0 => only accel.
        cf.update(10.0, 0.5);
        assert!(approx(cf.angle, 0.5, 1e-4));
    }

    #[test]
    fn test_comp_filter_1d_fusion() {
        let mut cf = CompFilter1D::new(0.98, 0.01).unwrap();
        cf.angle = 0.5;
        let result = cf.update(0.0, 0.5); // gyro=0 + accel=0.5
        // 0.98*(0.5 + 0) + 0.02*0.5 = 0.5
        assert!(approx(result, 0.5, 1e-4));
    }

    #[test]
    fn test_comp_filter_1d_drift_correction() {
        let mut cf = CompFilter1D::new(0.98, 0.01).unwrap();
        // Gyro drifts, accel holds truth at 0.
        for _ in 0..1000 {
            cf.update(0.1, 0.0); // small gyro bias
        }
        // Should not drift far due to accel correction.
        assert!(cf.angle.abs() < 1.0);
    }

    #[test]
    fn test_comp_filter_1d_invalid_alpha() {
        assert!(CompFilter1D::new(1.5, 0.01).is_err());
        assert!(CompFilter1D::new(-0.1, 0.01).is_err());
    }

    #[test]
    fn test_comp_filter_1d_invalid_dt() {
        assert!(CompFilter1D::new(0.98, 0.0).is_err());
        assert!(CompFilter1D::new(0.98, -1.0).is_err());
    }

    #[test]
    fn test_quat_comp_filter_creation() {
        let cfg = QuatCompConfig::default_100hz();
        let filter = QuatCompFilter::new(cfg).unwrap();
        assert!(approx(filter.orientation.w, 1.0, 1e-10));
    }

    #[test]
    fn test_quat_comp_filter_stationary() {
        let cfg = QuatCompConfig::default_100hz();
        let mut filter = QuatCompFilter::new(cfg).unwrap();
        let gyro = Vec3::zero();
        let accel = Vec3::new(0.0, 0.0, 9.81);
        for _ in 0..100 {
            filter.update(&gyro, &accel);
        }
        let (r, p, _) = filter.euler();
        assert!(approx(r, 0.0, 0.1));
        assert!(approx(p, 0.0, 0.1));
    }

    #[test]
    fn test_quat_comp_filter_adaptive() {
        let mut cfg = QuatCompConfig::default_100hz();
        cfg.adaptive = true;
        cfg.gyro_threshold = 2.0;
        let filter = QuatCompFilter::new(cfg).unwrap();

        // Low rate => alpha near base.
        let low_gyro = Vec3::new(0.1, 0.0, 0.0);
        let alpha_low = filter.effective_alpha(&low_gyro);
        assert!(alpha_low < 0.99);

        // High rate => alpha near 1.
        let high_gyro = Vec3::new(5.0, 0.0, 0.0);
        let alpha_high = filter.effective_alpha(&high_gyro);
        assert!(approx(alpha_high, 1.0, 0.01));
    }

    #[test]
    fn test_quat_comp_filter_reset() {
        let cfg = QuatCompConfig::default_100hz();
        let mut filter = QuatCompFilter::new(cfg).unwrap();
        filter.orientation = Quaternion::from_euler(0.5, 0.3, 0.1);
        filter.reset();
        assert!(approx(filter.orientation.w, 1.0, 1e-10));
    }

    #[test]
    fn test_quaternion_mul_inverse_is_identity() {
        let q = Quaternion::from_euler(0.3, 0.5, 0.7);
        let qi = q.conjugate();
        let product = q.mul(&qi);
        assert!(approx(product.w, 1.0, 1e-8));
        assert!(approx(product.x, 0.0, 1e-8));
    }

    #[test]
    fn test_comp_filter_1d_reset() {
        let mut cf = CompFilter1D::new(0.98, 0.01).unwrap();
        cf.update(1.0, 0.5);
        cf.reset(0.0);
        assert!(approx(cf.angle, 0.0, 1e-10));
    }

    #[test]
    fn test_vec3_dot_product() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx(a.dot(&b), 32.0, 1e-10));
    }
}
