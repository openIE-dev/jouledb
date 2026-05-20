//! IMU sensor fusion — accelerometer and gyroscope integration with
//! complementary filter, Madgwick orientation filter, gyroscope bias
//! estimation, and quaternion-based attitude representation.
//!
//! Pure-Rust inertial measurement unit processing for robotics and
//! embedded navigation, requiring no external dependencies.

use std::f64::consts::PI;
use std::fmt;

// ── Vector3 ─────────────────────────────────────────────────────

/// 3D vector for sensor readings.
#[derive(Debug, Clone, Copy, PartialEq)]
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

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            return Self::zero();
        }
        Self { x: self.x / n, y: self.y / n, z: self.z / n }
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn scale(&self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

// ── Quaternion ──────────────────────────────────────────────────

/// Unit quaternion for orientation representation (w, x, y, z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let a = axis.normalized();
        Self { w: half.cos(), x: a.x * s, y: a.y * s, z: a.z * s }
    }

    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let cr = (roll / 2.0).cos();
        let sr = (roll / 2.0).sin();
        let cp = (pitch / 2.0).cos();
        let sp = (pitch / 2.0).sin();
        let cy = (yaw / 2.0).cos();
        let sy = (yaw / 2.0).sin();
        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            return Self::identity();
        }
        Self { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }

    pub fn conjugate(&self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Rotate a vector by this quaternion: v' = q * v * q^-1.
    pub fn rotate_vector(&self, v: &Vec3) -> Vec3 {
        let qv = Quaternion::new(0.0, v.x, v.y, v.z);
        let result = self.multiply(&qv).multiply(&self.conjugate());
        Vec3::new(result.x, result.y, result.z)
    }

    /// Extract Euler angles (roll, pitch, yaw) from quaternion.
    pub fn to_euler(&self) -> (f64, f64, f64) {
        // Roll (x-axis rotation)
        let sinr = 2.0 * (self.w * self.x + self.y * self.z);
        let cosr = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let roll = sinr.atan2(cosr);

        // Pitch (y-axis rotation)
        let sinp = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            (PI / 2.0).copysign(sinp)
        } else {
            sinp.asin()
        };

        // Yaw (z-axis rotation)
        let siny = 2.0 * (self.w * self.z + self.x * self.y);
        let cosy = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let yaw = siny.atan2(cosy);

        (roll, pitch, yaw)
    }

    /// Convert to 3x3 rotation matrix (row-major).
    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let q = self.normalized();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ]
    }

    /// Spherical linear interpolation between two quaternions.
    pub fn slerp(&self, other: &Quaternion, t: f64) -> Quaternion {
        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let mut other = *other;
        if dot < 0.0 {
            other = Quaternion::new(-other.w, -other.x, -other.y, -other.z);
            dot = -dot;
        }
        if dot > 0.9995 {
            // Linear interpolation for very close quaternions
            let result = Quaternion::new(
                self.w + t * (other.w - self.w),
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
            );
            return result.normalized();
        }
        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
        Quaternion::new(
            wa * self.w + wb * other.w,
            wa * self.x + wb * other.x,
            wa * self.y + wb * other.y,
            wa * self.z + wb * other.z,
        )
    }
}

impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q({:.4}, {:.4}, {:.4}, {:.4})", self.w, self.x, self.y, self.z)
    }
}

// ── IMU Reading ─────────────────────────────────────────────────

/// Raw IMU sensor reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuReading {
    /// Accelerometer in m/s^2.
    pub accel: Vec3,
    /// Gyroscope in rad/s.
    pub gyro: Vec3,
    /// Timestamp in seconds.
    pub timestamp_s: f64,
}

impl ImuReading {
    pub fn new(accel: Vec3, gyro: Vec3, timestamp_s: f64) -> Self {
        Self { accel, gyro, timestamp_s }
    }
}

impl fmt::Display for ImuReading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMU(t={:.3}s, a={}, g={})", self.timestamp_s, self.accel, self.gyro)
    }
}

// ── Complementary Filter ────────────────────────────────────────

/// Complementary filter for fusing accelerometer and gyroscope.
#[derive(Debug, Clone)]
pub struct ComplementaryFilter {
    pub alpha: f64,
    pub roll: f64,
    pub pitch: f64,
    pub last_time: f64,
    pub initialized: bool,
}

impl ComplementaryFilter {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            roll: 0.0,
            pitch: 0.0,
            last_time: 0.0,
            initialized: false,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Update with a new IMU reading.
    pub fn update(&mut self, reading: &ImuReading) {
        // Accelerometer-based angles
        let accel_roll = reading.accel.y.atan2(reading.accel.z);
        let accel_pitch = (-reading.accel.x).atan2(
            (reading.accel.y * reading.accel.y + reading.accel.z * reading.accel.z).sqrt(),
        );

        if !self.initialized {
            self.roll = accel_roll;
            self.pitch = accel_pitch;
            self.last_time = reading.timestamp_s;
            self.initialized = true;
            return;
        }

        let dt = reading.timestamp_s - self.last_time;
        if dt <= 0.0 {
            return;
        }

        // Gyro integration
        let gyro_roll = self.roll + reading.gyro.x * dt;
        let gyro_pitch = self.pitch + reading.gyro.y * dt;

        // Complementary fusion
        self.roll = self.alpha * gyro_roll + (1.0 - self.alpha) * accel_roll;
        self.pitch = self.alpha * gyro_pitch + (1.0 - self.alpha) * accel_pitch;
        self.last_time = reading.timestamp_s;
    }

    /// Current orientation as (roll, pitch) in radians.
    pub fn orientation(&self) -> (f64, f64) {
        (self.roll, self.pitch)
    }

    pub fn reset(&mut self) {
        self.roll = 0.0;
        self.pitch = 0.0;
        self.initialized = false;
    }
}

impl fmt::Display for ComplementaryFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompFilter(alpha={:.2}, roll={:.2}deg, pitch={:.2}deg)",
            self.alpha,
            self.roll.to_degrees(),
            self.pitch.to_degrees(),
        )
    }
}

// ── Gyroscope Bias Estimator ────────────────────────────────────

/// Online gyroscope bias estimator using running average during still periods.
#[derive(Debug, Clone)]
pub struct GyroBiasEstimator {
    pub bias: Vec3,
    pub accel_threshold: f64,
    pub sample_count: usize,
    sum: Vec3,
    gravity_magnitude: f64,
}

impl GyroBiasEstimator {
    pub fn new() -> Self {
        Self {
            bias: Vec3::zero(),
            accel_threshold: 0.5,
            sample_count: 0,
            sum: Vec3::zero(),
            gravity_magnitude: 9.81,
        }
    }

    pub fn with_accel_threshold(mut self, threshold: f64) -> Self {
        self.accel_threshold = threshold;
        self
    }

    pub fn with_gravity(mut self, g: f64) -> Self {
        self.gravity_magnitude = g;
        self
    }

    /// Feed an IMU reading. Bias is updated only when the device is approximately still.
    pub fn update(&mut self, reading: &ImuReading) {
        let accel_mag = reading.accel.norm();
        if (accel_mag - self.gravity_magnitude).abs() < self.accel_threshold {
            self.sample_count += 1;
            self.sum = self.sum.add(&reading.gyro);
            let n = self.sample_count as f64;
            self.bias = Vec3::new(self.sum.x / n, self.sum.y / n, self.sum.z / n);
        }
    }

    /// Remove estimated bias from a gyroscope reading.
    pub fn correct(&self, gyro: &Vec3) -> Vec3 {
        gyro.sub(&self.bias)
    }

    pub fn reset(&mut self) {
        self.bias = Vec3::zero();
        self.sum = Vec3::zero();
        self.sample_count = 0;
    }
}

impl fmt::Display for GyroBiasEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GyroBias(bias={}, samples={})",
            self.bias, self.sample_count
        )
    }
}

// ── Madgwick Filter ─────────────────────────────────────────────

/// Madgwick's AHRS orientation filter using gradient descent.
#[derive(Debug, Clone)]
pub struct MadgwickFilter {
    pub beta: f64,
    pub quaternion: Quaternion,
    pub last_time: f64,
    pub initialized: bool,
    pub sample_freq: f64,
}

impl MadgwickFilter {
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            quaternion: Quaternion::identity(),
            last_time: 0.0,
            initialized: false,
            sample_freq: 100.0,
        }
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_sample_freq(mut self, freq: f64) -> Self {
        self.sample_freq = freq;
        self
    }

    pub fn with_initial_quaternion(mut self, q: Quaternion) -> Self {
        self.quaternion = q;
        self.initialized = true;
        self
    }

    /// Update with accelerometer and gyroscope data.
    pub fn update(&mut self, reading: &ImuReading) {
        let dt = if self.initialized {
            let d = reading.timestamp_s - self.last_time;
            if d <= 0.0 { 1.0 / self.sample_freq } else { d }
        } else {
            self.initialized = true;
            self.last_time = reading.timestamp_s;
            // Initialize from accelerometer
            let a = reading.accel.normalized();
            let pitch = (-a.x).asin();
            let roll = a.y.atan2(a.z);
            self.quaternion = Quaternion::from_euler(roll, pitch, 0.0);
            return;
        };
        self.last_time = reading.timestamp_s;

        let q = self.quaternion;
        let a = reading.accel.normalized();
        if a.norm() < 1e-10 {
            // No valid accelerometer data — integrate gyro only
            let gx = reading.gyro.x;
            let gy = reading.gyro.y;
            let gz = reading.gyro.z;
            let q_dot = Quaternion::new(
                0.5 * (-q.x * gx - q.y * gy - q.z * gz),
                0.5 * (q.w * gx + q.y * gz - q.z * gy),
                0.5 * (q.w * gy - q.x * gz + q.z * gx),
                0.5 * (q.w * gz + q.x * gy - q.y * gx),
            );
            self.quaternion = Quaternion::new(
                q.w + q_dot.w * dt,
                q.x + q_dot.x * dt,
                q.y + q_dot.y * dt,
                q.z + q_dot.z * dt,
            ).normalized();
            return;
        }

        // Gradient descent step
        // Objective function: f = q* (x) [0,0,0,1] (x) q - [ax,ay,az]
        let f1 = 2.0 * (q.x * q.z - q.w * q.y) - a.x;
        let f2 = 2.0 * (q.w * q.x + q.y * q.z) - a.y;
        let f3 = 2.0 * (0.5 - q.x * q.x - q.y * q.y) - a.z;

        // Jacobian
        let j11 = -2.0 * q.y;
        let j12 = 2.0 * q.z;
        let j13 = -2.0 * q.w;
        let j14 = 2.0 * q.x;
        let j21 = 2.0 * q.x;
        let j22 = 2.0 * q.w;
        let j23 = 2.0 * q.z;
        let j24 = 2.0 * q.y;
        let j31 = 0.0;
        let j32 = -4.0 * q.x;
        let j33 = -4.0 * q.y;
        let j34 = 0.0;

        // Gradient
        let mut gw = j11 * f1 + j21 * f2 + j31 * f3;
        let mut gx = j12 * f1 + j22 * f2 + j32 * f3;
        let mut gy = j13 * f1 + j23 * f2 + j33 * f3;
        let mut gz = j14 * f1 + j24 * f2 + j34 * f3;

        let grad_norm = (gw * gw + gx * gx + gy * gy + gz * gz).sqrt();
        if grad_norm > 1e-15 {
            gw /= grad_norm;
            gx /= grad_norm;
            gy /= grad_norm;
            gz /= grad_norm;
        }

        // Gyroscope quaternion derivative
        let qg = reading.gyro;
        let q_dot_w = 0.5 * (-q.x * qg.x - q.y * qg.y - q.z * qg.z);
        let q_dot_x = 0.5 * (q.w * qg.x + q.y * qg.z - q.z * qg.y);
        let q_dot_y = 0.5 * (q.w * qg.y - q.x * qg.z + q.z * qg.x);
        let q_dot_z = 0.5 * (q.w * qg.z + q.x * qg.y - q.y * qg.x);

        // Apply correction
        self.quaternion = Quaternion::new(
            q.w + (q_dot_w - self.beta * gw) * dt,
            q.x + (q_dot_x - self.beta * gx) * dt,
            q.y + (q_dot_y - self.beta * gy) * dt,
            q.z + (q_dot_z - self.beta * gz) * dt,
        ).normalized();
    }

    pub fn orientation(&self) -> Quaternion {
        self.quaternion
    }

    pub fn euler_angles(&self) -> (f64, f64, f64) {
        self.quaternion.to_euler()
    }

    pub fn reset(&mut self) {
        self.quaternion = Quaternion::identity();
        self.initialized = false;
    }
}

impl fmt::Display for MadgwickFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, p, y) = self.euler_angles();
        write!(
            f,
            "Madgwick(beta={:.3}, rpy=({:.1}, {:.1}, {:.1}) deg)",
            self.beta, r.to_degrees(), p.to_degrees(), y.to_degrees()
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_vec3_norm() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(0.0, 3.0, 4.0);
        let n = v.normalized();
        assert!((n.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!((z.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_vec3_display() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let s = format!("{v}");
        assert!(s.contains("1.0000"));
    }

    #[test]
    fn test_quaternion_identity_rotation() {
        let q = Quaternion::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = q.rotate_vector(&v);
        assert!((r.x - 1.0).abs() < 1e-10);
        assert!((r.y - 2.0).abs() < 1e-10);
        assert!((r.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_90deg_z() {
        let q = Quaternion::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let r = q.rotate_vector(&v);
        assert!((r.x).abs() < 1e-10);
        assert!((r.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_euler_roundtrip() {
        let q = Quaternion::from_euler(0.1, 0.2, 0.3);
        let (r, p, y) = q.to_euler();
        assert!((r - 0.1).abs() < 1e-10);
        assert!((p - 0.2).abs() < 1e-10);
        assert!((y - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_slerp() {
        let a = Quaternion::identity();
        let b = Quaternion::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let mid = a.slerp(&b, 0.5);
        let (_, _, yaw) = mid.to_euler();
        assert!((yaw - PI / 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_multiply_inverse() {
        let q = Quaternion::from_euler(0.3, 0.5, 0.7);
        let qi = q.conjugate();
        let result = q.multiply(&qi);
        assert!((result.w - 1.0).abs() < 1e-10);
        assert!(result.x.abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_display() {
        let q = Quaternion::identity();
        let s = format!("{q}");
        assert!(s.contains("Q("));
    }

    #[test]
    fn test_rotation_matrix_identity() {
        let q = Quaternion::identity();
        let m = q.to_rotation_matrix();
        assert!((m[0][0] - 1.0).abs() < 1e-12);
        assert!((m[1][1] - 1.0).abs() < 1e-12);
        assert!((m[2][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_complementary_filter_init() {
        let mut filter = ComplementaryFilter::new(0.98);
        let reading = ImuReading::new(
            Vec3::new(0.0, 0.0, 9.81),
            Vec3::zero(),
            0.0,
        );
        filter.update(&reading);
        let (roll, pitch) = filter.orientation();
        assert!(roll.abs() < 0.01);
        assert!(pitch.abs() < 0.01);
    }

    #[test]
    fn test_complementary_filter_tilted() {
        let mut filter = ComplementaryFilter::new(0.98);
        // Tilted 45 degrees in roll
        let reading = ImuReading::new(
            Vec3::new(0.0, 6.94, 6.94),
            Vec3::zero(),
            0.0,
        );
        filter.update(&reading);
        let (roll, _) = filter.orientation();
        assert!((roll - PI / 4.0).abs() < 0.05);
    }

    #[test]
    fn test_complementary_filter_display() {
        let filter = ComplementaryFilter::new(0.95);
        let s = format!("{filter}");
        assert!(s.contains("CompFilter"));
    }

    #[test]
    fn test_gyro_bias_still() {
        let mut estimator = GyroBiasEstimator::new().with_accel_threshold(1.0);
        for i in 0..100 {
            let reading = ImuReading::new(
                Vec3::new(0.0, 0.0, 9.81),
                Vec3::new(0.01, -0.02, 0.005),
                i as f64 * 0.01,
            );
            estimator.update(&reading);
        }
        assert!((estimator.bias.x - 0.01).abs() < 1e-10);
        assert!((estimator.bias.y - (-0.02)).abs() < 1e-10);
    }

    #[test]
    fn test_gyro_bias_correct() {
        let mut estimator = GyroBiasEstimator::new();
        estimator.bias = Vec3::new(0.01, -0.02, 0.0);
        let corrected = estimator.correct(&Vec3::new(0.51, 0.48, 1.0));
        assert!((corrected.x - 0.5).abs() < 1e-12);
        assert!((corrected.y - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_gyro_bias_display() {
        let est = GyroBiasEstimator::new();
        let s = format!("{est}");
        assert!(s.contains("GyroBias"));
    }

    #[test]
    fn test_madgwick_initial() {
        let filter = MadgwickFilter::new(0.1);
        let q = filter.orientation();
        assert!((q.w - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_madgwick_stationary() {
        let mut filter = MadgwickFilter::new(0.1).with_sample_freq(100.0);
        for i in 0..200 {
            let reading = ImuReading::new(
                Vec3::new(0.0, 0.0, 9.81),
                Vec3::zero(),
                i as f64 * 0.01,
            );
            filter.update(&reading);
        }
        let (roll, pitch, _) = filter.euler_angles();
        assert!(roll.abs() < 0.1);
        assert!(pitch.abs() < 0.1);
    }

    #[test]
    fn test_madgwick_display() {
        let filter = MadgwickFilter::new(0.1);
        let s = format!("{filter}");
        assert!(s.contains("Madgwick"));
    }

    #[test]
    fn test_imu_reading_display() {
        let reading = ImuReading::new(Vec3::new(0.0, 0.0, 9.81), Vec3::zero(), 1.5);
        let s = format!("{reading}");
        assert!(s.contains("IMU"));
    }
}
