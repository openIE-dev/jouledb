//! EKF-SLAM — Extended Kalman Filter for Simultaneous Localization and Mapping.
//!
//! State augmentation for new landmarks, Mahalanobis-gated data association,
//! and full covariance update in a single dense state vector
//! `[x, y, θ, lx₁, ly₁, …, lxₙ, lyₙ]`.

use std::fmt;

// ── Linear-algebra helpers ────────────────────────────────────────

/// Dense matrix stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn mul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "dimension mismatch");
        let mut out = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 { continue; }
                for j in 0..other.cols {
                    let cur = out.get(i, j);
                    out.set(i, j, cur + a * other.get(k, j));
                }
            }
        }
        out
    }

    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    pub fn add(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Mat { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Mat { rows: self.rows, cols: self.cols, data }
    }

    pub fn scale(&self, s: f64) -> Mat {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        Mat { rows: self.rows, cols: self.cols, data }
    }

    /// 2×2 inverse (for innovation covariance).
    pub fn inv2x2(&self) -> Option<Mat> {
        assert_eq!(self.rows, 2);
        assert_eq!(self.cols, 2);
        let a = self.get(0, 0);
        let b = self.get(0, 1);
        let c = self.get(1, 0);
        let d = self.get(1, 1);
        let det = a * d - b * c;
        if det.abs() < 1e-15 { return None; }
        let inv_det = 1.0 / det;
        let mut out = Mat::zeros(2, 2);
        out.set(0, 0, d * inv_det);
        out.set(0, 1, -b * inv_det);
        out.set(1, 0, -c * inv_det);
        out.set(1, 1, a * inv_det);
        Some(out)
    }
}

// ── 2-D pose ──────────────────────────────────────────────────────

/// Robot pose `(x, y, θ)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2D {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose2D {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta: normalize_angle(theta) }
    }
}

impl fmt::Display for Pose2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pose2D({:.3}, {:.3}, {:.4} rad)", self.x, self.y, self.theta)
    }
}

fn normalize_angle(a: f64) -> f64 {
    let mut a = a % (2.0 * std::f64::consts::PI);
    if a > std::f64::consts::PI { a -= 2.0 * std::f64::consts::PI; }
    if a < -std::f64::consts::PI { a += 2.0 * std::f64::consts::PI; }
    a
}

// ── Landmark ──────────────────────────────────────────────────────

/// A 2-D landmark with an ID.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmark {
    pub id: usize,
    pub x: f64,
    pub y: f64,
}

impl fmt::Display for Landmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Landmark(id={}, {:.3}, {:.3})", self.id, self.x, self.y)
    }
}

// ── Observation ───────────────────────────────────────────────────

/// Range-bearing observation `(range, bearing)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub range: f64,
    pub bearing: f64,
}

// ── Configuration ─────────────────────────────────────────────────

/// EKF-SLAM configuration.
#[derive(Debug, Clone)]
pub struct EkfSlamConfig {
    pub motion_noise: [f64; 3],
    pub observation_noise: [f64; 2],
    pub mahalanobis_gate: f64,
    pub initial_landmark_cov: f64,
}

impl Default for EkfSlamConfig {
    fn default() -> Self {
        Self {
            motion_noise: [0.1, 0.1, 0.01],
            observation_noise: [0.1, 0.05],
            mahalanobis_gate: 5.991, // χ² 95% for 2 DOF
            initial_landmark_cov: 1000.0,
        }
    }
}

impl EkfSlamConfig {
    pub fn new() -> Self { Self::default() }

    pub fn with_motion_noise(mut self, noise: [f64; 3]) -> Self {
        self.motion_noise = noise;
        self
    }

    pub fn with_observation_noise(mut self, noise: [f64; 2]) -> Self {
        self.observation_noise = noise;
        self
    }

    pub fn with_mahalanobis_gate(mut self, gate: f64) -> Self {
        self.mahalanobis_gate = gate;
        self
    }

    pub fn with_initial_landmark_cov(mut self, cov: f64) -> Self {
        self.initial_landmark_cov = cov;
        self
    }
}

// ── EKF-SLAM filter ──────────────────────────────────────────────

/// EKF-SLAM state: dense state vector + covariance.
#[derive(Debug, Clone)]
pub struct EkfSlam {
    /// State vector: [x, y, θ, lx₁, ly₁, …].
    pub state: Vec<f64>,
    /// Covariance matrix (row-major, n×n).
    pub cov: Mat,
    /// Number of landmarks currently in the map.
    pub num_landmarks: usize,
    /// Mapping from landmark ID → index in state vector.
    pub landmark_ids: Vec<usize>,
    pub config: EkfSlamConfig,
}

impl EkfSlam {
    pub fn new(pose: Pose2D, config: EkfSlamConfig) -> Self {
        let state = vec![pose.x, pose.y, pose.theta];
        let cov = Mat::identity(3).scale(0.001);
        Self { state, cov, num_landmarks: 0, landmark_ids: Vec::new(), config }
    }

    /// Current robot pose.
    pub fn pose(&self) -> Pose2D {
        Pose2D::new(self.state[0], self.state[1], self.state[2])
    }

    /// State dimension.
    pub fn dim(&self) -> usize { 3 + 2 * self.num_landmarks }

    /// Get landmark position by slot index (0-based).
    pub fn landmark_position(&self, idx: usize) -> (f64, f64) {
        let base = 3 + idx * 2;
        (self.state[base], self.state[base + 1])
    }

    /// Find internal slot for a landmark ID, or None.
    pub fn find_landmark(&self, id: usize) -> Option<usize> {
        self.landmark_ids.iter().position(|lid| *lid == id)
    }

    // ── Prediction ────────────────────────────────────────────────

    /// Predict step with velocity model: `(v, ω, dt)`.
    pub fn predict(&mut self, v: f64, omega: f64, dt: f64) {
        let theta = self.state[2];
        let (st, ct) = theta.sin_cos();

        // State update
        if omega.abs() < 1e-10 {
            self.state[0] += v * ct * dt;
            self.state[1] += v * st * dt;
        } else {
            let r = v / omega;
            self.state[0] += r * ((theta + omega * dt).sin() - st);
            self.state[1] += r * (ct - (theta + omega * dt).cos());
            self.state[2] = normalize_angle(self.state[2] + omega * dt);
        }

        // Jacobian of motion model w.r.t. robot state (3×3 in top-left)
        let n = self.dim();
        let mut fx = Mat::identity(n);
        if omega.abs() < 1e-10 {
            fx.set(0, 2, -v * st * dt);
            fx.set(1, 2, v * ct * dt);
        } else {
            let r = v / omega;
            let new_theta = theta + omega * dt;
            fx.set(0, 2, r * (new_theta.cos() - ct));
            fx.set(1, 2, r * (new_theta.sin() - st));
        }

        // Process noise
        let mut q = Mat::zeros(n, n);
        q.set(0, 0, self.config.motion_noise[0]);
        q.set(1, 1, self.config.motion_noise[1]);
        q.set(2, 2, self.config.motion_noise[2]);

        // P = Fx P Fx^T + Q
        let fx_t = fx.transpose();
        self.cov = fx.mul(&self.cov).mul(&fx_t).add(&q);
    }

    // ── Data association ──────────────────────────────────────────

    /// Mahalanobis distance for an observation to a known landmark slot.
    pub fn mahalanobis_distance(&self, obs: &Observation, slot: usize) -> f64 {
        let (innovation, s_mat) = self.innovation(obs, slot);
        if let Some(s_inv) = s_mat.inv2x2() {
            let tmp = s_inv.mul(&innovation);
            innovation.get(0, 0) * tmp.get(0, 0) + innovation.get(1, 0) * tmp.get(1, 0)
        } else {
            f64::MAX
        }
    }

    /// Associate observation with nearest landmark below the gate, or None.
    pub fn associate(&self, obs: &Observation) -> Option<usize> {
        let mut best_slot = None;
        let mut best_dist = self.config.mahalanobis_gate;
        for slot in 0..self.num_landmarks {
            let d = self.mahalanobis_distance(obs, slot);
            if d < best_dist {
                best_dist = d;
                best_slot = Some(slot);
            }
        }
        best_slot
    }

    fn innovation(&self, obs: &Observation, slot: usize) -> (Mat, Mat) {
        let rx = self.state[0];
        let ry = self.state[1];
        let rtheta = self.state[2];
        let (lx, ly) = self.landmark_position(slot);

        let dx = lx - rx;
        let dy = ly - ry;
        let q = dx * dx + dy * dy;
        let sq = q.sqrt();

        let z_hat_range = sq;
        let z_hat_bearing = normalize_angle(dy.atan2(dx) - rtheta);

        let mut innov = Mat::zeros(2, 1);
        innov.set(0, 0, obs.range - z_hat_range);
        innov.set(1, 0, normalize_angle(obs.bearing - z_hat_bearing));

        // Observation Jacobian H (2 × n)
        let n = self.dim();
        let mut h = Mat::zeros(2, n);
        // w.r.t. robot
        h.set(0, 0, -dx / sq);
        h.set(0, 1, -dy / sq);
        h.set(0, 2, 0.0);
        h.set(1, 0, dy / q);
        h.set(1, 1, -dx / q);
        h.set(1, 2, -1.0);
        // w.r.t. landmark
        let li = 3 + slot * 2;
        h.set(0, li, dx / sq);
        h.set(0, li + 1, dy / sq);
        h.set(1, li, -dy / q);
        h.set(1, li + 1, dx / q);

        let mut r = Mat::zeros(2, 2);
        r.set(0, 0, self.config.observation_noise[0]);
        r.set(1, 1, self.config.observation_noise[1]);

        let s = h.mul(&self.cov).mul(&h.transpose()).add(&r);
        (innov, s)
    }

    // ── Update ────────────────────────────────────────────────────

    /// Update with an observation associated to a known landmark slot.
    pub fn update(&mut self, obs: &Observation, slot: usize) {
        let (innov, s) = self.innovation(obs, slot);
        let s_inv = match s.inv2x2() {
            Some(inv) => inv,
            None => return,
        };

        let n = self.dim();
        // Recompute H
        let rx = self.state[0];
        let ry = self.state[1];
        let rtheta = self.state[2];
        let (lx, ly) = self.landmark_position(slot);
        let dx = lx - rx;
        let dy = ly - ry;
        let q = dx * dx + dy * dy;
        let sq = q.sqrt();

        let mut h = Mat::zeros(2, n);
        h.set(0, 0, -dx / sq);
        h.set(0, 1, -dy / sq);
        h.set(1, 0, dy / q);
        h.set(1, 1, -dx / q);
        h.set(1, 2, -1.0);
        let li = 3 + slot * 2;
        h.set(0, li, dx / sq);
        h.set(0, li + 1, dy / sq);
        h.set(1, li, -dy / q);
        h.set(1, li + 1, dx / q);

        // Kalman gain: K = P H^T S^{-1}
        let ht = h.transpose();
        let k = self.cov.mul(&ht).mul(&s_inv);

        // State update
        let delta = k.mul(&innov);
        for i in 0..n {
            self.state[i] += delta.get(i, 0);
        }
        self.state[2] = normalize_angle(self.state[2]);

        // Covariance update: P = (I - K H) P
        let kh = k.mul(&h);
        let eye = Mat::identity(n);
        self.cov = eye.sub(&kh).mul(&self.cov);
    }

    // ── State augmentation ────────────────────────────────────────

    /// Add a new landmark from an unassociated observation.
    pub fn add_landmark(&mut self, obs: &Observation, id: usize) -> usize {
        let rx = self.state[0];
        let ry = self.state[1];
        let rtheta = self.state[2];

        let lx = rx + obs.range * (rtheta + obs.bearing).cos();
        let ly = ry + obs.range * (rtheta + obs.bearing).sin();

        self.state.push(lx);
        self.state.push(ly);
        self.landmark_ids.push(id);

        let old_n = self.dim();
        let new_n = old_n + 2;

        // Expand covariance
        let mut new_cov = Mat::zeros(new_n, new_n);
        for r in 0..old_n {
            for c in 0..old_n {
                new_cov.set(r, c, self.cov.get(r, c));
            }
        }
        new_cov.set(old_n, old_n, self.config.initial_landmark_cov);
        new_cov.set(old_n + 1, old_n + 1, self.config.initial_landmark_cov);

        self.cov = new_cov;
        self.num_landmarks += 1;
        self.num_landmarks - 1
    }

    /// Process an observation: associate or augment, then update.
    pub fn process_observation(&mut self, obs: &Observation, id: usize) {
        if let Some(slot) = self.find_landmark(id) {
            self.update(obs, slot);
        } else if let Some(slot) = self.associate(obs) {
            self.update(obs, slot);
        } else {
            let slot = self.add_landmark(obs, id);
            self.update(obs, slot);
        }
    }

    /// All landmark positions as (id, x, y).
    pub fn landmarks(&self) -> Vec<(usize, f64, f64)> {
        (0..self.num_landmarks)
            .map(|i| {
                let (lx, ly) = self.landmark_position(i);
                (self.landmark_ids[i], lx, ly)
            })
            .collect()
    }
}

impl fmt::Display for EkfSlam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EkfSlam(pose={}, landmarks={}, dim={})",
            self.pose(),
            self.num_landmarks,
            self.dim()
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_slam() -> EkfSlam {
        EkfSlam::new(Pose2D::new(0.0, 0.0, 0.0), EkfSlamConfig::default())
    }

    #[test]
    fn test_mat_identity() {
        let m = Mat::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(2, 2), 1.0);
    }

    #[test]
    fn test_mat_multiply() {
        let a = Mat { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let b = Mat { rows: 2, cols: 2, data: vec![5.0, 6.0, 7.0, 8.0] };
        let c = a.mul(&b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_mat_transpose() {
        let m = Mat { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(1, 0), 2.0);
    }

    #[test]
    fn test_mat_inv2x2() {
        let m = Mat { rows: 2, cols: 2, data: vec![4.0, 7.0, 2.0, 6.0] };
        let inv = m.inv2x2().unwrap();
        let product = m.mul(&inv);
        assert!((product.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((product.get(1, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_angle() {
        assert!((normalize_angle(4.0) - (4.0 - 2.0 * std::f64::consts::PI)).abs() < 1e-10);
        assert!((normalize_angle(-4.0) - (-4.0 + 2.0 * std::f64::consts::PI)).abs() < 1e-10);
        assert!((normalize_angle(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pose_display() {
        let p = Pose2D::new(1.0, 2.0, 0.5);
        let s = format!("{}", p);
        assert!(s.contains("Pose2D"));
    }

    #[test]
    fn test_initial_state() {
        let slam = default_slam();
        assert_eq!(slam.dim(), 3);
        assert_eq!(slam.num_landmarks, 0);
        let p = slam.pose();
        assert!((p.x).abs() < 1e-10);
    }

    #[test]
    fn test_predict_straight() {
        let mut slam = default_slam();
        slam.predict(1.0, 0.0, 1.0);
        let p = slam.pose();
        assert!((p.x - 1.0).abs() < 1e-6);
        assert!((p.y).abs() < 1e-6);
    }

    #[test]
    fn test_predict_turn() {
        let mut slam = default_slam();
        slam.predict(1.0, std::f64::consts::FRAC_PI_2, 1.0);
        let p = slam.pose();
        assert!((p.theta - std::f64::consts::FRAC_PI_2).abs() < 0.2);
    }

    #[test]
    fn test_add_landmark() {
        let mut slam = default_slam();
        let obs = Observation { range: 5.0, bearing: 0.0 };
        let slot = slam.add_landmark(&obs, 42);
        assert_eq!(slot, 0);
        assert_eq!(slam.num_landmarks, 1);
        assert_eq!(slam.dim(), 5);
        let (lx, ly) = slam.landmark_position(0);
        assert!((lx - 5.0).abs() < 1e-6);
        assert!(ly.abs() < 1e-6);
    }

    #[test]
    fn test_find_landmark() {
        let mut slam = default_slam();
        let obs = Observation { range: 5.0, bearing: 0.0 };
        slam.add_landmark(&obs, 10);
        slam.add_landmark(&obs, 20);
        assert_eq!(slam.find_landmark(20), Some(1));
        assert_eq!(slam.find_landmark(99), None);
    }

    #[test]
    fn test_mahalanobis_distance() {
        let mut slam = default_slam();
        let obs = Observation { range: 5.0, bearing: 0.0 };
        slam.add_landmark(&obs, 1);
        // Same observation should yield small distance
        let d = slam.mahalanobis_distance(&obs, 0);
        assert!(d < 1.0);
    }

    #[test]
    fn test_data_association_new() {
        let slam = default_slam();
        let obs = Observation { range: 5.0, bearing: 0.0 };
        assert!(slam.associate(&obs).is_none());
    }

    #[test]
    fn test_update_reduces_covariance() {
        let mut slam = default_slam();
        let obs = Observation { range: 5.0, bearing: 0.0 };
        slam.add_landmark(&obs, 1);
        let cov_before = slam.cov.get(3, 3);
        slam.update(&obs, 0);
        let cov_after = slam.cov.get(3, 3);
        assert!(cov_after < cov_before);
    }

    #[test]
    fn test_process_observation_new_landmark() {
        let mut slam = default_slam();
        let obs = Observation { range: 3.0, bearing: 0.5 };
        slam.process_observation(&obs, 7);
        assert_eq!(slam.num_landmarks, 1);
    }

    #[test]
    fn test_process_observation_known_landmark() {
        let mut slam = default_slam();
        let obs = Observation { range: 3.0, bearing: 0.5 };
        slam.process_observation(&obs, 7);
        slam.process_observation(&obs, 7);
        // Still one landmark
        assert_eq!(slam.num_landmarks, 1);
    }

    #[test]
    fn test_landmarks_list() {
        let mut slam = default_slam();
        slam.add_landmark(&Observation { range: 3.0, bearing: 0.0 }, 10);
        slam.add_landmark(&Observation { range: 5.0, bearing: 1.0 }, 20);
        let lms = slam.landmarks();
        assert_eq!(lms.len(), 2);
        assert_eq!(lms[0].0, 10);
        assert_eq!(lms[1].0, 20);
    }

    #[test]
    fn test_config_builder() {
        let cfg = EkfSlamConfig::new()
            .with_motion_noise([0.2, 0.2, 0.02])
            .with_mahalanobis_gate(9.21);
        assert_eq!(cfg.motion_noise[0], 0.2);
        assert!((cfg.mahalanobis_gate - 9.21).abs() < 1e-10);
    }

    #[test]
    fn test_ekf_slam_display() {
        let slam = default_slam();
        let s = format!("{}", slam);
        assert!(s.contains("EkfSlam"));
        assert!(s.contains("landmarks=0"));
    }

    #[test]
    fn test_multiple_predict_accumulates() {
        let mut slam = default_slam();
        for _ in 0..10 {
            slam.predict(0.5, 0.0, 0.1);
        }
        let p = slam.pose();
        assert!((p.x - 0.5).abs() < 0.1);
    }
}
