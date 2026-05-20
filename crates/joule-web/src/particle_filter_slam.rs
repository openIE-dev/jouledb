//! Particle filter SLAM (FastSLAM) — Rao-Blackwellized particle filter.
//!
//! Each particle carries a robot pose trajectory and an independent landmark
//! map backed by per-landmark EKF estimates. Importance sampling with
//! systematic resampling keeps the particle set focused on high-weight
//! regions of the posterior.

use std::fmt;

// ── 2-D geometry ──────────────────────────────────────────────────

/// 2-D pose `(x, y, θ)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2D {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose2D {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta: wrap_angle(theta) }
    }

    pub fn origin() -> Self { Self::new(0.0, 0.0, 0.0) }
}

impl fmt::Display for Pose2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.4}rad)", self.x, self.y, self.theta)
    }
}

fn wrap_angle(a: f64) -> f64 {
    let mut v = a % (2.0 * std::f64::consts::PI);
    if v > std::f64::consts::PI { v -= 2.0 * std::f64::consts::PI; }
    if v < -std::f64::consts::PI { v += 2.0 * std::f64::consts::PI; }
    v
}

// ── Per-landmark EKF ──────────────────────────────────────────────

/// A single landmark estimated per-particle.
#[derive(Debug, Clone)]
pub struct LandmarkEstimate {
    pub id: usize,
    pub mean: [f64; 2],
    /// 2×2 covariance stored as [a, b, c, d] row-major.
    pub cov: [f64; 4],
}

impl LandmarkEstimate {
    pub fn new(id: usize, x: f64, y: f64, init_cov: f64) -> Self {
        Self {
            id,
            mean: [x, y],
            cov: [init_cov, 0.0, 0.0, init_cov],
        }
    }

    /// Update this landmark EKF with a range-bearing observation from `pose`.
    pub fn update(&mut self, pose: &Pose2D, range: f64, bearing: f64, obs_noise: &[f64; 2]) {
        let dx = self.mean[0] - pose.x;
        let dy = self.mean[1] - pose.y;
        let q = dx * dx + dy * dy;
        let sq = q.sqrt();
        if sq < 1e-12 { return; }

        let pred_range = sq;
        let pred_bearing = wrap_angle(dy.atan2(dx) - pose.theta);

        // Jacobian H (2×2)
        let h00 = dx / sq;
        let h01 = dy / sq;
        let h10 = -dy / q;
        let h11 = dx / q;

        // S = H P H^T + R
        let (p00, p01, p10, p11) = (self.cov[0], self.cov[1], self.cov[2], self.cov[3]);
        let hp00 = h00 * p00 + h01 * p10;
        let hp01 = h00 * p01 + h01 * p11;
        let hp10 = h10 * p00 + h11 * p10;
        let hp11 = h10 * p01 + h11 * p11;

        let s00 = hp00 * h00 + hp01 * h01 + obs_noise[0];
        let s01 = hp00 * h10 + hp01 * h11;
        let s10 = hp10 * h00 + hp11 * h01;
        let s11 = hp10 * h10 + hp11 * h11 + obs_noise[1];

        let det = s00 * s11 - s01 * s10;
        if det.abs() < 1e-15 { return; }
        let inv_det = 1.0 / det;
        let si00 = s11 * inv_det;
        let si01 = -s01 * inv_det;
        let si10 = -s10 * inv_det;
        let si11 = s00 * inv_det;

        // K = P H^T S^{-1}   (2×2)
        let pht00 = p00 * h00 + p01 * h01;
        let pht01 = p00 * h10 + p01 * h11;
        let pht10 = p10 * h00 + p11 * h01;
        let pht11 = p10 * h10 + p11 * h11;

        let k00 = pht00 * si00 + pht01 * si10;
        let k01 = pht00 * si01 + pht01 * si11;
        let k10 = pht10 * si00 + pht11 * si10;
        let k11 = pht10 * si01 + pht11 * si11;

        let innov_r = range - pred_range;
        let innov_b = wrap_angle(bearing - pred_bearing);

        self.mean[0] += k00 * innov_r + k01 * innov_b;
        self.mean[1] += k10 * innov_r + k11 * innov_b;

        // P = (I - K H) P
        let a = 1.0 - k00 * h00 - k01 * h10;
        let b = -(k00 * h01 + k01 * h11);
        let c = -(k10 * h00 + k11 * h10);
        let d = 1.0 - k10 * h01 - k11 * h11;

        self.cov = [
            a * p00 + b * p10,
            a * p01 + b * p11,
            c * p00 + d * p10,
            c * p01 + d * p11,
        ];
    }
}

// ── Observation ───────────────────────────────────────────────────

/// Range-bearing measurement with a landmark ID.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub landmark_id: usize,
    pub range: f64,
    pub bearing: f64,
}

// ── Particle ──────────────────────────────────────────────────────

/// A single particle: pose + per-particle map.
#[derive(Debug, Clone)]
pub struct Particle {
    pub pose: Pose2D,
    pub weight: f64,
    pub landmarks: Vec<LandmarkEstimate>,
}

impl Particle {
    pub fn new(pose: Pose2D) -> Self {
        Self { pose, weight: 1.0, landmarks: Vec::new() }
    }

    pub fn find_landmark(&self, id: usize) -> Option<usize> {
        self.landmarks.iter().position(|l| l.id == id)
    }
}

// ── Simple LCG RNG ────────────────────────────────────────────────

/// Lightweight deterministic RNG (linear congruential).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Gaussian via Box-Muller.
    pub fn next_gaussian(&mut self, mean: f64, stddev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        mean + stddev * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Configuration ─────────────────────────────────────────────────

/// FastSLAM configuration.
#[derive(Debug, Clone)]
pub struct FastSlamConfig {
    pub num_particles: usize,
    pub motion_noise: [f64; 3],
    pub observation_noise: [f64; 2],
    pub initial_landmark_cov: f64,
    pub resample_threshold: f64,
    pub seed: u64,
}

impl Default for FastSlamConfig {
    fn default() -> Self {
        Self {
            num_particles: 100,
            motion_noise: [0.05, 0.05, 0.01],
            observation_noise: [0.1, 0.05],
            initial_landmark_cov: 100.0,
            resample_threshold: 0.5,
            seed: 42,
        }
    }
}

impl FastSlamConfig {
    pub fn new() -> Self { Self::default() }

    pub fn with_num_particles(mut self, n: usize) -> Self {
        self.num_particles = n;
        self
    }

    pub fn with_motion_noise(mut self, noise: [f64; 3]) -> Self {
        self.motion_noise = noise;
        self
    }

    pub fn with_observation_noise(mut self, noise: [f64; 2]) -> Self {
        self.observation_noise = noise;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_resample_threshold(mut self, t: f64) -> Self {
        self.resample_threshold = t;
        self
    }
}

// ── FastSLAM filter ───────────────────────────────────────────────

/// FastSLAM 1.0 with systematic resampling.
#[derive(Debug, Clone)]
pub struct FastSlam {
    pub particles: Vec<Particle>,
    pub config: FastSlamConfig,
    rng: Rng,
}

impl FastSlam {
    pub fn new(config: FastSlamConfig) -> Self {
        let n = config.num_particles;
        let mut rng = Rng::new(config.seed);
        let particles = (0..n)
            .map(|_| {
                let x = rng.next_gaussian(0.0, 0.01);
                let y = rng.next_gaussian(0.0, 0.01);
                let t = rng.next_gaussian(0.0, 0.005);
                Particle::new(Pose2D::new(x, y, t))
            })
            .collect();
        Self { particles, config, rng }
    }

    /// Number of effective particles: N_eff = 1 / Σ w_i².
    pub fn effective_particles(&self) -> f64 {
        let sum_w: f64 = self.particles.iter().map(|p| p.weight).sum();
        if sum_w < 1e-15 { return 0.0; }
        let sum_sq: f64 = self.particles.iter().map(|p| (p.weight / sum_w).powi(2)).sum();
        if sum_sq < 1e-15 { return self.particles.len() as f64; }
        1.0 / sum_sq
    }

    /// Predict: sample new pose for each particle using motion model + noise.
    pub fn predict(&mut self, v: f64, omega: f64, dt: f64) {
        let mn = self.config.motion_noise;
        for p in &mut self.particles {
            let noisy_v = v + self.rng.next_gaussian(0.0, mn[0].sqrt());
            let noisy_omega = omega + self.rng.next_gaussian(0.0, mn[2].sqrt());
            let theta = p.pose.theta;
            if noisy_omega.abs() < 1e-10 {
                p.pose.x += noisy_v * theta.cos() * dt;
                p.pose.y += noisy_v * theta.sin() * dt;
                p.pose.theta = wrap_angle(theta + self.rng.next_gaussian(0.0, mn[1].sqrt()) * dt);
            } else {
                let r = noisy_v / noisy_omega;
                let new_theta = theta + noisy_omega * dt;
                p.pose.x += r * (new_theta.sin() - theta.sin());
                p.pose.y += r * (theta.cos() - new_theta.cos());
                p.pose.theta = wrap_angle(new_theta);
            }
        }
    }

    /// Update each particle with a set of observations.
    pub fn update(&mut self, observations: &[Observation]) {
        let obs_noise = self.config.observation_noise;
        let init_cov = self.config.initial_landmark_cov;

        for particle in &mut self.particles {
            for obs in observations {
                if let Some(idx) = particle.find_landmark(obs.landmark_id) {
                    // Update existing landmark EKF
                    particle.landmarks[idx].update(
                        &particle.pose, obs.range, obs.bearing, &obs_noise,
                    );
                    // Weight proportional to measurement likelihood (simplified)
                    let lm = &particle.landmarks[idx];
                    let dx = lm.mean[0] - particle.pose.x;
                    let dy = lm.mean[1] - particle.pose.y;
                    let pred_r = (dx * dx + dy * dy).sqrt();
                    let pred_b = wrap_angle(dy.atan2(dx) - particle.pose.theta);
                    let dr = obs.range - pred_r;
                    let db = wrap_angle(obs.bearing - pred_b);
                    let likelihood = (-0.5 * (dr * dr / obs_noise[0] + db * db / obs_noise[1])).exp();
                    particle.weight *= likelihood.max(1e-30);
                } else {
                    // New landmark
                    let lx = particle.pose.x + obs.range * (particle.pose.theta + obs.bearing).cos();
                    let ly = particle.pose.y + obs.range * (particle.pose.theta + obs.bearing).sin();
                    particle.landmarks.push(LandmarkEstimate::new(obs.landmark_id, lx, ly, init_cov));
                }
            }
        }
        self.normalize_weights();
    }

    fn normalize_weights(&mut self) {
        let sum: f64 = self.particles.iter().map(|p| p.weight).sum();
        if sum < 1e-30 {
            let w = 1.0 / self.particles.len() as f64;
            for p in &mut self.particles { p.weight = w; }
        } else {
            for p in &mut self.particles { p.weight /= sum; }
        }
    }

    /// Systematic resampling.
    pub fn resample(&mut self) {
        let n = self.particles.len();
        let n_eff = self.effective_particles();
        if n_eff > self.config.resample_threshold * n as f64 {
            return;
        }

        let mut cumulative = Vec::with_capacity(n);
        let mut sum = 0.0;
        for p in &self.particles {
            sum += p.weight;
            cumulative.push(sum);
        }

        let step = 1.0 / n as f64;
        let start = self.rng.next_f64() * step;

        let mut new_particles = Vec::with_capacity(n);
        let mut idx = 0;
        for i in 0..n {
            let target = start + i as f64 * step;
            while idx < n - 1 && cumulative[idx] < target {
                idx += 1;
            }
            let mut cloned = self.particles[idx].clone();
            cloned.weight = 1.0 / n as f64;
            new_particles.push(cloned);
        }

        self.particles = new_particles;
    }

    /// Weighted mean pose across all particles.
    pub fn mean_pose(&self) -> Pose2D {
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sin_sum = 0.0;
        let mut cos_sum = 0.0;
        for p in &self.particles {
            sx += p.weight * p.pose.x;
            sy += p.weight * p.pose.y;
            sin_sum += p.weight * p.pose.theta.sin();
            cos_sum += p.weight * p.pose.theta.cos();
        }
        Pose2D::new(sx, sy, sin_sum.atan2(cos_sum))
    }

    /// Best particle (highest weight).
    pub fn best_particle(&self) -> &Particle {
        self.particles
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
    }

    /// Full step: predict → update → resample.
    pub fn step(&mut self, v: f64, omega: f64, dt: f64, observations: &[Observation]) {
        self.predict(v, omega, dt);
        self.update(observations);
        self.resample();
    }
}

impl fmt::Display for FastSlam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FastSlam(particles={}, n_eff={:.1}, mean_pose={})",
            self.particles.len(),
            self.effective_particles(),
            self.mean_pose()
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> FastSlamConfig {
        FastSlamConfig::new()
            .with_num_particles(20)
            .with_seed(123)
    }

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        assert_eq!(r1.next_f64(), r2.next_f64());
    }

    #[test]
    fn test_rng_gaussian() {
        let mut r = Rng::new(99);
        let vals: Vec<f64> = (0..1000).map(|_| r.next_gaussian(0.0, 1.0)).collect();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_pose_wrap() {
        let p = Pose2D::new(0.0, 0.0, 7.0);
        assert!(p.theta.abs() < std::f64::consts::PI + 0.01);
    }

    #[test]
    fn test_particle_creation() {
        let p = Particle::new(Pose2D::origin());
        assert_eq!(p.weight, 1.0);
        assert!(p.landmarks.is_empty());
    }

    #[test]
    fn test_landmark_estimate_init() {
        let lm = LandmarkEstimate::new(5, 1.0, 2.0, 100.0);
        assert_eq!(lm.id, 5);
        assert_eq!(lm.mean, [1.0, 2.0]);
        assert_eq!(lm.cov[0], 100.0);
    }

    #[test]
    fn test_fastslam_creation() {
        let fs = FastSlam::new(small_config());
        assert_eq!(fs.particles.len(), 20);
    }

    #[test]
    fn test_effective_particles_uniform() {
        let mut fs = FastSlam::new(small_config());
        let w = 1.0 / 20.0;
        for p in &mut fs.particles { p.weight = w; }
        let neff = fs.effective_particles();
        assert!((neff - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_predict_moves_particles() {
        let mut fs = FastSlam::new(small_config());
        let before: Vec<f64> = fs.particles.iter().map(|p| p.pose.x).collect();
        fs.predict(1.0, 0.0, 1.0);
        let after: Vec<f64> = fs.particles.iter().map(|p| p.pose.x).collect();
        let moved = before.iter().zip(&after).any(|(b, a)| (a - b).abs() > 0.01);
        assert!(moved);
    }

    #[test]
    fn test_update_adds_new_landmarks() {
        let mut fs = FastSlam::new(small_config());
        let obs = vec![Observation { landmark_id: 1, range: 5.0, bearing: 0.0 }];
        fs.update(&obs);
        for p in &fs.particles {
            assert_eq!(p.landmarks.len(), 1);
        }
    }

    #[test]
    fn test_update_known_landmark() {
        let mut fs = FastSlam::new(small_config());
        let obs = vec![Observation { landmark_id: 1, range: 5.0, bearing: 0.0 }];
        fs.update(&obs);
        fs.update(&obs);
        // Still just 1 landmark per particle
        for p in &fs.particles {
            assert_eq!(p.landmarks.len(), 1);
        }
    }

    #[test]
    fn test_resample() {
        let mut fs = FastSlam::new(small_config());
        // Concentrate weight on first particle
        fs.particles[0].weight = 0.99;
        for i in 1..fs.particles.len() {
            fs.particles[i].weight = 0.01 / (fs.particles.len() - 1) as f64;
        }
        fs.resample();
        assert_eq!(fs.particles.len(), 20);
    }

    #[test]
    fn test_mean_pose() {
        let mut fs = FastSlam::new(small_config());
        for p in &mut fs.particles { p.pose = Pose2D::new(1.0, 2.0, 0.0); p.weight = 1.0 / 20.0; }
        let mp = fs.mean_pose();
        assert!((mp.x - 1.0).abs() < 0.01);
        assert!((mp.y - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_best_particle() {
        let mut fs = FastSlam::new(small_config());
        fs.particles[5].weight = 999.0;
        let best = fs.best_particle();
        assert!((best.weight - 999.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_full_cycle() {
        let mut fs = FastSlam::new(small_config());
        let obs = vec![Observation { landmark_id: 1, range: 3.0, bearing: 0.5 }];
        fs.step(1.0, 0.1, 0.1, &obs);
        assert_eq!(fs.particles.len(), 20);
    }

    #[test]
    fn test_config_builder() {
        let cfg = FastSlamConfig::new()
            .with_num_particles(50)
            .with_seed(7);
        assert_eq!(cfg.num_particles, 50);
        assert_eq!(cfg.seed, 7);
    }

    #[test]
    fn test_display() {
        let fs = FastSlam::new(small_config());
        let s = format!("{}", fs);
        assert!(s.contains("FastSlam"));
        assert!(s.contains("particles=20"));
    }

    #[test]
    fn test_normalize_weights() {
        let mut fs = FastSlam::new(small_config());
        for (i, p) in fs.particles.iter_mut().enumerate() {
            p.weight = (i + 1) as f64;
        }
        fs.normalize_weights();
        let sum: f64 = fs.particles.iter().map(|p| p.weight).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_landmarks() {
        let mut fs = FastSlam::new(small_config());
        let obs = vec![
            Observation { landmark_id: 1, range: 3.0, bearing: 0.0 },
            Observation { landmark_id: 2, range: 5.0, bearing: 1.0 },
            Observation { landmark_id: 3, range: 4.0, bearing: -0.5 },
        ];
        fs.update(&obs);
        for p in &fs.particles {
            assert_eq!(p.landmarks.len(), 3);
        }
    }
}
