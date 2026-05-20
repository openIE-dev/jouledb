// Importance sampling utilities for Monte Carlo rendering.
// Distributions, PDF computation, MIS, and low-discrepancy sequences.

use std::fmt;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;
const INV_PI: f64 = 1.0 / PI;
const INV_TWO_PI: f64 = 1.0 / TWO_PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-15 { Self::zero() } else { self * (1.0 / l) }
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

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

/// Build orthonormal basis from a normal vector.
pub fn build_onb(n: Vec3) -> (Vec3, Vec3, Vec3) {
    let w = n.normalized();
    let a = if w.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let v = w.cross(a).normalized();
    let u = w.cross(v);
    (u, v, w)
}

/// Convert local (tangent space) direction to world space using ONB.
fn local_to_world(local: Vec3, u: Vec3, v: Vec3, w: Vec3) -> Vec3 {
    (u * local.x + v * local.y + w * local.z).normalized()
}

// ─── Hemisphere and shape sampling ───

/// Sample uniform hemisphere direction given two random values in [0,1).
pub fn sample_uniform_hemisphere(r1: f64, r2: f64, normal: Vec3) -> Vec3 {
    let (u, v, w) = build_onb(normal);
    let cos_theta = r1;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TWO_PI * r2;
    let local = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    local_to_world(local, u, v, w)
}

/// PDF for uniform hemisphere sampling.
pub fn pdf_uniform_hemisphere() -> f64 {
    INV_TWO_PI
}

/// Cosine-weighted hemisphere sampling (Malley's method).
pub fn sample_cosine_hemisphere(r1: f64, r2: f64, normal: Vec3) -> Vec3 {
    let (u, v, w) = build_onb(normal);
    // Malley's method: sample disc then project up
    let r = r1.sqrt();
    let phi = TWO_PI * r2;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - r1).max(0.0).sqrt();
    let local = Vec3::new(x, y, z);
    local_to_world(local, u, v, w)
}

/// PDF for cosine-weighted hemisphere sampling.
pub fn pdf_cosine_hemisphere(cos_theta: f64) -> f64 {
    if cos_theta <= 0.0 { return 0.0; }
    cos_theta * INV_PI
}

/// GGX NDF sampling in tangent space. Returns half-vector.
pub fn sample_ggx(r1: f64, r2: f64, roughness: f64, normal: Vec3) -> Vec3 {
    let (u, v, w) = build_onb(normal);
    let a = roughness * roughness;
    let a2 = a * a;
    let cos_theta = ((1.0 - r1) / (r1 * (a2 - 1.0) + 1.0)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TWO_PI * r2;
    let local = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    local_to_world(local, u, v, w)
}

/// GGX NDF value D(h).
pub fn ggx_ndf(cos_theta: f64, roughness: f64) -> f64 {
    let a = roughness * roughness;
    let a2 = a * a;
    let cos2 = cos_theta * cos_theta;
    let denom = cos2 * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// PDF for GGX sampling: D(h) * cos_theta_h / (4 * dot(wo, h)).
pub fn pdf_ggx(cos_theta_h: f64, roughness: f64, wo_dot_h: f64) -> f64 {
    if wo_dot_h <= 0.0 { return 0.0; }
    ggx_ndf(cos_theta_h, roughness) * cos_theta_h / (4.0 * wo_dot_h)
}

/// Beckmann NDF sampling.
pub fn sample_beckmann(r1: f64, r2: f64, roughness: f64, normal: Vec3) -> Vec3 {
    let (u, v, w) = build_onb(normal);
    let a = roughness * roughness;
    let log_sample = if r1 < 1e-15 { 0.0 } else { r1.ln() };
    let tan2_theta = -a * a * log_sample;
    let cos_theta = 1.0 / (1.0 + tan2_theta).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TWO_PI * r2;
    let local = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    local_to_world(local, u, v, w)
}

/// Beckmann NDF value.
pub fn beckmann_ndf(cos_theta: f64, roughness: f64) -> f64 {
    if cos_theta <= 0.0 { return 0.0; }
    let a = roughness * roughness;
    let a2 = a * a;
    let cos2 = cos_theta * cos_theta;
    let tan2 = (1.0 - cos2) / cos2;
    let exp_term = (-tan2 / a2).exp();
    exp_term / (PI * a2 * cos2 * cos2)
}

/// Sample uniform direction inside a cone with half-angle cos_theta_max.
pub fn sample_uniform_cone(r1: f64, r2: f64, cos_theta_max: f64, normal: Vec3) -> Vec3 {
    let (u, v, w) = build_onb(normal);
    let cos_theta = 1.0 - r1 * (1.0 - cos_theta_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TWO_PI * r2;
    let local = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    local_to_world(local, u, v, w)
}

/// PDF for uniform cone sampling.
pub fn pdf_uniform_cone(cos_theta_max: f64) -> f64 {
    if cos_theta_max >= 1.0 { return 0.0; }
    1.0 / (TWO_PI * (1.0 - cos_theta_max))
}

/// Sample a uniform point on a disc (returns x, y).
pub fn sample_uniform_disc(r1: f64, r2: f64) -> (f64, f64) {
    let r = r1.sqrt();
    let theta = TWO_PI * r2;
    (r * theta.cos(), r * theta.sin())
}

/// Sample a uniform point on a triangle using barycentric coordinates.
/// Returns (u, v) where the third coordinate w = 1 - u - v.
pub fn sample_uniform_triangle(r1: f64, r2: f64) -> (f64, f64) {
    let sqrt_r1 = r1.sqrt();
    let u = 1.0 - sqrt_r1;
    let v = r2 * sqrt_r1;
    (u, v)
}

// ─── Multiple Importance Sampling ───

/// Balance heuristic for MIS.
pub fn mis_balance_heuristic(pdf_f: f64, pdf_g: f64, n_f: usize, n_g: usize) -> f64 {
    let wf = n_f as f64 * pdf_f;
    let wg = n_g as f64 * pdf_g;
    let denom = wf + wg;
    if denom < 1e-15 { return 0.0; }
    wf / denom
}

/// Power heuristic for MIS (beta = 2 is standard).
pub fn mis_power_heuristic(pdf_f: f64, pdf_g: f64, n_f: usize, n_g: usize, beta: f64) -> f64 {
    let wf = (n_f as f64 * pdf_f).powf(beta);
    let wg = (n_g as f64 * pdf_g).powf(beta);
    let denom = wf + wg;
    if denom < 1e-15 { return 0.0; }
    wf / denom
}

/// Multi-strategy MIS weight: given this strategy's pdf and all pdfs.
pub fn mis_multi_balance(this_pdf: f64, this_count: usize, all_pdfs: &[(f64, usize)]) -> f64 {
    let numerator = this_count as f64 * this_pdf;
    let denominator: f64 = all_pdfs.iter().map(|(p, n)| *n as f64 * p).sum();
    if denominator < 1e-15 { return 0.0; }
    numerator / denominator
}

// ─── Low-discrepancy sequences ───

/// Halton sequence value for given index and base.
pub fn halton(index: usize, base: usize) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    let mut i = index;
    while i > 0 {
        result += f * (i % base) as f64;
        i /= base;
        f /= base as f64;
    }
    result
}

/// Hammersley point set: (i/N, radical_inverse_base2(i)).
pub fn hammersley(index: usize, total: usize) -> (f64, f64) {
    (index as f64 / total as f64, radical_inverse_base2(index))
}

fn radical_inverse_base2(mut n: usize) -> f64 {
    let mut result = 0.0;
    let mut f = 0.5;
    while n > 0 {
        if n & 1 == 1 {
            result += f;
        }
        n >>= 1;
        f *= 0.5;
    }
    result
}

/// Generate stratified jittered samples for an N×N grid.
pub fn stratified_jittered_2d(n: usize, rng_fn: &mut dyn FnMut() -> f64) -> Vec<(f64, f64)> {
    let inv_n = 1.0 / n as f64;
    let mut samples = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let sx = (i as f64 + rng_fn()) * inv_n;
            let sy = (j as f64 + rng_fn()) * inv_n;
            samples.push((sx, sy));
        }
    }
    samples
}

/// Generate 1D stratified jittered samples.
pub fn stratified_jittered_1d(n: usize, rng_fn: &mut dyn FnMut() -> f64) -> Vec<f64> {
    let inv_n = 1.0 / n as f64;
    (0..n).map(|i| (i as f64 + rng_fn()) * inv_n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    fn vec3_approx_eq(a: Vec3, b: Vec3, eps: f64) -> bool {
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
    }

    struct TestRng { state: u64 }
    impl TestRng {
        fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }
        fn next_f64(&mut self) -> f64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn test_uniform_hemisphere_on_hemisphere() {
        let mut rng = TestRng::new(42);
        let n = Vec3::new(0.0, 1.0, 0.0);
        for _ in 0..200 {
            let r1 = rng.next_f64();
            let r2 = rng.next_f64();
            let d = sample_uniform_hemisphere(r1, r2, n);
            assert!(approx_eq(d.length(), 1.0, 1e-6), "len={}", d.length());
            assert!(d.dot(n) >= -1e-6, "dot={}", d.dot(n));
        }
    }

    #[test]
    fn test_cosine_hemisphere_on_hemisphere() {
        let mut rng = TestRng::new(77);
        let n = Vec3::new(0.0, 0.0, 1.0);
        for _ in 0..200 {
            let r1 = rng.next_f64();
            let r2 = rng.next_f64();
            let d = sample_cosine_hemisphere(r1, r2, n);
            assert!(approx_eq(d.length(), 1.0, 1e-6));
            assert!(d.dot(n) >= -1e-6);
        }
    }

    #[test]
    fn test_cosine_hemisphere_pdf_integral() {
        // Monte Carlo estimate of integral of pdf over hemisphere should be ~1
        let mut rng = TestRng::new(99);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let mut sum = 0.0;
        let samples = 10000;
        for _ in 0..samples {
            let r1 = rng.next_f64();
            let r2 = rng.next_f64();
            let d = sample_cosine_hemisphere(r1, r2, n);
            let cos_theta = d.dot(n).max(0.0);
            let pdf = pdf_cosine_hemisphere(cos_theta);
            if pdf > 1e-12 {
                sum += 1.0 / pdf; // integration of 1 over hemisphere
            }
        }
        let integral = sum / samples as f64;
        // Should be close to hemisphere area = 2*pi
        assert!(approx_eq(integral, TWO_PI, 0.3), "integral={}", integral);
    }

    #[test]
    fn test_pdf_uniform_hemisphere() {
        let pdf = pdf_uniform_hemisphere();
        assert!(approx_eq(pdf, INV_TWO_PI, 1e-9));
    }

    #[test]
    fn test_ggx_sample_on_hemisphere() {
        let mut rng = TestRng::new(42);
        let n = Vec3::new(0.0, 1.0, 0.0);
        for _ in 0..100 {
            let r1 = rng.next_f64();
            let r2 = rng.next_f64();
            let h = sample_ggx(r1, r2, 0.5, n);
            assert!(approx_eq(h.length(), 1.0, 1e-6));
            assert!(h.dot(n) >= -1e-6);
        }
    }

    #[test]
    fn test_ggx_ndf_peak_at_normal() {
        // For smooth surface (low roughness), NDF peaks when cos_theta=1
        let d_at_1 = ggx_ndf(1.0, 0.1);
        let d_at_half = ggx_ndf(0.5, 0.1);
        assert!(d_at_1 > d_at_half);
    }

    #[test]
    fn test_beckmann_sample_on_hemisphere() {
        let mut rng = TestRng::new(42);
        let n = Vec3::new(0.0, 0.0, 1.0);
        for _ in 0..100 {
            let r1 = rng.next_f64().max(1e-10);
            let r2 = rng.next_f64();
            let h = sample_beckmann(r1, r2, 0.3, n);
            assert!(approx_eq(h.length(), 1.0, 1e-6));
        }
    }

    #[test]
    fn test_beckmann_ndf_positive() {
        let d = beckmann_ndf(0.8, 0.3);
        assert!(d > 0.0);
        assert!(d.is_finite());
    }

    #[test]
    fn test_uniform_cone_contains_axis() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let d = sample_uniform_cone(0.0, 0.0, 0.5, n);
        // r1=0 -> cos_theta=1, should be parallel to normal
        assert!(d.dot(n) > 0.99);
    }

    #[test]
    fn test_pdf_uniform_cone() {
        let pdf = pdf_uniform_cone(0.5);
        assert!(pdf > 0.0);
        let expected = 1.0 / (TWO_PI * 0.5);
        assert!(approx_eq(pdf, expected, 1e-9));
    }

    #[test]
    fn test_sample_uniform_disc() {
        let mut rng = TestRng::new(42);
        for _ in 0..100 {
            let (x, y) = sample_uniform_disc(rng.next_f64(), rng.next_f64());
            let r = (x * x + y * y).sqrt();
            assert!(r <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn test_sample_uniform_triangle() {
        let mut rng = TestRng::new(42);
        for _ in 0..100 {
            let (u, v) = sample_uniform_triangle(rng.next_f64(), rng.next_f64());
            assert!(u >= 0.0 && v >= 0.0);
            assert!(u + v <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn test_mis_balance_heuristic() {
        let w = mis_balance_heuristic(1.0, 1.0, 1, 1);
        assert!(approx_eq(w, 0.5, 1e-9));
    }

    #[test]
    fn test_mis_balance_dominant() {
        let w = mis_balance_heuristic(10.0, 0.1, 1, 1);
        assert!(w > 0.9);
    }

    #[test]
    fn test_mis_power_heuristic() {
        let w = mis_power_heuristic(1.0, 1.0, 1, 1, 2.0);
        assert!(approx_eq(w, 0.5, 1e-9));
    }

    #[test]
    fn test_mis_multi_balance() {
        let all = vec![(0.5, 1), (0.5, 1), (0.5, 1)];
        let w = mis_multi_balance(0.5, 1, &all);
        assert!(approx_eq(w, 1.0 / 3.0, 1e-9));
    }

    #[test]
    fn test_halton_base2() {
        assert!(approx_eq(halton(1, 2), 0.5, 1e-9));
        assert!(approx_eq(halton(2, 2), 0.25, 1e-9));
        assert!(approx_eq(halton(3, 2), 0.75, 1e-9));
    }

    #[test]
    fn test_halton_base3() {
        assert!(approx_eq(halton(1, 3), 1.0 / 3.0, 1e-9));
        assert!(approx_eq(halton(2, 3), 2.0 / 3.0, 1e-9));
        assert!(approx_eq(halton(3, 3), 1.0 / 9.0, 1e-9));
    }

    #[test]
    fn test_halton_range() {
        for i in 1..100 {
            let v = halton(i, 2);
            assert!(v >= 0.0 && v < 1.0, "halton({}, 2) = {}", i, v);
        }
    }

    #[test]
    fn test_hammersley() {
        let (u, v) = hammersley(0, 4);
        assert!(approx_eq(u, 0.0, 1e-9));
        let (u, v) = hammersley(2, 4);
        assert!(approx_eq(u, 0.5, 1e-9));
        assert!(approx_eq(v, 0.25, 1e-9));
    }

    #[test]
    fn test_stratified_jittered_2d() {
        let mut seed = 42u64;
        let rng_fn = &mut || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        let samples = stratified_jittered_2d(4, rng_fn);
        assert_eq!(samples.len(), 16);
        for (sx, sy) in &samples {
            assert!(*sx >= 0.0 && *sx <= 1.0);
            assert!(*sy >= 0.0 && *sy <= 1.0);
        }
    }

    #[test]
    fn test_stratified_jittered_1d() {
        let mut seed = 42u64;
        let rng_fn = &mut || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        let samples = stratified_jittered_1d(8, rng_fn);
        assert_eq!(samples.len(), 8);
        for s in &samples {
            assert!(*s >= 0.0 && *s <= 1.0);
        }
    }

    #[test]
    fn test_onb_orthogonal() {
        let n = Vec3::new(0.3, 0.7, 0.5).normalized();
        let (u, v, w) = build_onb(n);
        assert!(approx_eq(u.dot(v), 0.0, 1e-6));
        assert!(approx_eq(u.dot(w), 0.0, 1e-6));
        assert!(approx_eq(v.dot(w), 0.0, 1e-6));
        assert!(approx_eq(u.length(), 1.0, 1e-6));
        assert!(approx_eq(v.length(), 1.0, 1e-6));
    }

    #[test]
    fn test_pdf_cosine_hemisphere_zero_below() {
        assert!(approx_eq(pdf_cosine_hemisphere(-0.5), 0.0, 1e-9));
        assert!(approx_eq(pdf_cosine_hemisphere(0.0), 0.0, 1e-9));
    }

    #[test]
    fn test_pdf_ggx_zero_below() {
        assert!(approx_eq(pdf_ggx(0.5, 0.3, -0.1), 0.0, 1e-9));
        assert!(approx_eq(pdf_ggx(0.5, 0.3, 0.0), 0.0, 1e-9));
    }

    #[test]
    fn test_mis_balance_zero_pdfs() {
        let w = mis_balance_heuristic(0.0, 0.0, 1, 1);
        assert!(approx_eq(w, 0.0, 1e-9));
    }
}
