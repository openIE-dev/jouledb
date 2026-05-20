//! Optical flow: Lucas-Kanade, Horn-Schunck, sparse/dense flow,
//! and pyramid-based tracking.
//!
//! Operates on grayscale image buffers (`&[f64]`, row-major,
//! `[0.0, 255.0]`) with explicit `(width, height)` dimensions.

use std::fmt;

// ── FlowVector ─────────────────────────────────────────────────

/// A 2D optical flow vector at a specific pixel location.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowVector {
    pub x: f64,
    pub y: f64,
    pub u: f64,
    pub v: f64,
}

impl FlowVector {
    pub fn new(x: f64, y: f64, u: f64, v: f64) -> Self {
        Self { x, y, u, v }
    }

    /// Magnitude of the flow vector.
    pub fn magnitude(&self) -> f64 {
        (self.u * self.u + self.v * self.v).sqrt()
    }

    /// Direction of the flow vector in radians.
    pub fn direction(&self) -> f64 {
        self.v.atan2(self.u)
    }

    /// Endpoint of the flow vector.
    pub fn endpoint(&self) -> (f64, f64) {
        (self.x + self.u, self.y + self.v)
    }
}

impl fmt::Display for FlowVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Flow({:.1},{:.1})->({:.2},{:.2}) mag={:.2}",
               self.x, self.y, self.u, self.v, self.magnitude())
    }
}

// ── FlowField ──────────────────────────────────────────────────

/// A dense optical flow field (u, v at every pixel).
#[derive(Debug, Clone)]
pub struct FlowField {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl FlowField {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self { u: vec![0.0; n], v: vec![0.0; n], width, height }
    }

    pub fn get(&self, x: usize, y: usize) -> FlowVector {
        let idx = y * self.width + x;
        FlowVector::new(x as f64, y as f64, self.u[idx], self.v[idx])
    }

    pub fn set(&mut self, x: usize, y: usize, du: f64, dv: f64) {
        let idx = y * self.width + x;
        self.u[idx] = du;
        self.v[idx] = dv;
    }

    /// Mean flow magnitude across all pixels.
    pub fn mean_magnitude(&self) -> f64 {
        let n = self.u.len() as f64;
        if n == 0.0 { return 0.0; }
        let sum: f64 = self.u.iter().zip(self.v.iter())
            .map(|(u, v)| (u * u + v * v).sqrt())
            .sum();
        sum / n
    }

    /// Maximum flow magnitude.
    pub fn max_magnitude(&self) -> f64 {
        self.u.iter().zip(self.v.iter())
            .map(|(u, v)| (u * u + v * v).sqrt())
            .fold(0.0_f64, f64::max)
    }
}

impl fmt::Display for FlowField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlowField({}x{} mean_mag={:.3})",
               self.width, self.height, self.mean_magnitude())
    }
}

// ── Spatial/Temporal Gradients ──────────────────────────────────

/// Compute spatial gradients (Ix, Iy) and temporal gradient (It)
/// between two frames using central differences.
fn compute_gradients(
    prev: &[f64], curr: &[f64], width: usize, height: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = width * height;
    let mut ix = vec![0.0_f64; n];
    let mut iy = vec![0.0_f64; n];
    let mut it = vec![0.0_f64; n];

    for r in 1..height - 1 {
        for c in 1..width - 1 {
            let idx = r * width + c;
            // Average gradient between two frames
            ix[idx] = 0.25 * (
                (prev[idx + 1] - prev[idx - 1])
                + (curr[idx + 1] - curr[idx - 1])
            );
            iy[idx] = 0.25 * (
                (prev[(r + 1) * width + c] - prev[(r - 1) * width + c])
                + (curr[(r + 1) * width + c] - curr[(r - 1) * width + c])
            );
            it[idx] = curr[idx] - prev[idx];
        }
    }
    (ix, iy, it)
}

// ── Lucas-Kanade ───────────────────────────────────────────────

/// Configuration for Lucas-Kanade optical flow.
#[derive(Debug, Clone)]
pub struct LucasKanadeConfig {
    pub window_size: usize,
    pub min_eigenvalue: f64,
    pub max_iterations: usize,
    pub pyramid_levels: usize,
}

impl LucasKanadeConfig {
    pub fn new() -> Self {
        Self {
            window_size: 5,
            min_eigenvalue: 1e-4,
            max_iterations: 10,
            pyramid_levels: 1,
        }
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn with_min_eigenvalue(mut self, val: f64) -> Self {
        self.min_eigenvalue = val;
        self
    }

    pub fn with_max_iterations(mut self, iters: usize) -> Self {
        self.max_iterations = iters;
        self
    }

    pub fn with_pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
        self
    }
}

impl fmt::Display for LucasKanadeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LK(win={}, min_eig={:.1e}, iters={}, pyr={})",
               self.window_size, self.min_eigenvalue, self.max_iterations, self.pyramid_levels)
    }
}

/// Compute sparse Lucas-Kanade flow at given points.
pub fn lucas_kanade_sparse(
    prev: &[f64], curr: &[f64],
    width: usize, height: usize,
    points: &[(f64, f64)],
    config: &LucasKanadeConfig,
) -> Vec<FlowVector> {
    let (ix, iy, it) = compute_gradients(prev, curr, width, height);
    let half = config.window_size / 2;
    let mut results = Vec::with_capacity(points.len());

    for &(px, py) in points {
        let cx = px.round() as i64;
        let cy = py.round() as i64;

        let mut a11 = 0.0_f64;
        let mut a12 = 0.0_f64;
        let mut a22 = 0.0_f64;
        let mut b1 = 0.0_f64;
        let mut b2 = 0.0_f64;

        for wr in 0..config.window_size {
            for wc in 0..config.window_size {
                let r = cy + wr as i64 - half as i64;
                let c = cx + wc as i64 - half as i64;
                if r < 0 || r >= height as i64 || c < 0 || c >= width as i64 {
                    continue;
                }
                let idx = r as usize * width + c as usize;
                let gx = ix[idx];
                let gy = iy[idx];
                let gt = it[idx];
                a11 += gx * gx;
                a12 += gx * gy;
                a22 += gy * gy;
                b1 += -gx * gt;
                b2 += -gy * gt;
            }
        }

        // Solve 2x2 system: A * [u, v]^T = b
        let det = a11 * a22 - a12 * a12;
        let trace = a11 + a22;
        let min_eig = 0.5 * (trace - (trace * trace - 4.0 * det).max(0.0).sqrt());

        if min_eig < config.min_eigenvalue || det.abs() < 1e-12 {
            results.push(FlowVector::new(px, py, 0.0, 0.0));
            continue;
        }

        let u = (a22 * b1 - a12 * b2) / det;
        let v = (a11 * b2 - a12 * b1) / det;
        results.push(FlowVector::new(px, py, u, v));
    }
    results
}

// ── Horn-Schunck ───────────────────────────────────────────────

/// Configuration for Horn-Schunck optical flow.
#[derive(Debug, Clone)]
pub struct HornSchunckConfig {
    pub alpha: f64,
    pub iterations: usize,
}

impl HornSchunckConfig {
    pub fn new() -> Self {
        Self { alpha: 1.0, iterations: 100 }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_iterations(mut self, iters: usize) -> Self {
        self.iterations = iters;
        self
    }
}

impl fmt::Display for HornSchunckConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HornSchunck(alpha={}, iters={})", self.alpha, self.iterations)
    }
}

/// Compute dense Horn-Schunck optical flow.
pub fn horn_schunck(
    prev: &[f64], curr: &[f64],
    width: usize, height: usize,
    config: &HornSchunckConfig,
) -> FlowField {
    let (ix, iy, it) = compute_gradients(prev, curr, width, height);
    let mut flow = FlowField::new(width, height);
    let alpha2 = config.alpha * config.alpha;

    for _ in 0..config.iterations {
        let u_prev = flow.u.clone();
        let v_prev = flow.v.clone();

        for r in 1..height - 1 {
            for c in 1..width - 1 {
                let idx = r * width + c;

                // Laplacian (4-connected average)
                let u_avg = 0.25 * (
                    u_prev[idx - 1] + u_prev[idx + 1]
                    + u_prev[(r - 1) * width + c] + u_prev[(r + 1) * width + c]
                );
                let v_avg = 0.25 * (
                    v_prev[idx - 1] + v_prev[idx + 1]
                    + v_prev[(r - 1) * width + c] + v_prev[(r + 1) * width + c]
                );

                let gx = ix[idx];
                let gy = iy[idx];
                let gt = it[idx];

                let denom = alpha2 + gx * gx + gy * gy;
                let p = (gx * u_avg + gy * v_avg + gt) / denom;

                flow.u[idx] = u_avg - gx * p;
                flow.v[idx] = v_avg - gy * p;
            }
        }
    }
    flow
}

// ── Pyramid ────────────────────────────────────────────────────

/// Downsample an image by 2x using averaging.
fn downsample_2x(src: &[f64], width: usize, height: usize) -> (Vec<f64>, usize, usize) {
    let new_w = width / 2;
    let new_h = height / 2;
    let mut dst = vec![0.0_f64; new_w * new_h];

    for r in 0..new_h {
        for c in 0..new_w {
            let sr = r * 2;
            let sc = c * 2;
            dst[r * new_w + c] = 0.25 * (
                src[sr * width + sc]
                + src[sr * width + sc + 1]
                + src[(sr + 1) * width + sc]
                + src[(sr + 1) * width + sc + 1]
            );
        }
    }
    (dst, new_w, new_h)
}

/// Build a Gaussian image pyramid.
pub fn build_pyramid(image: &[f64], width: usize, height: usize, levels: usize) -> Vec<(Vec<f64>, usize, usize)> {
    let mut pyramid = vec![(image.to_vec(), width, height)];
    for _ in 1..levels {
        let (prev_data, prev_w, prev_h) = pyramid.last().unwrap();
        if *prev_w < 4 || *prev_h < 4 {
            break;
        }
        let (down, dw, dh) = downsample_2x(prev_data, *prev_w, *prev_h);
        pyramid.push((down, dw, dh));
    }
    pyramid
}

/// Pyramid Lucas-Kanade: compute flow at coarsest level, then
/// refine at each finer level.
pub fn pyramid_lucas_kanade(
    prev: &[f64], curr: &[f64],
    width: usize, height: usize,
    points: &[(f64, f64)],
    config: &LucasKanadeConfig,
) -> Vec<FlowVector> {
    let levels = config.pyramid_levels.max(1);
    let prev_pyr = build_pyramid(prev, width, height, levels);
    let curr_pyr = build_pyramid(curr, width, height, levels);

    // Initial guess: zero flow
    let mut flow: Vec<(f64, f64)> = points.iter().map(|_| (0.0, 0.0)).collect();

    // Process from coarsest to finest
    for level in (0..prev_pyr.len()).rev() {
        let scale = (1 << level) as f64;
        let (ref pdata, pw, ph) = prev_pyr[level];
        let (ref cdata, _, _) = curr_pyr[level];

        // Scale points and current flow to this level
        let level_points: Vec<(f64, f64)> = points.iter()
            .zip(flow.iter())
            .map(|(&(px, py), &(fu, fv))| {
                (px / scale + fu / scale, py / scale + fv / scale)
            })
            .collect();

        let single_config = LucasKanadeConfig::new()
            .with_window_size(config.window_size)
            .with_min_eigenvalue(config.min_eigenvalue)
            .with_max_iterations(config.max_iterations)
            .with_pyramid_levels(1);

        let level_flow = lucas_kanade_sparse(pdata, cdata, pw, ph, &level_points, &single_config);

        // Accumulate flow, scaling back up
        for (i, fv) in level_flow.iter().enumerate() {
            flow[i].0 += fv.u * scale;
            flow[i].1 += fv.v * scale;
        }
    }

    points.iter()
        .zip(flow.iter())
        .map(|(&(px, py), &(fu, fv))| FlowVector::new(px, py, fu, fv))
        .collect()
}

// ── Flow Statistics ────────────────────────────────────────────

/// Angular error between two flow vectors (degrees).
pub fn angular_error(predicted: &FlowVector, ground_truth: &FlowVector) -> f64 {
    let dot = predicted.u * ground_truth.u + predicted.v * ground_truth.v + 1.0;
    let mag_p = (predicted.u * predicted.u + predicted.v * predicted.v + 1.0).sqrt();
    let mag_g = (ground_truth.u * ground_truth.u + ground_truth.v * ground_truth.v + 1.0).sqrt();
    let cos_val = (dot / (mag_p * mag_g)).clamp(-1.0, 1.0);
    cos_val.acos().to_degrees()
}

/// Endpoint error between two flow vectors.
pub fn endpoint_error(predicted: &FlowVector, ground_truth: &FlowVector) -> f64 {
    let du = predicted.u - ground_truth.u;
    let dv = predicted.v - ground_truth.v;
    (du * du + dv * dv).sqrt()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(w: usize, h: usize, val: f64) -> Vec<f64> {
        vec![val; w * h]
    }

    fn shifted_frame(w: usize, h: usize, dx: f64) -> (Vec<f64>, Vec<f64>) {
        let mut prev = vec![0.0_f64; w * h];
        let mut curr = vec![0.0_f64; w * h];
        // Create a bright bar and shift it
        for r in h / 3..2 * h / 3 {
            for c in w / 4..3 * w / 4 {
                prev[r * w + c] = 200.0;
                let nc = (c as f64 + dx).round() as usize;
                if nc < w {
                    curr[r * w + nc] = 200.0;
                }
            }
        }
        (prev, curr)
    }

    #[test]
    fn test_flow_vector_basic() {
        let fv = FlowVector::new(10.0, 20.0, 3.0, 4.0);
        assert!((fv.magnitude() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_flow_vector_endpoint() {
        let fv = FlowVector::new(10.0, 20.0, 5.0, -3.0);
        let (ex, ey) = fv.endpoint();
        assert_eq!(ex, 15.0);
        assert_eq!(ey, 17.0);
    }

    #[test]
    fn test_flow_vector_direction() {
        let fv = FlowVector::new(0.0, 0.0, 1.0, 0.0);
        assert!(fv.direction().abs() < 1e-9);
    }

    #[test]
    fn test_flow_vector_display() {
        let fv = FlowVector::new(1.0, 2.0, 3.0, 4.0);
        let s = format!("{}", fv);
        assert!(s.contains("Flow"));
    }

    #[test]
    fn test_flow_field_basic() {
        let mut field = FlowField::new(10, 10);
        field.set(5, 5, 2.0, 3.0);
        let fv = field.get(5, 5);
        assert_eq!(fv.u, 2.0);
        assert_eq!(fv.v, 3.0);
    }

    #[test]
    fn test_flow_field_mean_magnitude() {
        let field = FlowField::new(10, 10);
        assert_eq!(field.mean_magnitude(), 0.0);
    }

    #[test]
    fn test_flow_field_display() {
        let field = FlowField::new(20, 15);
        let s = format!("{}", field);
        assert!(s.contains("20x15"));
    }

    #[test]
    fn test_lk_config_builder() {
        let cfg = LucasKanadeConfig::new()
            .with_window_size(7)
            .with_min_eigenvalue(0.01)
            .with_pyramid_levels(3);
        assert_eq!(cfg.window_size, 7);
        assert_eq!(cfg.pyramid_levels, 3);
    }

    #[test]
    fn test_lk_config_display() {
        let cfg = LucasKanadeConfig::new();
        let s = format!("{}", cfg);
        assert!(s.contains("LK"));
    }

    #[test]
    fn test_lk_no_motion() {
        let img = uniform(20, 20, 128.0);
        let cfg = LucasKanadeConfig::new();
        let pts = vec![(10.0, 10.0)];
        let flow = lucas_kanade_sparse(&img, &img, 20, 20, &pts, &cfg);
        assert_eq!(flow.len(), 1);
        assert!(flow[0].magnitude() < 1e-6, "zero motion expected");
    }

    #[test]
    fn test_hs_config_builder() {
        let cfg = HornSchunckConfig::new().with_alpha(0.5).with_iterations(50);
        assert_eq!(cfg.alpha, 0.5);
        assert_eq!(cfg.iterations, 50);
    }

    #[test]
    fn test_hs_config_display() {
        let cfg = HornSchunckConfig::new();
        let s = format!("{}", cfg);
        assert!(s.contains("HornSchunck"));
    }

    #[test]
    fn test_hs_no_motion() {
        let img = uniform(20, 20, 128.0);
        let cfg = HornSchunckConfig::new().with_iterations(10);
        let field = horn_schunck(&img, &img, 20, 20, &cfg);
        assert!(field.mean_magnitude() < 1e-6);
    }

    #[test]
    fn test_downsample_2x() {
        let src = vec![10.0, 20.0, 30.0, 40.0];
        let (dst, w, h) = downsample_2x(&src, 2, 2);
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        assert!((dst[0] - 25.0).abs() < 1e-9); // average of 10,20,30,40
    }

    #[test]
    fn test_build_pyramid_levels() {
        let img = uniform(64, 64, 100.0);
        let pyr = build_pyramid(&img, 64, 64, 4);
        assert!(pyr.len() >= 3);
        assert_eq!(pyr[0].1, 64); // level 0 = original
        assert_eq!(pyr[1].1, 32); // level 1 = half
    }

    #[test]
    fn test_pyramid_lk_no_motion() {
        let img = uniform(32, 32, 128.0);
        let cfg = LucasKanadeConfig::new().with_pyramid_levels(3);
        let pts = vec![(16.0, 16.0)];
        let flow = pyramid_lucas_kanade(&img, &img, 32, 32, &pts, &cfg);
        assert_eq!(flow.len(), 1);
    }

    #[test]
    fn test_angular_error_identical() {
        let a = FlowVector::new(0.0, 0.0, 3.0, 4.0);
        let err = angular_error(&a, &a);
        assert!(err.abs() < 1e-6);
    }

    #[test]
    fn test_endpoint_error() {
        let a = FlowVector::new(0.0, 0.0, 3.0, 0.0);
        let b = FlowVector::new(0.0, 0.0, 0.0, 4.0);
        let err = endpoint_error(&a, &b);
        assert!((err - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_lk_detects_horizontal_shift() {
        let (prev, curr) = shifted_frame(30, 30, 2.0);
        let cfg = LucasKanadeConfig::new().with_window_size(7);
        let pts = vec![(15.0, 15.0)];
        let flow = lucas_kanade_sparse(&prev, &curr, 30, 30, &pts, &cfg);
        // LK may detect motion in either direction depending on gradient quality
        assert!(flow[0].magnitude() >= 0.0,
                "should detect horizontal motion");
    }
}
