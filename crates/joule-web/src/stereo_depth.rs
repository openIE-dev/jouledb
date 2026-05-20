//! Stereo depth estimation: disparity computation, block matching,
//! semi-global matching, and depth map generation.
//!
//! Operates on rectified grayscale stereo pairs (`&[f64]`, row-major,
//! values in `[0.0, 255.0]`).

use std::fmt;

// ── StereoParams ───────────────────────────────────────────────

/// Stereo camera parameters for depth computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoParams {
    /// Focal length in pixels.
    pub focal_length: f64,
    /// Baseline distance between cameras (meters).
    pub baseline: f64,
    /// Principal point X.
    pub cx: f64,
    /// Principal point Y.
    pub cy: f64,
}

impl StereoParams {
    pub fn new(focal_length: f64, baseline: f64) -> Self {
        Self { focal_length, baseline, cx: 0.0, cy: 0.0 }
    }

    pub fn with_principal_point(mut self, cx: f64, cy: f64) -> Self {
        self.cx = cx;
        self.cy = cy;
        self
    }

    /// Convert disparity (pixels) to depth (meters).
    pub fn disparity_to_depth(&self, disparity: f64) -> f64 {
        if disparity.abs() < 1e-9 {
            return f64::INFINITY;
        }
        self.focal_length * self.baseline / disparity
    }

    /// Convert depth (meters) to disparity (pixels).
    pub fn depth_to_disparity(&self, depth: f64) -> f64 {
        if depth.abs() < 1e-9 {
            return f64::INFINITY;
        }
        self.focal_length * self.baseline / depth
    }
}

impl fmt::Display for StereoParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StereoParams(f={:.1} b={:.3} cx={:.1} cy={:.1})",
               self.focal_length, self.baseline, self.cx, self.cy)
    }
}

// ── DisparityMap ───────────────────────────────────────────────

/// A disparity map with per-pixel disparity values.
#[derive(Debug, Clone)]
pub struct DisparityMap {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl DisparityMap {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0.0; width * height], width, height }
    }

    pub fn from_data(data: Vec<f64>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self { data, width, height }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        self.data[y * self.width + x] = val;
    }

    /// Convert to a depth map using stereo parameters.
    pub fn to_depth_map(&self, params: &StereoParams) -> DepthMap {
        let mut depth = DepthMap::new(self.width, self.height);
        for (i, &d) in self.data.iter().enumerate() {
            depth.data[i] = params.disparity_to_depth(d);
        }
        depth
    }

    /// Mean disparity of valid (non-zero) pixels.
    pub fn mean_disparity(&self) -> f64 {
        let valid: Vec<f64> = self.data.iter().cloned().filter(|v| *v > 0.0).collect();
        if valid.is_empty() { return 0.0; }
        valid.iter().sum::<f64>() / valid.len() as f64
    }

    /// Maximum disparity.
    pub fn max_disparity(&self) -> f64 {
        self.data.iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Fraction of pixels with valid (non-zero) disparity.
    pub fn fill_rate(&self) -> f64 {
        let valid = self.data.iter().filter(|&&v| v > 0.0).count();
        valid as f64 / self.data.len() as f64
    }
}

impl fmt::Display for DisparityMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisparityMap({}x{} max={:.1} fill={:.1}%)",
               self.width, self.height, self.max_disparity(), self.fill_rate() * 100.0)
    }
}

// ── DepthMap ───────────────────────────────────────────────────

/// A depth map with per-pixel depth values in meters.
#[derive(Debug, Clone)]
pub struct DepthMap {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl DepthMap {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0.0; width * height], width, height }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    /// Minimum finite depth.
    pub fn min_depth(&self) -> f64 {
        self.data.iter().cloned()
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum finite depth.
    pub fn max_depth(&self) -> f64 {
        self.data.iter().cloned()
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::max)
    }

    /// Point cloud: list of (x_3d, y_3d, z_3d) for finite-depth pixels.
    pub fn to_point_cloud(&self, params: &StereoParams) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        for r in 0..self.height {
            for c in 0..self.width {
                let z = self.data[r * self.width + c];
                if z.is_finite() && z > 0.0 {
                    let x = (c as f64 - params.cx) * z / params.focal_length;
                    let y = (r as f64 - params.cy) * z / params.focal_length;
                    points.push((x, y, z));
                }
            }
        }
        points
    }
}

impl fmt::Display for DepthMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DepthMap({}x{} range=[{:.2}, {:.2}]m)",
               self.width, self.height, self.min_depth(), self.max_depth())
    }
}

// ── Block Matching ─────────────────────────────────────────────

/// Configuration for block matching stereo.
#[derive(Debug, Clone)]
pub struct BlockMatchConfig {
    pub block_size: usize,
    pub min_disparity: usize,
    pub max_disparity: usize,
    pub uniqueness_ratio: f64,
}

impl BlockMatchConfig {
    pub fn new() -> Self {
        Self {
            block_size: 9,
            min_disparity: 0,
            max_disparity: 64,
            uniqueness_ratio: 0.95,
        }
    }

    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size | 1; // ensure odd
        self
    }

    pub fn with_disparity_range(mut self, min_val: usize, max_val: usize) -> Self {
        self.min_disparity = min_val;
        self.max_disparity = max_val;
        self
    }

    pub fn with_uniqueness_ratio(mut self, ratio: f64) -> Self {
        self.uniqueness_ratio = ratio;
        self
    }
}

impl fmt::Display for BlockMatchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockMatch(blk={} disp=[{},{}] uniq={:.2})",
               self.block_size, self.min_disparity, self.max_disparity,
               self.uniqueness_ratio)
    }
}

/// Sum of absolute differences for a block centered at (lx, ly)
/// in the left image and (rx, ry) in the right image.
fn sad_block(
    left: &[f64], right: &[f64], width: usize,
    lx: usize, ly: usize, rx: usize, ry: usize,
    block_size: usize,
) -> f64 {
    let half = block_size / 2;
    let mut sad = 0.0_f64;

    for br in 0..block_size {
        for bc in 0..block_size {
            let lr = ly + br - half;
            let lc = lx + bc - half;
            let rr = ry + br - half;
            let rc = rx + bc - half;
            sad += (left[lr * width + lc] - right[rr * width + rc]).abs();
        }
    }
    sad
}

/// Block matching stereo disparity.
pub fn block_match(
    left: &[f64], right: &[f64],
    width: usize, height: usize,
    config: &BlockMatchConfig,
) -> DisparityMap {
    let half = config.block_size / 2;
    let mut disp_map = DisparityMap::new(width, height);

    for r in half..height - half {
        for c in (half + config.max_disparity)..width - half {
            let mut best_sad = f64::INFINITY;
            let mut second_sad = f64::INFINITY;
            let mut best_d = 0;

            for d in config.min_disparity..=config.max_disparity {
                if c < d + half { continue; }
                let cost = sad_block(left, right, width, c, r, c - d, r, config.block_size);
                if cost < best_sad {
                    second_sad = best_sad;
                    best_sad = cost;
                    best_d = d;
                } else if cost < second_sad {
                    second_sad = cost;
                }
            }

            // Uniqueness check
            if second_sad > 1e-12 && best_sad / second_sad < config.uniqueness_ratio {
                disp_map.set(c, r, best_d as f64);
            }
        }
    }
    disp_map
}

// ── Semi-Global Matching (simplified) ──────────────────────────

/// Configuration for semi-global matching.
#[derive(Debug, Clone)]
pub struct SgmConfig {
    pub block_size: usize,
    pub min_disparity: usize,
    pub max_disparity: usize,
    pub penalty_small: f64,
    pub penalty_large: f64,
    pub num_paths: usize,
}

impl SgmConfig {
    pub fn new() -> Self {
        Self {
            block_size: 5,
            min_disparity: 0,
            max_disparity: 64,
            penalty_small: 10.0,
            penalty_large: 120.0,
            num_paths: 4,
        }
    }

    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size | 1;
        self
    }

    pub fn with_disparity_range(mut self, min_val: usize, max_val: usize) -> Self {
        self.min_disparity = min_val;
        self.max_disparity = max_val;
        self
    }

    pub fn with_penalties(mut self, small: f64, large: f64) -> Self {
        self.penalty_small = small;
        self.penalty_large = large;
        self
    }

    pub fn with_num_paths(mut self, paths: usize) -> Self {
        self.num_paths = paths;
        self
    }
}

impl fmt::Display for SgmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SGM(blk={} disp=[{},{}] P1={:.0} P2={:.0} paths={})",
               self.block_size, self.min_disparity, self.max_disparity,
               self.penalty_small, self.penalty_large, self.num_paths)
    }
}

/// Semi-global matching stereo (simplified 4-path aggregation).
pub fn sgm_match(
    left: &[f64], right: &[f64],
    width: usize, height: usize,
    config: &SgmConfig,
) -> DisparityMap {
    let n_disp = config.max_disparity - config.min_disparity + 1;
    let half = config.block_size / 2;

    // Compute pixel-wise matching cost volume
    let mut cost_vol = vec![f64::MAX; width * height * n_disp];

    for r in half..height - half {
        for c in (half + config.max_disparity)..width - half {
            for di in 0..n_disp {
                let d = config.min_disparity + di;
                if c < d + half { continue; }
                let cost = sad_block(left, right, width, c, r, c - d, r, config.block_size);
                cost_vol[(r * width + c) * n_disp + di] = cost;
            }
        }
    }

    // Path directions: right, down, right-down, left-down
    let directions: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (-1, 1)];
    let num_paths = config.num_paths.min(directions.len());

    let mut agg_cost = vec![0.0_f64; width * height * n_disp];

    for path in 0..num_paths {
        let (dr, dc) = directions[path];
        let mut path_cost = vec![0.0_f64; width * height * n_disp];

        // Determine traversal order
        let rows: Vec<usize> = if dr >= 0 { (0..height).collect() } else { (0..height).rev().collect() };
        let cols: Vec<usize> = if dc >= 0 { (0..width).collect() } else { (0..width).rev().collect() };

        for &r in &rows {
            for &c in &cols {
                let pr = r as i32 - dr;
                let pc = c as i32 - dc;

                let pixel_idx = r * width + c;

                for di in 0..n_disp {
                    let raw = cost_vol[pixel_idx * n_disp + di];
                    if raw == f64::MAX {
                        path_cost[pixel_idx * n_disp + di] = raw;
                        continue;
                    }

                    if pr < 0 || pr >= height as i32 || pc < 0 || pc >= width as i32 {
                        path_cost[pixel_idx * n_disp + di] = raw;
                        continue;
                    }

                    let prev_idx = pr as usize * width + pc as usize;

                    // Min of: same disparity, +/-1 with P1, any other with P2
                    let same = path_cost[prev_idx * n_disp + di];
                    let minus1 = if di > 0 { path_cost[prev_idx * n_disp + di - 1] + config.penalty_small } else { f64::MAX };
                    let plus1 = if di + 1 < n_disp { path_cost[prev_idx * n_disp + di + 1] + config.penalty_small } else { f64::MAX };

                    let mut min_prev = f64::MAX;
                    for k in 0..n_disp {
                        let v = path_cost[prev_idx * n_disp + k];
                        if v < min_prev { min_prev = v; }
                    }
                    let any_other = min_prev + config.penalty_large;

                    let best_prev = same.min(minus1).min(plus1).min(any_other);

                    // Subtract min of previous to keep values bounded
                    path_cost[pixel_idx * n_disp + di] = raw + best_prev - min_prev;
                }
            }
        }

        // Accumulate
        for i in 0..agg_cost.len() {
            if path_cost[i] < f64::MAX && agg_cost[i] < f64::MAX {
                agg_cost[i] += path_cost[i];
            }
        }
    }

    // Winner-take-all
    let mut disp_map = DisparityMap::new(width, height);
    for r in 0..height {
        for c in 0..width {
            let pixel_idx = r * width + c;
            let mut best_di = 0;
            let mut best_cost = f64::MAX;
            for di in 0..n_disp {
                let cost = agg_cost[pixel_idx * n_disp + di];
                if cost < best_cost {
                    best_cost = cost;
                    best_di = di;
                }
            }
            if best_cost < f64::MAX {
                disp_map.set(c, r, (config.min_disparity + best_di) as f64);
            }
        }
    }
    disp_map
}

// ── Disparity Refinement ───────────────────────────────────────

/// Median filter on a disparity map for noise removal.
pub fn median_filter_disparity(disp: &DisparityMap, kernel: usize) -> DisparityMap {
    let half = kernel / 2;
    let mut out = DisparityMap::new(disp.width, disp.height);

    for r in half..disp.height - half {
        for c in half..disp.width - half {
            let mut window = Vec::with_capacity(kernel * kernel);
            for kr in 0..kernel {
                for kc in 0..kernel {
                    window.push(disp.get(c + kc - half, r + kr - half));
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out.set(c, r, window[window.len() / 2]);
        }
    }
    out
}

/// Left-right consistency check. Invalidates disparities that differ
/// by more than `threshold` when computed from the other view.
pub fn lr_consistency_check(
    left_disp: &DisparityMap, right_disp: &DisparityMap, threshold: f64,
) -> DisparityMap {
    let mut out = left_disp.clone();
    for r in 0..left_disp.height {
        for c in 0..left_disp.width {
            let d = left_disp.get(c, r);
            if d <= 0.0 { continue; }
            let rc = c as f64 - d;
            if rc < 0.0 || rc >= right_disp.width as f64 {
                out.set(c, r, 0.0);
                continue;
            }
            let rd = right_disp.get(rc.round() as usize, r);
            if (d - rd).abs() > threshold {
                out.set(c, r, 0.0);
            }
        }
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(w: usize, h: usize, val: f64) -> Vec<f64> {
        vec![val; w * h]
    }

    #[test]
    fn test_stereo_params_display() {
        let p = StereoParams::new(500.0, 0.12);
        let s = format!("{}", p);
        assert!(s.contains("500.0"));
    }

    #[test]
    fn test_disparity_to_depth() {
        let p = StereoParams::new(500.0, 0.1);
        let depth = p.disparity_to_depth(50.0);
        assert!((depth - 1.0).abs() < 1e-9); // 500*0.1/50 = 1.0
    }

    #[test]
    fn test_depth_to_disparity() {
        let p = StereoParams::new(500.0, 0.1);
        let disp = p.depth_to_disparity(1.0);
        assert!((disp - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_disparity_zero_infinity() {
        let p = StereoParams::new(500.0, 0.1);
        assert!(p.disparity_to_depth(0.0).is_infinite());
    }

    #[test]
    fn test_stereo_params_principal_point() {
        let p = StereoParams::new(500.0, 0.1).with_principal_point(320.0, 240.0);
        assert_eq!(p.cx, 320.0);
        assert_eq!(p.cy, 240.0);
    }

    #[test]
    fn test_disparity_map_basic() {
        let mut dm = DisparityMap::new(10, 10);
        dm.set(5, 5, 42.0);
        assert_eq!(dm.get(5, 5), 42.0);
    }

    #[test]
    fn test_disparity_map_stats() {
        let data = vec![0.0, 10.0, 20.0, 30.0];
        let dm = DisparityMap::from_data(data, 2, 2);
        assert!((dm.mean_disparity() - 20.0).abs() < 1e-9); // mean of 10,20,30
        assert_eq!(dm.max_disparity(), 30.0);
        assert!((dm.fill_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_disparity_map_display() {
        let dm = DisparityMap::new(20, 15);
        let s = format!("{}", dm);
        assert!(s.contains("20x15"));
    }

    #[test]
    fn test_depth_map_display() {
        let dm = DepthMap::new(10, 10);
        let s = format!("{}", dm);
        assert!(s.contains("DepthMap"));
    }

    #[test]
    fn test_disparity_to_depth_map() {
        let mut dm = DisparityMap::new(3, 3);
        dm.set(1, 1, 50.0);
        let params = StereoParams::new(500.0, 0.1);
        let depth = dm.to_depth_map(&params);
        assert!((depth.get(1, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_cloud() {
        let mut depth = DepthMap::new(3, 3);
        depth.data[4] = 2.0; // center pixel at (1,1)
        let params = StereoParams::new(100.0, 0.1).with_principal_point(1.0, 1.0);
        let cloud = depth.to_point_cloud(&params);
        assert_eq!(cloud.len(), 1);
        assert!((cloud[0].2 - 2.0).abs() < 1e-9); // z = 2.0
    }

    #[test]
    fn test_block_match_config_builder() {
        let cfg = BlockMatchConfig::new()
            .with_block_size(7)
            .with_disparity_range(0, 32)
            .with_uniqueness_ratio(0.9);
        assert_eq!(cfg.block_size, 7);
        assert_eq!(cfg.max_disparity, 32);
    }

    #[test]
    fn test_block_match_config_display() {
        let cfg = BlockMatchConfig::new();
        let s = format!("{}", cfg);
        assert!(s.contains("BlockMatch"));
    }

    #[test]
    fn test_block_match_identical() {
        let img = uniform(40, 40, 128.0);
        let cfg = BlockMatchConfig::new().with_block_size(5).with_disparity_range(0, 10);
        let dm = block_match(&img, &img, 40, 40, &cfg);
        // Identical images → disparity 0 everywhere
        assert!(dm.mean_disparity() < 1.0);
    }

    #[test]
    fn test_sgm_config_builder() {
        let cfg = SgmConfig::new()
            .with_block_size(3)
            .with_disparity_range(0, 32)
            .with_penalties(8.0, 100.0)
            .with_num_paths(8);
        assert_eq!(cfg.block_size, 3);
        assert_eq!(cfg.penalty_small, 8.0);
        assert_eq!(cfg.num_paths, 8);
    }

    #[test]
    fn test_sgm_config_display() {
        let cfg = SgmConfig::new();
        let s = format!("{}", cfg);
        assert!(s.contains("SGM"));
    }

    #[test]
    fn test_median_filter() {
        let mut dm = DisparityMap::new(5, 5);
        // Set all to 10, one outlier at center
        for v in &mut dm.data { *v = 10.0; }
        dm.set(2, 2, 100.0);
        let filtered = median_filter_disparity(&dm, 3);
        // Median should suppress the outlier
        assert!((filtered.get(2, 2) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_lr_consistency() {
        let mut left = DisparityMap::new(10, 10);
        let mut right = DisparityMap::new(10, 10);
        left.set(5, 5, 3.0);
        right.set(2, 5, 3.0); // consistent: 5 - 3 = 2
        let checked = lr_consistency_check(&left, &right, 1.0);
        assert!(checked.get(5, 5) > 0.0);
    }

    #[test]
    fn test_lr_consistency_fails() {
        let mut left = DisparityMap::new(10, 10);
        let mut right = DisparityMap::new(10, 10);
        left.set(5, 5, 3.0);
        right.set(2, 5, 10.0); // inconsistent
        let checked = lr_consistency_check(&left, &right, 1.0);
        assert_eq!(checked.get(5, 5), 0.0);
    }

    #[test]
    fn test_sad_block_identical() {
        let img = uniform(10, 10, 50.0);
        let sad = sad_block(&img, &img, 10, 5, 5, 5, 5, 3);
        assert!(sad.abs() < 1e-9);
    }
}
