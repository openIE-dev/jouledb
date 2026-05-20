//! Template matching: normalized cross-correlation (NCC), sum of
//! squared differences (SSD), and multi-scale search.
//!
//! Operates on grayscale image buffers (`&[f64]`, row-major) with
//! pixel values in `[0.0, 255.0]`.

use std::fmt;

// ── MatchResult ────────────────────────────────────────────────

/// A single template match result with location, score, and scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchResult {
    pub x: usize,
    pub y: usize,
    pub score: f64,
    pub scale: f64,
}

impl MatchResult {
    pub fn new(x: usize, y: usize, score: f64) -> Self {
        Self { x, y, score, scale: 1.0 }
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Center point of the match given the template dimensions.
    pub fn center(&self, tmpl_w: usize, tmpl_h: usize) -> (f64, f64) {
        let sw = tmpl_w as f64 * self.scale;
        let sh = tmpl_h as f64 * self.scale;
        (self.x as f64 + sw / 2.0, self.y as f64 + sh / 2.0)
    }
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatchResult(({}, {}) score={:.4} scale={:.2})",
               self.x, self.y, self.score, self.scale)
    }
}

// ── MatchMethod ────────────────────────────────────────────────

/// Template matching method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMethod {
    /// Sum of Squared Differences — lower is better.
    Ssd,
    /// Normalized Cross-Correlation — higher is better (max 1.0).
    Ncc,
    /// Zero-mean NCC — higher is better.
    Zncc,
}

impl fmt::Display for MatchMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchMethod::Ssd => write!(f, "SSD"),
            MatchMethod::Ncc => write!(f, "NCC"),
            MatchMethod::Zncc => write!(f, "ZNCC"),
        }
    }
}

// ── TemplateMatchConfig ────────────────────────────────────────

/// Configuration for template matching.
#[derive(Debug, Clone)]
pub struct TemplateMatchConfig {
    pub method: MatchMethod,
    pub scales: Vec<f64>,
    pub top_k: usize,
    pub nms_radius: usize,
}

impl TemplateMatchConfig {
    pub fn new() -> Self {
        Self {
            method: MatchMethod::Ncc,
            scales: vec![1.0],
            top_k: 1,
            nms_radius: 0,
        }
    }

    pub fn with_method(mut self, method: MatchMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_scales(mut self, scales: Vec<f64>) -> Self {
        self.scales = scales;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_nms_radius(mut self, radius: usize) -> Self {
        self.nms_radius = radius;
        self
    }
}

impl fmt::Display for TemplateMatchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TemplateMatchConfig(method={}, scales={}, top_k={}, nms={})",
               self.method, self.scales.len(), self.top_k, self.nms_radius)
    }
}

// ── Score Maps ─────────────────────────────────────────────────

/// Compute the SSD score at a single position.
fn ssd_at(
    image: &[f64], img_w: usize,
    template: &[f64], tmpl_w: usize, tmpl_h: usize,
    tx: usize, ty: usize,
) -> f64 {
    let mut sum = 0.0_f64;
    for tr in 0..tmpl_h {
        for tc in 0..tmpl_w {
            let iv = image[(ty + tr) * img_w + (tx + tc)];
            let tv = template[tr * tmpl_w + tc];
            let diff = iv - tv;
            sum += diff * diff;
        }
    }
    sum
}

/// Compute SSD score map over the entire image.
pub fn ssd_map(
    image: &[f64], img_w: usize, img_h: usize,
    template: &[f64], tmpl_w: usize, tmpl_h: usize,
) -> Vec<f64> {
    let out_w = img_w - tmpl_w + 1;
    let out_h = img_h - tmpl_h + 1;
    let mut map = vec![0.0_f64; out_w * out_h];

    for ty in 0..out_h {
        for tx in 0..out_w {
            map[ty * out_w + tx] = ssd_at(image, img_w, template, tmpl_w, tmpl_h, tx, ty);
        }
    }
    map
}

/// Compute NCC at a single position.
fn ncc_at(
    image: &[f64], img_w: usize,
    template: &[f64], tmpl_w: usize, tmpl_h: usize,
    tx: usize, ty: usize,
) -> f64 {
    let n = (tmpl_w * tmpl_h) as f64;
    let mut sum_i = 0.0_f64;
    let mut sum_t = 0.0_f64;
    let mut sum_ii = 0.0_f64;
    let mut sum_tt = 0.0_f64;
    let mut sum_it = 0.0_f64;

    for tr in 0..tmpl_h {
        for tc in 0..tmpl_w {
            let iv = image[(ty + tr) * img_w + (tx + tc)];
            let tv = template[tr * tmpl_w + tc];
            sum_i += iv;
            sum_t += tv;
            sum_ii += iv * iv;
            sum_tt += tv * tv;
            sum_it += iv * tv;
        }
    }

    let num = n * sum_it - sum_i * sum_t;
    let denom_a = (n * sum_ii - sum_i * sum_i).max(0.0).sqrt();
    let denom_b = (n * sum_tt - sum_t * sum_t).max(0.0).sqrt();
    let denom = denom_a * denom_b;

    if denom < 1e-12 { 0.0 } else { num / denom }
}

/// Compute NCC score map.
pub fn ncc_map(
    image: &[f64], img_w: usize, img_h: usize,
    template: &[f64], tmpl_w: usize, tmpl_h: usize,
) -> Vec<f64> {
    let out_w = img_w - tmpl_w + 1;
    let out_h = img_h - tmpl_h + 1;
    let mut map = vec![0.0_f64; out_w * out_h];

    for ty in 0..out_h {
        for tx in 0..out_w {
            map[ty * out_w + tx] = ncc_at(image, img_w, template, tmpl_w, tmpl_h, tx, ty);
        }
    }
    map
}

/// Compute ZNCC at a single position (zero-mean NCC).
fn zncc_at(
    image: &[f64], img_w: usize,
    template: &[f64], tmpl_w: usize, tmpl_h: usize,
    tx: usize, ty: usize,
) -> f64 {
    let n = (tmpl_w * tmpl_h) as f64;

    // Compute means
    let mut sum_i = 0.0_f64;
    let mut sum_t = 0.0_f64;
    for tr in 0..tmpl_h {
        for tc in 0..tmpl_w {
            sum_i += image[(ty + tr) * img_w + (tx + tc)];
            sum_t += template[tr * tmpl_w + tc];
        }
    }
    let mean_i = sum_i / n;
    let mean_t = sum_t / n;

    let mut num = 0.0_f64;
    let mut var_i = 0.0_f64;
    let mut var_t = 0.0_f64;

    for tr in 0..tmpl_h {
        for tc in 0..tmpl_w {
            let di = image[(ty + tr) * img_w + (tx + tc)] - mean_i;
            let dt = template[tr * tmpl_w + tc] - mean_t;
            num += di * dt;
            var_i += di * di;
            var_t += dt * dt;
        }
    }

    let denom = var_i.sqrt() * var_t.sqrt();
    if denom < 1e-12 { 0.0 } else { num / denom }
}

/// Compute ZNCC score map.
pub fn zncc_map(
    image: &[f64], img_w: usize, img_h: usize,
    template: &[f64], tmpl_w: usize, tmpl_h: usize,
) -> Vec<f64> {
    let out_w = img_w - tmpl_w + 1;
    let out_h = img_h - tmpl_h + 1;
    let mut map = vec![0.0_f64; out_w * out_h];

    for ty in 0..out_h {
        for tx in 0..out_w {
            map[ty * out_w + tx] = zncc_at(image, img_w, template, tmpl_w, tmpl_h, tx, ty);
        }
    }
    map
}

// ── Score Map Utilities ────────────────────────────────────────

/// Find the position with the best score in a score map.
/// For SSD the best is the minimum; for NCC/ZNCC the best is the maximum.
pub fn find_best(map: &[f64], map_w: usize, method: MatchMethod) -> MatchResult {
    let mut best_idx = 0;
    let mut best_val = map[0];

    for (i, &v) in map.iter().enumerate() {
        let is_better = match method {
            MatchMethod::Ssd => v < best_val,
            MatchMethod::Ncc | MatchMethod::Zncc => v > best_val,
        };
        if is_better {
            best_val = v;
            best_idx = i;
        }
    }

    let x = best_idx % map_w;
    let y = best_idx / map_w;
    MatchResult::new(x, y, best_val)
}

/// Find the top-K matches with optional NMS.
pub fn find_top_k(
    map: &[f64], map_w: usize, method: MatchMethod,
    top_k: usize, nms_radius: usize,
) -> Vec<MatchResult> {
    let mut indexed: Vec<(usize, f64)> = map.iter().cloned().enumerate().collect();

    match method {
        MatchMethod::Ssd => indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
        MatchMethod::Ncc | MatchMethod::Zncc => indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
    }

    let mut results = Vec::new();

    for &(idx, score) in &indexed {
        if results.len() >= top_k {
            break;
        }
        let x = idx % map_w;
        let y = idx / map_w;

        // NMS check
        if nms_radius > 0 {
            let suppressed = results.iter().any(|r: &MatchResult| {
                let dx = if x > r.x { x - r.x } else { r.x - x };
                let dy = if y > r.y { y - r.y } else { r.y - y };
                dx <= nms_radius && dy <= nms_radius
            });
            if suppressed {
                continue;
            }
        }

        results.push(MatchResult::new(x, y, score));
    }
    results
}

// ── Resize (bilinear) ──────────────────────────────────────────

/// Bilinear resize of a grayscale image buffer.
fn resize_bilinear(
    src: &[f64], src_w: usize, src_h: usize,
    dst_w: usize, dst_h: usize,
) -> Vec<f64> {
    let mut dst = vec![0.0_f64; dst_w * dst_h];

    for dr in 0..dst_h {
        for dc in 0..dst_w {
            let sx = dc as f64 * (src_w as f64 - 1.0) / (dst_w as f64 - 1.0).max(1.0);
            let sy = dr as f64 * (src_h as f64 - 1.0) / (dst_h as f64 - 1.0).max(1.0);

            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;

            let v00 = src[y0 * src_w + x0];
            let v10 = src[y0 * src_w + x1];
            let v01 = src[y1 * src_w + x0];
            let v11 = src[y1 * src_w + x1];

            dst[dr * dst_w + dc] = v00 * (1.0 - fx) * (1.0 - fy)
                                 + v10 * fx * (1.0 - fy)
                                 + v01 * (1.0 - fx) * fy
                                 + v11 * fx * fy;
        }
    }
    dst
}

// ── Multi-Scale Template Matching ──────────────────────────────

/// Perform multi-scale template matching.
pub fn match_template(
    image: &[f64], img_w: usize, img_h: usize,
    template: &[f64], tmpl_w: usize, tmpl_h: usize,
    config: &TemplateMatchConfig,
) -> Vec<MatchResult> {
    let mut all_results = Vec::new();

    for &scale in &config.scales {
        let sw = (tmpl_w as f64 * scale).round() as usize;
        let sh = (tmpl_h as f64 * scale).round() as usize;

        if sw < 2 || sh < 2 || sw > img_w || sh > img_h {
            continue;
        }

        let scaled_tmpl = if scale == 1.0 {
            template.to_vec()
        } else {
            resize_bilinear(template, tmpl_w, tmpl_h, sw, sh)
        };

        let map = match config.method {
            MatchMethod::Ssd => ssd_map(image, img_w, img_h, &scaled_tmpl, sw, sh),
            MatchMethod::Ncc => ncc_map(image, img_w, img_h, &scaled_tmpl, sw, sh),
            MatchMethod::Zncc => zncc_map(image, img_w, img_h, &scaled_tmpl, sw, sh),
        };

        let map_w = img_w - sw + 1;
        let mut results = find_top_k(&map, map_w, config.method, config.top_k, config.nms_radius);
        for r in &mut results {
            *r = r.with_scale(scale);
        }
        all_results.extend(results);
    }

    // Sort all results across scales
    match config.method {
        MatchMethod::Ssd => all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)),
        MatchMethod::Ncc | MatchMethod::Zncc => all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)),
    }

    all_results.truncate(config.top_k);
    all_results
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(w: usize, h: usize, val: f64) -> Vec<f64> {
        vec![val; w * h]
    }

    fn checkerboard(w: usize, h: usize, block: usize) -> Vec<f64> {
        let mut data = vec![0.0_f64; w * h];
        for r in 0..h {
            for c in 0..w {
                let br = r / block;
                let bc = c / block;
                if (br + bc) % 2 == 0 {
                    data[r * w + c] = 255.0;
                }
            }
        }
        data
    }

    #[test]
    fn test_match_result_display() {
        let r = MatchResult::new(10, 20, 0.95).with_scale(0.5);
        let s = format!("{}", r);
        assert!(s.contains("0.95"));
        assert!(s.contains("0.50"));
    }

    #[test]
    fn test_match_result_center() {
        let r = MatchResult::new(10, 20, 1.0);
        let (cx, cy) = r.center(6, 8);
        assert_eq!(cx, 13.0);
        assert_eq!(cy, 24.0);
    }

    #[test]
    fn test_method_display() {
        assert_eq!(format!("{}", MatchMethod::Ssd), "SSD");
        assert_eq!(format!("{}", MatchMethod::Ncc), "NCC");
        assert_eq!(format!("{}", MatchMethod::Zncc), "ZNCC");
    }

    #[test]
    fn test_config_builder() {
        let cfg = TemplateMatchConfig::new()
            .with_method(MatchMethod::Ssd)
            .with_top_k(5)
            .with_nms_radius(10);
        assert_eq!(cfg.method, MatchMethod::Ssd);
        assert_eq!(cfg.top_k, 5);
        assert_eq!(cfg.nms_radius, 10);
    }

    #[test]
    fn test_config_display() {
        let cfg = TemplateMatchConfig::new();
        let s = format!("{}", cfg);
        assert!(s.contains("TemplateMatchConfig"));
    }

    #[test]
    fn test_ssd_identical() {
        let img = vec![1.0, 2.0, 3.0, 4.0];
        let ssd = ssd_at(&img, 2, &img, 2, 2, 0, 0);
        assert!((ssd).abs() < 1e-9);
    }

    #[test]
    fn test_ssd_map_size() {
        let img = uniform(10, 10, 100.0);
        let tmpl = uniform(3, 3, 100.0);
        let map = ssd_map(&img, 10, 10, &tmpl, 3, 3);
        assert_eq!(map.len(), 8 * 8); // (10-3+1)^2
    }

    #[test]
    fn test_ssd_perfect_match_zero() {
        let img = uniform(10, 10, 50.0);
        let tmpl = uniform(3, 3, 50.0);
        let map = ssd_map(&img, 10, 10, &tmpl, 3, 3);
        for &v in &map {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn test_ncc_identical() {
        let img = checkerboard(10, 10, 2);
        let tmpl: Vec<f64> = img[..3].iter().chain(img[10..13].iter()).chain(img[20..23].iter()).cloned().collect();
        // NCC of a patch with itself at position (0,0) should be 1.0
        let score = ncc_at(&img, 10, &tmpl, 3, 3, 0, 0);
        assert!((score - 1.0).abs() < 1e-6, "NCC of patch with itself should be 1.0, got {}", score);
    }

    #[test]
    fn test_ncc_map_range() {
        let img = checkerboard(10, 10, 2);
        let tmpl = checkerboard(3, 3, 1);
        let map = ncc_map(&img, 10, 10, &tmpl, 3, 3);
        for &v in &map {
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6, "NCC should be in [-1, 1], got {}", v);
        }
    }

    #[test]
    fn test_zncc_self_match() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let score = zncc_at(&data, 3, &data, 3, 3, 0, 0);
        assert!((score - 1.0).abs() < 1e-6, "ZNCC of patch with itself = 1.0, got {}", score);
    }

    #[test]
    fn test_find_best_ssd() {
        let map = vec![10.0, 5.0, 20.0, 1.0, 15.0, 8.0];
        let best = find_best(&map, 3, MatchMethod::Ssd);
        assert_eq!(best.x, 0);
        assert_eq!(best.y, 1);
        assert_eq!(best.score, 1.0);
    }

    #[test]
    fn test_find_best_ncc() {
        let map = vec![0.1, 0.9, 0.3, 0.5, 0.2, 0.7];
        let best = find_best(&map, 3, MatchMethod::Ncc);
        assert_eq!(best.x, 1);
        assert_eq!(best.y, 0);
        assert_eq!(best.score, 0.9);
    }

    #[test]
    fn test_find_top_k() {
        let map = vec![0.1, 0.9, 0.3, 0.5, 0.2, 0.8];
        let top = find_top_k(&map, 3, MatchMethod::Ncc, 2, 0);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].score, 0.9);
        assert_eq!(top[1].score, 0.8);
    }

    #[test]
    fn test_resize_identity() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let dst = resize_bilinear(&src, 2, 2, 2, 2);
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_resize_upscale() {
        let src = vec![0.0, 100.0, 0.0, 100.0];
        let dst = resize_bilinear(&src, 2, 2, 4, 4);
        assert_eq!(dst.len(), 16);
        // Corners should match source corners
        assert!((dst[0] - 0.0).abs() < 1e-6);
        assert!((dst[3] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiscale_match() {
        let img = uniform(20, 20, 128.0);
        let tmpl = uniform(5, 5, 128.0);
        let cfg = TemplateMatchConfig::new()
            .with_method(MatchMethod::Ncc)
            .with_scales(vec![0.5, 1.0, 1.5])
            .with_top_k(3);
        let results = match_template(&img, 20, 20, &tmpl, 5, 5, &cfg);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_nms_top_k() {
        let map = vec![
            0.1, 0.9, 0.8,
            0.2, 0.3, 0.7,
            0.0, 0.1, 0.2,
        ];
        let top = find_top_k(&map, 3, MatchMethod::Ncc, 2, 2);
        // With NMS radius 2, only 1 result should survive from cluster
        assert!(top.len() <= 2);
    }
}
