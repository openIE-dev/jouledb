//! Feature detection: Harris corners, FAST keypoints, ORB descriptors,
//! keypoint matching, and non-maximum suppression.
//!
//! All algorithms operate on grayscale image buffers represented as
//! `&[f64]` with explicit `(width, height)` dimensions. Pixel values
//! are expected in `[0.0, 255.0]`.

use std::fmt;

// ── Keypoint ───────────────────────────────────────────────────

/// A detected keypoint with sub-pixel position, response strength,
/// scale octave, and dominant orientation (radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub x: f64,
    pub y: f64,
    pub response: f64,
    pub octave: u32,
    pub angle: f64,
}

impl Keypoint {
    pub fn new(x: f64, y: f64, response: f64) -> Self {
        Self { x, y, response, octave: 0, angle: 0.0 }
    }

    pub fn with_octave(mut self, octave: u32) -> Self {
        self.octave = octave;
        self
    }

    pub fn with_angle(mut self, angle: f64) -> Self {
        self.angle = angle;
        self
    }

    /// Euclidean distance to another keypoint.
    pub fn distance(&self, other: &Keypoint) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl fmt::Display for Keypoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Keypoint({:.1}, {:.1} r={:.4} oct={} ang={:.2})",
               self.x, self.y, self.response, self.octave, self.angle)
    }
}

// ── Descriptor ─────────────────────────────────────────────────

/// A binary descriptor (256-bit) attached to a keypoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Descriptor {
    pub keypoint: Keypoint,
    pub bits: [u8; 32],
}

impl Descriptor {
    pub fn new(keypoint: Keypoint, bits: [u8; 32]) -> Self {
        Self { keypoint, bits }
    }

    /// Hamming distance (number of differing bits) to another descriptor.
    pub fn hamming_distance(&self, other: &Descriptor) -> u32 {
        self.bits.iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Descriptor(kp={}, ham_weight={})",
               self.keypoint,
               self.bits.iter().map(|b| b.count_ones()).sum::<u32>())
    }
}

// ── Match ──────────────────────────────────────────────────────

/// A match between two descriptors with distance.
#[derive(Debug, Clone)]
pub struct Match {
    pub query_idx: usize,
    pub train_idx: usize,
    pub distance: u32,
}

impl Match {
    pub fn new(query_idx: usize, train_idx: usize, distance: u32) -> Self {
        Self { query_idx, train_idx, distance }
    }
}

impl fmt::Display for Match {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Match(q={} t={} d={})", self.query_idx, self.train_idx, self.distance)
    }
}

// ── Harris Corner Detector ─────────────────────────────────────

/// Configuration for Harris corner detection.
#[derive(Debug, Clone)]
pub struct HarrisConfig {
    pub k: f64,
    pub threshold: f64,
    pub window_size: usize,
    pub nms_radius: usize,
}

impl HarrisConfig {
    pub fn new() -> Self {
        Self { k: 0.04, threshold: 1e6, window_size: 3, nms_radius: 3 }
    }

    pub fn with_k(mut self, k: f64) -> Self {
        self.k = k;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn with_nms_radius(mut self, radius: usize) -> Self {
        self.nms_radius = radius;
        self
    }
}

impl fmt::Display for HarrisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HarrisConfig(k={}, thr={}, win={}, nms={})",
               self.k, self.threshold, self.window_size, self.nms_radius)
    }
}

/// Compute image gradients using Sobel 3x3.
fn sobel_gradients(pixels: &[f64], width: usize, height: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ix = vec![0.0_f64; width * height];
    let mut iy = vec![0.0_f64; width * height];

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let idx = |r: usize, c: usize| pixels[r * width + c];
            let gx = -idx(row - 1, col - 1) + idx(row - 1, col + 1)
                   - 2.0 * idx(row, col - 1) + 2.0 * idx(row, col + 1)
                   - idx(row + 1, col - 1) + idx(row + 1, col + 1);
            let gy = -idx(row - 1, col - 1) - 2.0 * idx(row - 1, col)
                   - idx(row - 1, col + 1)
                   + idx(row + 1, col - 1) + 2.0 * idx(row + 1, col)
                   + idx(row + 1, col + 1);
            ix[row * width + col] = gx;
            iy[row * width + col] = gy;
        }
    }
    (ix, iy)
}

/// Harris corner response map.
fn harris_response(
    ix: &[f64], iy: &[f64], width: usize, height: usize, config: &HarrisConfig,
) -> Vec<f64> {
    let mut response = vec![0.0_f64; width * height];
    let half = config.window_size / 2;

    for row in half..height.saturating_sub(half) {
        for col in half..width.saturating_sub(half) {
            let mut sxx = 0.0_f64;
            let mut syy = 0.0_f64;
            let mut sxy = 0.0_f64;

            for wr in 0..config.window_size {
                for wc in 0..config.window_size {
                    let r = row + wr - half;
                    let c = col + wc - half;
                    let pos = r * width + c;
                    sxx += ix[pos] * ix[pos];
                    syy += iy[pos] * iy[pos];
                    sxy += ix[pos] * iy[pos];
                }
            }

            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            response[row * width + col] = det - config.k * trace * trace;
        }
    }
    response
}

/// Detect Harris corners in a grayscale image.
pub fn detect_harris(
    pixels: &[f64], width: usize, height: usize, config: &HarrisConfig,
) -> Vec<Keypoint> {
    let (ix, iy) = sobel_gradients(pixels, width, height);
    let resp = harris_response(&ix, &iy, width, height, config);

    let mut keypoints = Vec::new();
    let r = config.nms_radius;

    for row in r..height.saturating_sub(r) {
        for col in r..width.saturating_sub(r) {
            let val = resp[row * width + col];
            if val < config.threshold {
                continue;
            }
            let mut is_max = true;
            'nms: for dr in 0..=2 * r {
                for dc in 0..=2 * r {
                    if dr == r && dc == r {
                        continue;
                    }
                    let nr = row + dr - r;
                    let nc = col + dc - r;
                    if resp[nr * width + nc] >= val {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                let angle = iy[row * width + col].atan2(ix[row * width + col]);
                keypoints.push(Keypoint::new(col as f64, row as f64, val).with_angle(angle));
            }
        }
    }
    keypoints
}

// ── FAST Corner Detector ───────────────────────────────────────

/// Configuration for FAST keypoint detection.
#[derive(Debug, Clone)]
pub struct FastConfig {
    pub threshold: f64,
    pub min_contiguous: usize,
    pub nms: bool,
}

impl FastConfig {
    pub fn new() -> Self {
        Self { threshold: 20.0, min_contiguous: 9, nms: true }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_min_contiguous(mut self, n: usize) -> Self {
        self.min_contiguous = n;
        self
    }

    pub fn with_nms(mut self, nms: bool) -> Self {
        self.nms = nms;
        self
    }
}

impl fmt::Display for FastConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FastConfig(thr={}, n={}, nms={})", self.threshold, self.min_contiguous, self.nms)
    }
}

/// Bresenham circle of radius 3 (16 pixels) — FAST-9/12/16.
const CIRCLE_OFFSETS: [(i32, i32); 16] = [
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3),
];

/// Detect FAST keypoints.
pub fn detect_fast(
    pixels: &[f64], width: usize, height: usize, config: &FastConfig,
) -> Vec<Keypoint> {
    let w = width as i32;
    let h = height as i32;
    let mut scores = vec![0.0_f64; width * height];
    let mut candidates = Vec::new();

    for row in 3..h - 3 {
        for col in 3..w - 3 {
            let center = pixels[(row as usize) * width + col as usize];
            let t = config.threshold;

            // Collect brighter/darker flags on the circle
            let mut brighter = [false; 16];
            let mut darker = [false; 16];
            for (i, &(dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
                let px = pixels[((row + dy) as usize) * width + (col + dx) as usize];
                brighter[i] = px > center + t;
                darker[i] = px < center - t;
            }

            // Check for N contiguous brighter or darker
            let n = config.min_contiguous;
            let passes = |flags: &[bool; 16]| -> bool {
                for start in 0..16 {
                    let mut count = 0;
                    for k in 0..n {
                        if flags[(start + k) % 16] {
                            count += 1;
                        } else {
                            break;
                        }
                    }
                    if count >= n {
                        return true;
                    }
                }
                false
            };

            if passes(&brighter) || passes(&darker) {
                // Score = sum of absolute differences above threshold
                let score: f64 = CIRCLE_OFFSETS.iter().map(|&(dx, dy)| {
                    let px = pixels[((row + dy) as usize) * width + (col + dx) as usize];
                    let diff = (px - center).abs();
                    if diff > t { diff - t } else { 0.0 }
                }).sum();
                scores[(row as usize) * width + col as usize] = score;
                candidates.push((col as usize, row as usize, score));
            }
        }
    }

    if !config.nms {
        return candidates.iter()
            .map(|&(x, y, s)| Keypoint::new(x as f64, y as f64, s))
            .collect();
    }

    // Non-maximum suppression on 3x3 neighborhood
    let mut keypoints = Vec::new();
    for &(cx, cy, score) in &candidates {
        let mut is_max = true;
        'outer: for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = (cx as i32 + dx) as usize;
                let ny = (cy as i32 + dy) as usize;
                if scores[ny * width + nx] >= score {
                    is_max = false;
                    break 'outer;
                }
            }
        }
        if is_max {
            keypoints.push(Keypoint::new(cx as f64, cy as f64, score));
        }
    }
    keypoints
}

// ── ORB Descriptor Extraction ──────────────────────────────────

/// Deterministic bit-pair sampling offsets for a 31x31 patch.
/// 256 pairs = 256 bits = 32 bytes.
fn orb_sample_pairs() -> Vec<(i32, i32, i32, i32)> {
    let mut pairs = Vec::with_capacity(256);
    // Quasi-random sampling using a simple LCG
    let mut state: u64 = 0xDEAD_BEEF;
    for _ in 0..256 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let a = ((state >> 16) as i32 % 31) - 15;
        let b = ((state >> 32) as i32 % 31) - 15;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let c = ((state >> 16) as i32 % 31) - 15;
        let d = ((state >> 32) as i32 % 31) - 15;
        pairs.push((a, b, c, d));
    }
    pairs
}

/// Extract ORB descriptors for given keypoints.
pub fn extract_orb(
    pixels: &[f64], width: usize, height: usize, keypoints: &[Keypoint],
) -> Vec<Descriptor> {
    let pairs = orb_sample_pairs();
    let mut descriptors = Vec::new();
    let margin = 16;

    for kp in keypoints {
        let cx = kp.x as i32;
        let cy = kp.y as i32;
        if cx < margin || cy < margin
            || cx >= (width as i32 - margin)
            || cy >= (height as i32 - margin)
        {
            continue;
        }

        let cos_a = kp.angle.cos();
        let sin_a = kp.angle.sin();

        let mut bits = [0u8; 32];
        for (i, &(ax, ay, bx, by)) in pairs.iter().enumerate() {
            // Rotate sample points by keypoint orientation
            let rax = (ax as f64 * cos_a - ay as f64 * sin_a).round() as i32;
            let ray = (ax as f64 * sin_a + ay as f64 * cos_a).round() as i32;
            let rbx = (bx as f64 * cos_a - by as f64 * sin_a).round() as i32;
            let rby = (bx as f64 * sin_a + by as f64 * cos_a).round() as i32;

            let pa = pixels[((cy + ray) as usize) * width + (cx + rax) as usize];
            let pb = pixels[((cy + rby) as usize) * width + (cx + rbx) as usize];

            if pa < pb {
                bits[i / 8] |= 1 << (i % 8);
            }
        }

        descriptors.push(Descriptor::new(*kp, bits));
    }
    descriptors
}

// ── Brute-Force Matcher ────────────────────────────────────────

/// Brute-force match descriptors using Hamming distance.
/// Returns matches sorted by distance.
pub fn match_descriptors(query: &[Descriptor], train: &[Descriptor]) -> Vec<Match> {
    let mut matches = Vec::new();
    for (qi, qd) in query.iter().enumerate() {
        let mut best_dist = u32::MAX;
        let mut best_idx = 0;
        for (ti, td) in train.iter().enumerate() {
            let d = qd.hamming_distance(td);
            if d < best_dist {
                best_dist = d;
                best_idx = ti;
            }
        }
        if best_dist < u32::MAX {
            matches.push(Match::new(qi, best_idx, best_dist));
        }
    }
    matches.sort_by_key(|m| m.distance);
    matches
}

/// Ratio test: keep matches where best/second-best distance < ratio.
pub fn ratio_test(query: &[Descriptor], train: &[Descriptor], ratio: f64) -> Vec<Match> {
    let mut matches = Vec::new();
    for (qi, qd) in query.iter().enumerate() {
        let mut best = u32::MAX;
        let mut second = u32::MAX;
        let mut best_idx = 0;

        for (ti, td) in train.iter().enumerate() {
            let d = qd.hamming_distance(td);
            if d < best {
                second = best;
                best = d;
                best_idx = ti;
            } else if d < second {
                second = d;
            }
        }

        if second > 0 && (best as f64) < ratio * (second as f64) {
            matches.push(Match::new(qi, best_idx, best));
        }
    }
    matches
}

// ── Non-Maximum Suppression ────────────────────────────────────

/// Suppress keypoints that are within `radius` of a stronger keypoint.
pub fn nms_keypoints(keypoints: &mut Vec<Keypoint>, radius: f64) {
    keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    let r2 = radius * radius;

    for kp in keypoints.iter() {
        let dominated = keep.iter().any(|kept: &Keypoint| {
            let dx = kp.x - kept.x;
            let dy = kp.y - kept.y;
            dx * dx + dy * dy < r2
        });
        if !dominated {
            keep.push(*kp);
        }
    }
    *keypoints = keep;
}

/// Retain only the top-N keypoints by response.
pub fn retain_best(keypoints: &mut Vec<Keypoint>, n: usize) {
    keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    keypoints.truncate(n);
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(width: usize, height: usize) -> Vec<f64> {
        vec![128.0; width * height]
    }

    fn make_corner_image(width: usize, height: usize) -> Vec<f64> {
        let mut img = vec![0.0_f64; width * height];
        // Create a bright square in the middle (corner at edges)
        for r in height / 4..3 * height / 4 {
            for c in width / 4..3 * width / 4 {
                img[r * width + c] = 255.0;
            }
        }
        img
    }

    #[test]
    fn test_keypoint_basic() {
        let kp = Keypoint::new(10.0, 20.0, 100.0);
        assert_eq!(kp.x, 10.0);
        assert_eq!(kp.y, 20.0);
        assert_eq!(kp.response, 100.0);
        assert_eq!(kp.octave, 0);
    }

    #[test]
    fn test_keypoint_builder() {
        let kp = Keypoint::new(5.0, 5.0, 50.0).with_octave(2).with_angle(1.57);
        assert_eq!(kp.octave, 2);
        assert!((kp.angle - 1.57).abs() < 1e-9);
    }

    #[test]
    fn test_keypoint_distance() {
        let a = Keypoint::new(0.0, 0.0, 1.0);
        let b = Keypoint::new(3.0, 4.0, 1.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_keypoint_display() {
        let kp = Keypoint::new(10.5, 20.3, 0.1234);
        let s = format!("{}", kp);
        assert!(s.contains("10.5"));
        assert!(s.contains("20.3"));
    }

    #[test]
    fn test_descriptor_hamming_identical() {
        let kp = Keypoint::new(0.0, 0.0, 1.0);
        let d1 = Descriptor::new(kp, [0xFF; 32]);
        let d2 = Descriptor::new(kp, [0xFF; 32]);
        assert_eq!(d1.hamming_distance(&d2), 0);
    }

    #[test]
    fn test_descriptor_hamming_opposite() {
        let kp = Keypoint::new(0.0, 0.0, 1.0);
        let d1 = Descriptor::new(kp, [0x00; 32]);
        let d2 = Descriptor::new(kp, [0xFF; 32]);
        assert_eq!(d1.hamming_distance(&d2), 256);
    }

    #[test]
    fn test_descriptor_hamming_single_bit() {
        let kp = Keypoint::new(0.0, 0.0, 1.0);
        let mut bits = [0u8; 32];
        bits[0] = 1;
        let d1 = Descriptor::new(kp, [0u8; 32]);
        let d2 = Descriptor::new(kp, bits);
        assert_eq!(d1.hamming_distance(&d2), 1);
    }

    #[test]
    fn test_descriptor_display() {
        let kp = Keypoint::new(1.0, 2.0, 3.0);
        let d = Descriptor::new(kp, [0u8; 32]);
        let s = format!("{}", d);
        assert!(s.contains("Descriptor"));
    }

    #[test]
    fn test_match_display() {
        let m = Match::new(0, 5, 42);
        let s = format!("{}", m);
        assert!(s.contains("42"));
    }

    #[test]
    fn test_harris_config_builder() {
        let cfg = HarrisConfig::new().with_k(0.06).with_threshold(500.0);
        assert_eq!(cfg.k, 0.06);
        assert_eq!(cfg.threshold, 500.0);
    }

    #[test]
    fn test_sobel_uniform() {
        let img = make_image(20, 20);
        let (ix, iy) = sobel_gradients(&img, 20, 20);
        // Uniform image → zero gradients everywhere
        for val in ix.iter().chain(iy.iter()) {
            assert!(val.abs() < 1e-9);
        }
    }

    #[test]
    fn test_harris_uniform_no_corners() {
        let img = make_image(30, 30);
        let cfg = HarrisConfig::new().with_threshold(1.0);
        let kps = detect_harris(&img, 30, 30, &cfg);
        assert!(kps.is_empty(), "uniform image should have no corners");
    }

    #[test]
    fn test_harris_corner_image() {
        let img = make_corner_image(40, 40);
        let cfg = HarrisConfig::new().with_threshold(100.0).with_nms_radius(2);
        let kps = detect_harris(&img, 40, 40, &cfg);
        // Should find corners at edges of the bright square
        assert!(!kps.is_empty(), "should detect corners in synthetic image");
    }

    #[test]
    fn test_fast_config_builder() {
        let cfg = FastConfig::new().with_threshold(30.0).with_nms(false);
        assert_eq!(cfg.threshold, 30.0);
        assert!(!cfg.nms);
    }

    #[test]
    fn test_fast_uniform_no_keypoints() {
        let img = make_image(30, 30);
        let cfg = FastConfig::new();
        let kps = detect_fast(&img, 30, 30, &cfg);
        assert!(kps.is_empty());
    }

    #[test]
    fn test_fast_corner_image() {
        let img = make_corner_image(40, 40);
        let cfg = FastConfig::new().with_threshold(10.0);
        let kps = detect_fast(&img, 40, 40, &cfg);
        assert!(!kps.is_empty(), "should detect FAST keypoints at edges");
    }

    #[test]
    fn test_orb_sample_pairs_count() {
        let pairs = orb_sample_pairs();
        assert_eq!(pairs.len(), 256);
    }

    #[test]
    fn test_match_descriptors() {
        let kp = Keypoint::new(0.0, 0.0, 1.0);
        let d1 = Descriptor::new(kp, [0xAA; 32]);
        let d2 = Descriptor::new(kp, [0xAA; 32]);
        let d3 = Descriptor::new(kp, [0x55; 32]);

        let matches = match_descriptors(&[d1], &[d2, d3]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_idx, 0); // d2 is closer
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_nms_keypoints() {
        let mut kps = vec![
            Keypoint::new(10.0, 10.0, 100.0),
            Keypoint::new(11.0, 10.0, 90.0),
            Keypoint::new(50.0, 50.0, 80.0),
        ];
        nms_keypoints(&mut kps, 5.0);
        assert_eq!(kps.len(), 2);
        assert_eq!(kps[0].response, 100.0);
        assert_eq!(kps[1].response, 80.0);
    }

    #[test]
    fn test_retain_best() {
        let mut kps = vec![
            Keypoint::new(0.0, 0.0, 10.0),
            Keypoint::new(1.0, 1.0, 50.0),
            Keypoint::new(2.0, 2.0, 30.0),
        ];
        retain_best(&mut kps, 2);
        assert_eq!(kps.len(), 2);
        assert_eq!(kps[0].response, 50.0);
        assert_eq!(kps[1].response, 30.0);
    }

    #[test]
    fn test_ratio_test_filter() {
        let kp = Keypoint::new(0.0, 0.0, 1.0);
        let q = Descriptor::new(kp, [0x00; 32]);
        let mut close_bits = [0u8; 32];
        close_bits[0] = 0x01; // 1 bit different
        let t1 = Descriptor::new(kp, close_bits);
        let t2 = Descriptor::new(kp, [0xFF; 32]); // 256 bits different

        let matches = ratio_test(&[q], &[t1, t2], 0.8);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].distance, 1);
    }
}
