// fxaa.rs — Fast Approximate Anti-Aliasing (FXAA 3.11)
// Pure Rust, no external deps beyond std.

/// RGBA pixel represented as 4 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Convert to luminance using perceptual weights (BT.709).
    pub fn luminance(&self) -> f32 {
        let rf = self.r as f32 / 255.0;
        let gf = self.g as f32 / 255.0;
        let bf = self.b as f32 / 255.0;
        0.2126 * rf + 0.7152 * gf + 0.0722 * bf
    }

    /// Linearly interpolate between two colors.
    pub fn lerp(a: &Rgba, b: &Rgba, t: f32) -> Rgba {
        let t = t.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        Rgba {
            r: (a.r as f32 * inv + b.r as f32 * t).round() as u8,
            g: (a.g as f32 * inv + b.g as f32 * t).round() as u8,
            b: (a.b as f32 * inv + b.b as f32 * t).round() as u8,
            a: (a.a as f32 * inv + b.a as f32 * t).round() as u8,
        }
    }
}

/// FXAA quality preset controlling edge detection sensitivity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FxaaConfig {
    /// Minimum contrast threshold for edge detection (0.0312 - 0.0833 typical).
    pub edge_threshold: f32,
    /// Relative threshold — edges below this fraction of local max luminance are ignored.
    pub edge_threshold_min: f32,
    /// Sub-pixel quality — higher values smooth more (0.0 off, 1.0 max).
    pub subpixel_quality: f32,
    /// Maximum search steps along edge direction.
    pub max_search_steps: usize,
}

impl Default for FxaaConfig {
    fn default() -> Self {
        Self {
            edge_threshold: 0.0625,
            edge_threshold_min: 0.0312,
            subpixel_quality: 0.75,
            max_search_steps: 12,
        }
    }
}

impl FxaaConfig {
    /// High quality preset.
    pub fn high_quality() -> Self {
        Self {
            edge_threshold: 0.0312,
            edge_threshold_min: 0.0156,
            subpixel_quality: 1.0,
            max_search_steps: 16,
        }
    }

    /// Low quality / performance preset.
    pub fn low_quality() -> Self {
        Self {
            edge_threshold: 0.0833,
            edge_threshold_min: 0.0625,
            subpixel_quality: 0.5,
            max_search_steps: 6,
        }
    }
}

/// Image buffer for FXAA processing.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBuffer {
    pub pixels: Vec<Rgba>,
    pub width: usize,
    pub height: usize,
}

impl ImageBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![Rgba::new(0, 0, 0, 255); width * height],
            width,
            height,
        }
    }

    pub fn from_pixels(pixels: Vec<Rgba>, width: usize, height: usize) -> Option<Self> {
        if pixels.len() != width * height {
            return None;
        }
        Some(Self { pixels, width, height })
    }

    fn pixel_at(&self, x: usize, y: usize) -> Rgba {
        self.pixels[y * self.width + x]
    }

    fn pixel_at_clamped(&self, x: isize, y: isize) -> Rgba {
        let cx = x.clamp(0, self.width as isize - 1) as usize;
        let cy = y.clamp(0, self.height as isize - 1) as usize;
        self.pixels[cy * self.width + cx]
    }

    fn luma_at(&self, x: isize, y: isize) -> f32 {
        self.pixel_at_clamped(x, y).luminance()
    }
}

/// Edge detection result for a single pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeInfo {
    /// Whether this pixel is on an edge.
    pub is_edge: bool,
    /// True if the edge is horizontal, false if vertical.
    pub is_horizontal: bool,
    /// Local contrast (max-min of neighborhood).
    pub contrast: f32,
    /// Pixel luminance.
    pub luminance: f32,
}

/// Compute luminance buffer for the entire image.
pub fn compute_luminance_buffer(image: &ImageBuffer) -> Vec<f32> {
    image.pixels.iter().map(|p| p.luminance()).collect()
}

/// Detect edge at a specific pixel location.
pub fn detect_edge(image: &ImageBuffer, x: usize, y: usize, config: &FxaaConfig) -> EdgeInfo {
    let ix = x as isize;
    let iy = y as isize;

    let luma_n = image.luma_at(ix, iy - 1);
    let luma_s = image.luma_at(ix, iy + 1);
    let luma_e = image.luma_at(ix + 1, iy);
    let luma_w = image.luma_at(ix - 1, iy);
    let luma_m = image.luma_at(ix, iy);

    let luma_min = luma_n.min(luma_s).min(luma_e).min(luma_w).min(luma_m);
    let luma_max = luma_n.max(luma_s).max(luma_e).max(luma_w).max(luma_m);
    let contrast = luma_max - luma_min;

    let threshold = config.edge_threshold.max(luma_max * config.edge_threshold_min);
    if contrast < threshold {
        return EdgeInfo {
            is_edge: false,
            is_horizontal: false,
            contrast,
            luminance: luma_m,
        };
    }

    // Determine edge orientation using diagonal neighbors.
    let luma_nw = image.luma_at(ix - 1, iy - 1);
    let luma_ne = image.luma_at(ix + 1, iy - 1);
    let luma_sw = image.luma_at(ix - 1, iy + 1);
    let luma_se = image.luma_at(ix + 1, iy + 1);

    let horizontal = (luma_n + luma_s - 2.0 * luma_m).abs() * 2.0
        + (luma_nw + luma_sw - 2.0 * luma_w).abs()
        + (luma_ne + luma_se - 2.0 * luma_e).abs();

    let vertical = (luma_e + luma_w - 2.0 * luma_m).abs() * 2.0
        + (luma_ne + luma_nw - 2.0 * luma_n).abs()
        + (luma_se + luma_sw - 2.0 * luma_s).abs();

    let is_horizontal = horizontal >= vertical;

    EdgeInfo {
        is_edge: true,
        is_horizontal,
        contrast,
        luminance: luma_m,
    }
}

/// Compute sub-pixel blend factor based on neighborhood.
fn subpixel_blend(image: &ImageBuffer, x: usize, y: usize, config: &FxaaConfig) -> f32 {
    let ix = x as isize;
    let iy = y as isize;

    let luma_m = image.luma_at(ix, iy);
    let luma_n = image.luma_at(ix, iy - 1);
    let luma_s = image.luma_at(ix, iy + 1);
    let luma_e = image.luma_at(ix + 1, iy);
    let luma_w = image.luma_at(ix - 1, iy);
    let luma_nw = image.luma_at(ix - 1, iy - 1);
    let luma_ne = image.luma_at(ix + 1, iy - 1);
    let luma_sw = image.luma_at(ix - 1, iy + 1);
    let luma_se = image.luma_at(ix + 1, iy + 1);

    // Low-pass filter of the 3x3 neighborhood.
    let avg = (2.0 * (luma_n + luma_s + luma_e + luma_w)
        + luma_nw + luma_ne + luma_sw + luma_se)
        / 12.0;

    let range = {
        let min_l = luma_n.min(luma_s).min(luma_e).min(luma_w).min(luma_m);
        let max_l = luma_n.max(luma_s).max(luma_e).max(luma_w).max(luma_m);
        max_l - min_l
    };

    let sub = ((avg - luma_m).abs() / range.max(1e-10)).clamp(0.0, 1.0);
    let sub_smooth = sub * sub * (3.0 - 2.0 * sub); // smoothstep
    sub_smooth * sub_smooth * config.subpixel_quality
}

/// Search along the edge in one direction and return the distance (in pixels) and end luminance.
fn edge_search(
    image: &ImageBuffer,
    start_x: isize,
    start_y: isize,
    step_x: isize,
    step_y: isize,
    luma_center: f32,
    luma_edge_avg: f32,
    max_steps: usize,
) -> (f32, bool) {
    let gradient_threshold = 0.25;

    for i in 1..=max_steps {
        let px = start_x + step_x * i as isize;
        let py = start_y + step_y * i as isize;
        let luma = image.luma_at(px, py);
        let diff = (luma - luma_edge_avg).abs();
        if diff >= gradient_threshold * (luma_center - luma_edge_avg).abs().max(0.01) {
            return (i as f32, luma < luma_edge_avg);
        }
    }
    (max_steps as f32, false)
}

/// Apply FXAA 3.11 to a single pixel — returns the anti-aliased color.
pub fn fxaa_pixel(image: &ImageBuffer, x: usize, y: usize, config: &FxaaConfig) -> Rgba {
    let edge = detect_edge(image, x, y, config);
    if !edge.is_edge {
        return image.pixel_at(x, y);
    }

    let ix = x as isize;
    let iy = y as isize;

    // Perpendicular direction to the edge (direction to blend toward).
    let (perp_x, perp_y): (isize, isize) = if edge.is_horizontal {
        (0, 1)
    } else {
        (1, 0)
    };

    let luma_pos = image.luma_at(ix + perp_x, iy + perp_y);
    let luma_neg = image.luma_at(ix - perp_x, iy - perp_y);

    let gradient_pos = (luma_pos - edge.luminance).abs();
    let gradient_neg = (luma_neg - edge.luminance).abs();

    // Determine which side of the edge has the steeper gradient.
    let step_sign: isize = if gradient_pos >= gradient_neg { 1 } else { -1 };

    let luma_edge_avg = if gradient_pos >= gradient_neg {
        (edge.luminance + luma_pos) / 2.0
    } else {
        (edge.luminance + luma_neg) / 2.0
    };

    // Search along the edge (tangent direction).
    let (search_step_x, search_step_y) = if edge.is_horizontal {
        (1isize, 0isize)
    } else {
        (0isize, 1isize)
    };

    let (dist_pos, _at_end_pos) = edge_search(
        image, ix, iy, search_step_x, search_step_y,
        edge.luminance, luma_edge_avg, config.max_search_steps,
    );
    let (dist_neg, _at_end_neg) = edge_search(
        image, ix, iy, -search_step_x, -search_step_y,
        edge.luminance, luma_edge_avg, config.max_search_steps,
    );

    // Compute pixel offset along edge perpendicular.
    let total_dist = dist_pos + dist_neg;
    let pixel_offset = if total_dist > 0.0 {
        let closer = dist_pos.min(dist_neg);
        0.5 - closer / total_dist
    } else {
        0.0
    };

    // Compute sub-pixel anti-aliasing blend.
    let sub_blend = subpixel_blend(image, x, y, config);
    let blend_factor = pixel_offset.max(sub_blend).clamp(0.0, 0.5);

    if blend_factor < 1e-6 {
        return image.pixel_at(x, y);
    }

    // Blend the center pixel toward the neighbor on the steeper-gradient side.
    let neighbor = image.pixel_at_clamped(
        ix + perp_x * step_sign,
        iy + perp_y * step_sign,
    );
    let original = image.pixel_at(x, y);
    Rgba::lerp(&original, &neighbor, blend_factor)
}

/// Apply FXAA to an entire image buffer.
pub fn apply_fxaa(image: &ImageBuffer, config: &FxaaConfig) -> ImageBuffer {
    let mut result = ImageBuffer::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            result.pixels[y * image.width + x] = fxaa_pixel(image, x, y, config);
        }
    }
    result
}

/// Comparison result between original and FXAA-processed image.
#[derive(Debug, Clone, PartialEq)]
pub struct FxaaComparison {
    /// Number of pixels that changed.
    pub pixels_changed: usize,
    /// Total pixels in the image.
    pub total_pixels: usize,
    /// Average luminance change across changed pixels.
    pub avg_luma_change: f32,
    /// Maximum luminance change.
    pub max_luma_change: f32,
    /// Fraction of pixels detected as edges.
    pub edge_fraction: f32,
}

/// Compare original and FXAA-processed images.
pub fn compare_before_after(
    original: &ImageBuffer,
    processed: &ImageBuffer,
    config: &FxaaConfig,
) -> FxaaComparison {
    let total = original.width * original.height;
    let mut changed = 0usize;
    let mut total_luma_diff = 0.0f32;
    let mut max_luma_diff = 0.0f32;
    let mut edge_count = 0usize;

    for y in 0..original.height {
        for x in 0..original.width {
            let idx = y * original.width + x;
            let o = original.pixels[idx];
            let p = processed.pixels[idx];
            if o != p {
                changed += 1;
                let diff = (o.luminance() - p.luminance()).abs();
                total_luma_diff += diff;
                if diff > max_luma_diff {
                    max_luma_diff = diff;
                }
            }
            let edge = detect_edge(original, x, y, config);
            if edge.is_edge {
                edge_count += 1;
            }
        }
    }

    FxaaComparison {
        pixels_changed: changed,
        total_pixels: total,
        avg_luma_change: if changed > 0 { total_luma_diff / changed as f32 } else { 0.0 },
        max_luma_change: max_luma_diff,
        edge_fraction: edge_count as f32 / total as f32,
    }
}

/// Generate an edge detection visualization (white = edge, black = non-edge).
pub fn visualize_edges(image: &ImageBuffer, config: &FxaaConfig) -> ImageBuffer {
    let mut result = ImageBuffer::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let edge = detect_edge(image, x, y, config);
            let val = if edge.is_edge { 255 } else { 0 };
            result.pixels[y * image.width + x] = Rgba::new(val, val, val, 255);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: usize, h: usize, color: Rgba) -> ImageBuffer {
        ImageBuffer {
            pixels: vec![color; w * h],
            width: w,
            height: h,
        }
    }

    fn checkerboard(w: usize, h: usize) -> ImageBuffer {
        let mut img = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = if (x + y) % 2 == 0 { 255 } else { 0 };
                img.pixels[y * w + x] = Rgba::new(c, c, c, 255);
            }
        }
        img
    }

    fn edge_image(w: usize, h: usize) -> ImageBuffer {
        let mut img = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = if x < w / 2 { 0 } else { 255 };
                img.pixels[y * w + x] = Rgba::new(c, c, c, 255);
            }
        }
        img
    }

    #[test]
    fn test_rgba_luminance() {
        let white = Rgba::new(255, 255, 255, 255);
        assert!((white.luminance() - 1.0).abs() < 1e-3);

        let black = Rgba::new(0, 0, 0, 255);
        assert!(black.luminance().abs() < 1e-6);

        let red = Rgba::new(255, 0, 0, 255);
        assert!((red.luminance() - 0.2126).abs() < 1e-3);
    }

    #[test]
    fn test_rgba_lerp() {
        let a = Rgba::new(0, 0, 0, 255);
        let b = Rgba::new(255, 255, 255, 255);
        let mid = Rgba::lerp(&a, &b, 0.5);
        assert!((mid.r as i16 - 128).abs() <= 1);
        assert!((mid.g as i16 - 128).abs() <= 1);

        let same = Rgba::lerp(&a, &b, 0.0);
        assert_eq!(same, a);

        let end = Rgba::lerp(&a, &b, 1.0);
        assert_eq!(end, b);
    }

    #[test]
    fn test_lerp_clamp() {
        let a = Rgba::new(100, 100, 100, 255);
        let b = Rgba::new(200, 200, 200, 255);
        let clamped_neg = Rgba::lerp(&a, &b, -1.0);
        assert_eq!(clamped_neg, a);
        let clamped_over = Rgba::lerp(&a, &b, 2.0);
        assert_eq!(clamped_over, b);
    }

    #[test]
    fn test_image_buffer_new() {
        let img = ImageBuffer::new(10, 10);
        assert_eq!(img.pixels.len(), 100);
        assert_eq!(img.width, 10);
        assert_eq!(img.height, 10);
    }

    #[test]
    fn test_image_buffer_from_pixels() {
        let pixels = vec![Rgba::new(0, 0, 0, 255); 6];
        let img = ImageBuffer::from_pixels(pixels, 3, 2);
        assert!(img.is_some());

        let bad = ImageBuffer::from_pixels(vec![Rgba::new(0, 0, 0, 255); 5], 3, 2);
        assert!(bad.is_none());
    }

    #[test]
    fn test_config_default() {
        let cfg = FxaaConfig::default();
        assert!((cfg.edge_threshold - 0.0625).abs() < 1e-6);
        assert!((cfg.subpixel_quality - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_config_presets() {
        let high = FxaaConfig::high_quality();
        let low = FxaaConfig::low_quality();
        assert!(high.edge_threshold < low.edge_threshold);
        assert!(high.subpixel_quality > low.subpixel_quality);
        assert!(high.max_search_steps > low.max_search_steps);
    }

    #[test]
    fn test_compute_luminance_buffer() {
        let img = solid_image(2, 2, Rgba::new(255, 255, 255, 255));
        let luma = compute_luminance_buffer(&img);
        assert_eq!(luma.len(), 4);
        for l in &luma {
            assert!((*l - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_no_edge_in_solid() {
        let img = solid_image(8, 8, Rgba::new(128, 128, 128, 255));
        let config = FxaaConfig::default();
        let edge = detect_edge(&img, 4, 4, &config);
        assert!(!edge.is_edge);
        assert!(edge.contrast < 1e-6);
    }

    #[test]
    fn test_edge_detected_at_boundary() {
        let img = edge_image(16, 16);
        let config = FxaaConfig::default();
        let edge = detect_edge(&img, 8, 8, &config);
        assert!(edge.is_edge);
        assert!(edge.contrast > 0.5);
    }

    #[test]
    fn test_edge_orientation_vertical_boundary() {
        let img = edge_image(16, 16);
        let config = FxaaConfig::default();
        let edge = detect_edge(&img, 8, 8, &config);
        assert!(edge.is_edge);
        // Vertical boundary -> the edge is detected as vertical (not horizontal).
        assert!(!edge.is_horizontal);
    }

    #[test]
    fn test_edge_orientation_horizontal_boundary() {
        let mut img = ImageBuffer::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                let c = if y < 8 { 0 } else { 255 };
                img.pixels[y * 16 + x] = Rgba::new(c, c, c, 255);
            }
        }
        let config = FxaaConfig::default();
        let edge = detect_edge(&img, 8, 8, &config);
        assert!(edge.is_edge);
        assert!(edge.is_horizontal);
    }

    #[test]
    fn test_fxaa_solid_unchanged() {
        let img = solid_image(8, 8, Rgba::new(100, 100, 100, 255));
        let config = FxaaConfig::default();
        let result = apply_fxaa(&img, &config);
        assert_eq!(result.pixels, img.pixels);
    }

    #[test]
    fn test_fxaa_modifies_edges() {
        // Use a sharp vertical edge (left half black, right half white) which
        // produces a clear edge detectable by FXAA.
        let img = edge_image(16, 16);
        let config = FxaaConfig::high_quality();
        let result = apply_fxaa(&img, &config);
        let mut changed = 0;
        for i in 0..img.pixels.len() {
            if img.pixels[i] != result.pixels[i] {
                changed += 1;
            }
        }
        // FXAA should modify at least some pixels along the vertical edge.
        assert!(changed > 0, "FXAA should modify edge pixels");
    }

    #[test]
    fn test_fxaa_preserves_dimensions() {
        let img = edge_image(20, 10);
        let config = FxaaConfig::default();
        let result = apply_fxaa(&img, &config);
        assert_eq!(result.width, 20);
        assert_eq!(result.height, 10);
        assert_eq!(result.pixels.len(), 200);
    }

    #[test]
    fn test_comparison_solid() {
        let img = solid_image(8, 8, Rgba::new(100, 100, 100, 255));
        let config = FxaaConfig::default();
        let result = apply_fxaa(&img, &config);
        let cmp = compare_before_after(&img, &result, &config);
        assert_eq!(cmp.pixels_changed, 0);
        assert_eq!(cmp.total_pixels, 64);
        assert!(cmp.avg_luma_change.abs() < 1e-6);
    }

    #[test]
    fn test_comparison_edge_image() {
        let img = edge_image(16, 16);
        let config = FxaaConfig::default();
        let result = apply_fxaa(&img, &config);
        let cmp = compare_before_after(&img, &result, &config);
        assert_eq!(cmp.total_pixels, 256);
        assert!(cmp.edge_fraction > 0.0);
    }

    #[test]
    fn test_visualize_edges_solid() {
        let img = solid_image(8, 8, Rgba::new(128, 128, 128, 255));
        let config = FxaaConfig::default();
        let vis = visualize_edges(&img, &config);
        for p in &vis.pixels {
            assert_eq!(p.r, 0);
        }
    }

    #[test]
    fn test_visualize_edges_boundary() {
        let img = edge_image(16, 16);
        let config = FxaaConfig::default();
        let vis = visualize_edges(&img, &config);
        let white_count = vis.pixels.iter().filter(|p| p.r == 255).count();
        assert!(white_count > 0, "should detect at least some edge pixels");
    }

    #[test]
    fn test_pixel_at_clamped() {
        let img = solid_image(4, 4, Rgba::new(42, 42, 42, 255));
        assert_eq!(img.pixel_at_clamped(-1, -1), Rgba::new(42, 42, 42, 255));
        assert_eq!(img.pixel_at_clamped(100, 100), Rgba::new(42, 42, 42, 255));
    }

    #[test]
    fn test_subpixel_blend_solid() {
        let img = solid_image(8, 8, Rgba::new(128, 128, 128, 255));
        let config = FxaaConfig::default();
        let blend = subpixel_blend(&img, 4, 4, &config);
        assert!(blend.abs() < 1e-3, "solid area should have ~0 sub-pixel blend");
    }

    #[test]
    fn test_fxaa_pixel_interior() {
        let img = solid_image(8, 8, Rgba::new(50, 50, 50, 255));
        let config = FxaaConfig::default();
        let result = fxaa_pixel(&img, 4, 4, &config);
        assert_eq!(result, Rgba::new(50, 50, 50, 255));
    }

    #[test]
    fn test_high_quality_more_changes() {
        let img = edge_image(16, 16);
        let low = FxaaConfig::low_quality();
        let high = FxaaConfig::high_quality();

        let result_low = apply_fxaa(&img, &low);
        let result_high = apply_fxaa(&img, &high);

        let changed_low = img.pixels.iter().zip(result_low.pixels.iter())
            .filter(|(a, b)| a != b).count();
        let changed_high = img.pixels.iter().zip(result_high.pixels.iter())
            .filter(|(a, b)| a != b).count();

        // High quality should detect at least as many edges.
        assert!(changed_high >= changed_low);
    }
}
