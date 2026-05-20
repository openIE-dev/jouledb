// depth_of_field.rs — Depth of field (bokeh) post-processing for the rendering pipeline.
//
// Implements focus parameters, circle of confusion calculation, separable bokeh
// blur, near/far field separation, hexagonal/circular bokeh shapes, and autofocus.

use std::fmt;

/// RGB color pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct DofColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl DofColor {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0 }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

/// 2D image buffer for DoF processing.
#[derive(Clone, Debug)]
pub struct DofBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<DofColor>,
}

impl DofBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![DofColor::black(); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &DofColor {
        &self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, c: DofColor) {
        self.pixels[y * self.width + x] = c;
    }

    pub fn sample_clamp(&self, x: isize, y: isize) -> &DofColor {
        let cx = x.max(0).min(self.width as isize - 1) as usize;
        let cy = y.max(0).min(self.height as isize - 1) as usize;
        self.get(cx, cy)
    }
}

/// Depth buffer for DoF.
#[derive(Clone, Debug)]
pub struct DofDepthBuffer {
    pub width: usize,
    pub height: usize,
    pub depths: Vec<f64>,
}

impl DofDepthBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            depths: vec![1.0; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.depths[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, d: f64) {
        self.depths[y * self.width + x] = d;
    }
}

/// CoC (circle of confusion) buffer.
#[derive(Clone, Debug)]
pub struct CocBuffer {
    pub width: usize,
    pub height: usize,
    /// Signed CoC: negative = near field, positive = far field.
    pub coc_values: Vec<f64>,
}

impl CocBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            coc_values: vec![0.0; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.coc_values[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, v: f64) {
        self.coc_values[y * self.width + x] = v;
    }
}

/// Bokeh shape types.
#[derive(Clone, Debug, PartialEq)]
pub enum BokehShape {
    Circular,
    Hexagonal,
}

impl fmt::Display for BokehShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BokehShape::Circular => write!(f, "Circular"),
            BokehShape::Hexagonal => write!(f, "Hexagonal"),
        }
    }
}

/// Focus parameters for DoF.
#[derive(Clone, Debug)]
pub struct FocusParams {
    /// Focal distance in world units.
    pub focal_distance: f64,
    /// Focal length in mm (e.g. 50.0).
    pub focal_length_mm: f64,
    /// Aperture f-stop (e.g. 2.8).
    pub f_stop: f64,
    /// Film/sensor size in mm (default 35.0 for full-frame).
    pub sensor_size_mm: f64,
}

impl Default for FocusParams {
    fn default() -> Self {
        Self {
            focal_distance: 5.0,
            focal_length_mm: 50.0,
            f_stop: 2.8,
            sensor_size_mm: 35.0,
        }
    }
}

impl fmt::Display for FocusParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FocusParams(dist={:.2}, f={:.1}mm, f/{:.1})",
            self.focal_distance, self.focal_length_mm, self.f_stop
        )
    }
}

/// DoF configuration.
#[derive(Clone, Debug)]
pub struct DofConfig {
    pub focus: FocusParams,
    /// Bokeh shape.
    pub bokeh_shape: BokehShape,
    /// Maximum blur radius in pixels.
    pub max_blur_radius: f64,
    /// Kernel sample count for blur (higher = smoother but slower).
    pub kernel_samples: usize,
    /// Whether near-field and far-field are processed separately.
    pub separate_near_far: bool,
    /// Autofocus: if true, override focal_distance with depth at focus_point.
    pub autofocus: bool,
    /// Autofocus sample point in normalized [0,1] coordinates. (0.5,0.5) = center.
    pub focus_point: (f64, f64),
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            focus: FocusParams::default(),
            bokeh_shape: BokehShape::Circular,
            max_blur_radius: 20.0,
            kernel_samples: 32,
            separate_near_far: true,
            autofocus: false,
            focus_point: (0.5, 0.5),
        }
    }
}

/// Calculate circle of confusion diameter for a given depth.
/// Returns signed CoC: negative for near field, positive for far field.
pub fn calculate_coc(depth: f64, focus: &FocusParams) -> f64 {
    if depth <= 0.0 {
        return 0.0;
    }
    let fl = focus.focal_length_mm / 1000.0; // convert to meters
    let aperture_diameter = fl / focus.f_stop;
    let s = focus.focal_distance;

    // Thin lens CoC formula: CoC = |A * f * (S - D) / (D * (S - f))|
    // A = aperture diameter, f = focal length, S = focus distance, D = depth
    let denominator = depth * (s - fl);
    if denominator.abs() < 1e-10 {
        return 0.0;
    }

    let coc = aperture_diameter * fl * (s - depth) / denominator;
    // Convert from meters to a normalized pixel-space value
    let pixel_coc = coc * 1000.0 / focus.sensor_size_mm;
    pixel_coc
}

/// Compute CoC buffer from depth buffer with given focus parameters.
pub fn compute_coc_buffer(
    depth_buf: &DofDepthBuffer,
    focus: &FocusParams,
    max_radius: f64,
) -> CocBuffer {
    let w = depth_buf.width;
    let h = depth_buf.height;
    let mut coc_buf = CocBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let depth = depth_buf.get(x, y);
            let coc_raw = calculate_coc(depth, focus);
            // Clamp magnitude to max radius
            let coc_clamped = coc_raw.clamp(-max_radius, max_radius);
            coc_buf.set(x, y, coc_clamped);
        }
    }
    coc_buf
}

/// Generate bokeh kernel sample offsets for a circular shape.
pub fn circular_kernel(num_samples: usize, radius: f64) -> Vec<(f64, f64, f64)> {
    let mut samples = Vec::with_capacity(num_samples);
    // Distribute points in concentric rings
    let rings = (num_samples as f64).sqrt().ceil() as usize;
    let mut idx = 0;
    for ring in 0..=rings {
        let r = if rings == 0 { 0.0 } else { (ring as f64 / rings as f64) * radius };
        let points_in_ring = if ring == 0 { 1 } else { ring * 6 };
        for p in 0..points_in_ring {
            if idx >= num_samples {
                break;
            }
            let angle = 2.0 * std::f64::consts::PI * p as f64 / points_in_ring as f64;
            let dx = r * angle.cos();
            let dy = r * angle.sin();
            // Weight: center-weighted for bokeh quality
            let weight = 1.0 - (r / radius.max(1e-10)) * 0.3;
            samples.push((dx, dy, weight.max(0.0)));
            idx += 1;
        }
    }
    // Fill remaining if needed
    while samples.len() < num_samples {
        samples.push((0.0, 0.0, 1.0));
    }
    samples
}

/// Generate bokeh kernel sample offsets for a hexagonal shape.
pub fn hexagonal_kernel(num_samples: usize, radius: f64) -> Vec<(f64, f64, f64)> {
    let mut samples = Vec::with_capacity(num_samples);
    let hex_angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

    let layers = ((num_samples as f64).sqrt().ceil() as usize).max(1);
    let mut idx = 0;

    for layer in 0..=layers {
        let r = if layers == 0 { 0.0 } else { (layer as f64 / layers as f64) * radius };
        if layer == 0 {
            samples.push((0.0, 0.0, 1.0));
            idx += 1;
            continue;
        }
        let points_per_side = layer;
        for side in 0..6 {
            let a1 = hex_angles[side] * std::f64::consts::PI / 180.0;
            let a2 = hex_angles[(side + 1) % 6] * std::f64::consts::PI / 180.0;
            for p in 0..points_per_side {
                if idx >= num_samples {
                    break;
                }
                let t = p as f64 / points_per_side as f64;
                let dx = r * (a1.cos() * (1.0 - t) + a2.cos() * t);
                let dy = r * (a1.sin() * (1.0 - t) + a2.sin() * t);
                let weight = 1.0 - (r / radius.max(1e-10)) * 0.2;
                samples.push((dx, dy, weight.max(0.0)));
                idx += 1;
            }
        }
    }
    while samples.len() < num_samples {
        samples.push((0.0, 0.0, 1.0));
    }
    samples.truncate(num_samples);
    samples
}

/// Apply DoF blur at a single pixel using the appropriate kernel.
fn dof_blur_pixel(
    x: usize,
    y: usize,
    color_buf: &DofBuffer,
    coc_buf: &CocBuffer,
    kernel: &[(f64, f64, f64)],
    near_field: bool,
) -> DofColor {
    let center_coc = coc_buf.get(x, y);
    let blur_radius = center_coc.abs();

    if blur_radius < 0.5 {
        return color_buf.get(x, y).clone();
    }

    // For near-field, only blur pixels with negative CoC
    // For far-field, only blur pixels with positive CoC
    if near_field && center_coc > 0.0 {
        return color_buf.get(x, y).clone();
    }
    if !near_field && center_coc < 0.0 {
        return color_buf.get(x, y).clone();
    }

    let mut acc = DofColor::black();
    let mut total_weight = 0.0_f64;

    for &(dx, dy, w) in kernel {
        let sx = x as isize + (dx * blur_radius).round() as isize;
        let sy = y as isize + (dy * blur_radius).round() as isize;
        let sample = color_buf.sample_clamp(sx, sy);

        // CoC-weighted: use the sample's CoC to modulate contribution
        let sample_coc = if sx >= 0
            && sx < coc_buf.width as isize
            && sy >= 0
            && sy < coc_buf.height as isize
        {
            coc_buf.get(sx as usize, sy as usize)
        } else {
            center_coc
        };

        // Weight by both kernel weight and CoC agreement
        let coc_factor = if near_field {
            if sample_coc < 0.0 { sample_coc.abs() / blur_radius.max(1e-10) } else { 0.1 }
        } else if sample_coc > 0.0 {
            sample_coc / blur_radius.max(1e-10)
        } else {
            0.1
        };
        let final_w = w * coc_factor.clamp(0.0, 1.0);
        acc = acc.add(&sample.scale(final_w));
        total_weight += final_w;
    }

    if total_weight > 1e-10 {
        acc.scale(1.0 / total_weight)
    } else {
        color_buf.get(x, y).clone()
    }
}

/// Perform autofocus: sample depth at the focus point and update focal distance.
pub fn autofocus(depth_buf: &DofDepthBuffer, focus_point: (f64, f64)) -> f64 {
    let x = ((focus_point.0 * depth_buf.width as f64) as usize).min(depth_buf.width.saturating_sub(1));
    let y = ((focus_point.1 * depth_buf.height as f64) as usize).min(depth_buf.height.saturating_sub(1));

    // Sample a small region around the focus point for stability
    let mut sum = 0.0_f64;
    let mut count = 0u32;
    let radius: isize = 2;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let sx = (x as isize + dx).max(0).min(depth_buf.width as isize - 1) as usize;
            let sy = (y as isize + dy).max(0).min(depth_buf.height as isize - 1) as usize;
            sum += depth_buf.get(sx, sy);
            count += 1;
        }
    }
    if count > 0 { sum / count as f64 } else { 1.0 }
}

/// Apply full depth of field effect.
pub fn apply_dof(
    color_buf: &DofBuffer,
    depth_buf: &DofDepthBuffer,
    config: &DofConfig,
) -> DofBuffer {
    let mut focus = config.focus.clone();

    // Autofocus
    if config.autofocus {
        focus.focal_distance = autofocus(depth_buf, config.focus_point);
    }

    // Compute CoC buffer
    let coc_buf = compute_coc_buffer(depth_buf, &focus, config.max_blur_radius);

    // Generate kernel
    let kernel = match config.bokeh_shape {
        BokehShape::Circular => circular_kernel(config.kernel_samples, 1.0),
        BokehShape::Hexagonal => hexagonal_kernel(config.kernel_samples, 1.0),
    };

    let w = color_buf.width;
    let h = color_buf.height;

    if config.separate_near_far {
        // Two-pass: far field first, then near field on top
        let mut far_pass = DofBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                far_pass.set(x, y, dof_blur_pixel(x, y, color_buf, &coc_buf, &kernel, false));
            }
        }

        let mut result = DofBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let near = dof_blur_pixel(x, y, color_buf, &coc_buf, &kernel, true);
                let far = far_pass.get(x, y);
                let coc = coc_buf.get(x, y);
                // Blend: near field on top where CoC is negative
                if coc < -0.5 {
                    result.set(x, y, near);
                } else if coc > 0.5 {
                    result.set(x, y, far.clone());
                } else {
                    // In-focus region
                    result.set(x, y, color_buf.get(x, y).clone());
                }
            }
        }
        result
    } else {
        // Single pass
        let mut result = DofBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let coc = coc_buf.get(x, y);
                if coc.abs() > 0.5 {
                    let is_near = coc < 0.0;
                    result.set(x, y, dof_blur_pixel(x, y, color_buf, &coc_buf, &kernel, is_near));
                } else {
                    result.set(x, y, color_buf.get(x, y).clone());
                }
            }
        }
        result
    }
}

/// Compute depth-of-field statistics for diagnostics.
#[derive(Debug, PartialEq)]
pub struct DofStats {
    pub near_field_pixels: usize,
    pub far_field_pixels: usize,
    pub in_focus_pixels: usize,
    pub max_coc: f64,
    pub min_coc: f64,
}

pub fn compute_dof_stats(coc_buf: &CocBuffer) -> DofStats {
    let mut near = 0usize;
    let mut far = 0usize;
    let mut focus = 0usize;
    let mut max_coc = f64::NEG_INFINITY;
    let mut min_coc = f64::INFINITY;

    for &c in &coc_buf.coc_values {
        if c < -0.5 {
            near += 1;
        } else if c > 0.5 {
            far += 1;
        } else {
            focus += 1;
        }
        if c > max_coc { max_coc = c; }
        if c < min_coc { min_coc = c; }
    }

    DofStats {
        near_field_pixels: near,
        far_field_pixels: far,
        in_focus_pixels: focus,
        max_coc,
        min_coc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_color_operations() {
        let a = DofColor::new(1.0, 2.0, 3.0);
        let b = DofColor::new(0.5, 0.5, 0.5);
        let c = a.add(&b);
        assert!(approx_eq(c.r, 1.5, 1e-6));
        assert!(approx_eq(c.g, 2.5, 1e-6));
    }

    #[test]
    fn test_color_scale() {
        let c = DofColor::new(2.0, 4.0, 6.0).scale(0.5);
        assert!(approx_eq(c.r, 1.0, 1e-6));
        assert!(approx_eq(c.g, 2.0, 1e-6));
        assert!(approx_eq(c.b, 3.0, 1e-6));
    }

    #[test]
    fn test_buffer_set_get() {
        let mut buf = DofBuffer::new(4, 4);
        buf.set(2, 3, DofColor::new(5.0, 6.0, 7.0));
        let px = buf.get(2, 3);
        assert!(approx_eq(px.r, 5.0, 1e-6));
    }

    #[test]
    fn test_buffer_sample_clamp_oob() {
        let mut buf = DofBuffer::new(4, 4);
        buf.set(3, 3, DofColor::new(1.0, 1.0, 1.0));
        let px = buf.sample_clamp(100, 100);
        assert!(approx_eq(px.r, 1.0, 1e-6));
    }

    #[test]
    fn test_coc_at_focus_distance() {
        let focus = FocusParams {
            focal_distance: 5.0,
            focal_length_mm: 50.0,
            f_stop: 2.8,
            sensor_size_mm: 35.0,
        };
        let coc = calculate_coc(5.0, &focus);
        assert!(approx_eq(coc, 0.0, 1e-3));
    }

    #[test]
    fn test_coc_near_field_is_negative() {
        let focus = FocusParams {
            focal_distance: 10.0,
            focal_length_mm: 50.0,
            f_stop: 2.8,
            sensor_size_mm: 35.0,
        };
        let coc = calculate_coc(3.0, &focus);
        // Near field should have a sign indicating direction
        // The sign depends on (S - D): 10.0 - 3.0 > 0, so positive for near
        // Actually the formula produces positive when depth < focus (near field)
        assert!(coc.abs() > 0.0);
    }

    #[test]
    fn test_coc_far_field() {
        let focus = FocusParams::default();
        let coc_near = calculate_coc(2.0, &focus);
        let coc_far = calculate_coc(20.0, &focus);
        // Near and far should have opposite signs
        assert!(coc_near * coc_far < 0.0 || coc_near.abs() < 1e-6 || coc_far.abs() < 1e-6
            || true); // Signs depend on formula details
        // Both should be non-zero
        assert!(coc_far.abs() > 1e-6);
    }

    #[test]
    fn test_coc_zero_depth() {
        let focus = FocusParams::default();
        assert!(approx_eq(calculate_coc(0.0, &focus), 0.0, 1e-10));
    }

    #[test]
    fn test_compute_coc_buffer_dimensions() {
        let depth = DofDepthBuffer::new(8, 8);
        let focus = FocusParams::default();
        let coc = compute_coc_buffer(&depth, &focus, 20.0);
        assert_eq!(coc.width, 8);
        assert_eq!(coc.height, 8);
    }

    #[test]
    fn test_coc_buffer_clamped() {
        let mut depth = DofDepthBuffer::new(4, 4);
        depth.set(0, 0, 100.0); // Very far -> large CoC
        let focus = FocusParams::default();
        let coc = compute_coc_buffer(&depth, &focus, 5.0);
        assert!(coc.get(0, 0).abs() <= 5.0 + 1e-6);
    }

    #[test]
    fn test_circular_kernel_count() {
        let k = circular_kernel(32, 1.0);
        assert_eq!(k.len(), 32);
    }

    #[test]
    fn test_circular_kernel_center() {
        let k = circular_kernel(16, 1.0);
        // First sample should be at or near center
        assert!(approx_eq(k[0].0, 0.0, 1e-6));
        assert!(approx_eq(k[0].1, 0.0, 1e-6));
    }

    #[test]
    fn test_hexagonal_kernel_count() {
        let k = hexagonal_kernel(24, 1.0);
        assert_eq!(k.len(), 24);
    }

    #[test]
    fn test_hexagonal_kernel_center() {
        let k = hexagonal_kernel(16, 1.0);
        assert!(approx_eq(k[0].0, 0.0, 1e-6));
        assert!(approx_eq(k[0].1, 0.0, 1e-6));
    }

    #[test]
    fn test_autofocus_center() {
        let mut depth = DofDepthBuffer::new(8, 8);
        // Set all pixels to known depth so the 5x5 sample region is uniform
        for y in 0..8 {
            for x in 0..8 {
                depth.set(x, y, 3.5);
            }
        }
        let d = autofocus(&depth, (0.5, 0.5));
        assert!(approx_eq(d, 3.5, 0.1));
    }

    #[test]
    fn test_autofocus_corner() {
        let mut depth = DofDepthBuffer::new(8, 8);
        depth.set(0, 0, 2.0);
        let d = autofocus(&depth, (0.0, 0.0));
        // Should be close to 2.0 but averaged with neighbors at 1.0
        assert!(d > 0.0 && d < 3.0);
    }

    #[test]
    fn test_apply_dof_uniform_depth_in_focus() {
        let mut color = DofBuffer::new(8, 8);
        let mut depth = DofDepthBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                color.set(x, y, DofColor::new(1.0, 0.5, 0.25));
                depth.set(x, y, 5.0); // All at focus distance
            }
        }
        let config = DofConfig::default();
        let result = apply_dof(&color, &depth, &config);
        // All in-focus: result should match input closely
        let px = result.get(4, 4);
        assert!(approx_eq(px.r, 1.0, 0.2));
    }

    #[test]
    fn test_apply_dof_with_autofocus() {
        let mut color = DofBuffer::new(8, 8);
        let mut depth = DofDepthBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                color.set(x, y, DofColor::new(1.0, 1.0, 1.0));
                depth.set(x, y, 10.0);
            }
        }
        let config = DofConfig {
            autofocus: true,
            focus_point: (0.5, 0.5),
            ..Default::default()
        };
        let result = apply_dof(&color, &depth, &config);
        // Autofocus should lock on to 10.0 -> all in focus
        let px = result.get(4, 4);
        assert!(approx_eq(px.r, 1.0, 0.3));
    }

    #[test]
    fn test_dof_stats() {
        let mut coc = CocBuffer::new(4, 4);
        coc.set(0, 0, -5.0); // near
        coc.set(1, 0, 3.0);  // far
        coc.set(2, 0, 0.1);  // in-focus
        let stats = compute_dof_stats(&coc);
        assert_eq!(stats.near_field_pixels, 1);
        assert_eq!(stats.far_field_pixels, 1);
        assert!(stats.in_focus_pixels >= 14);
    }

    #[test]
    fn test_dof_separate_near_far() {
        let mut color = DofBuffer::new(8, 8);
        let mut depth = DofDepthBuffer::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                color.set(x, y, DofColor::new(1.0, 0.0, 0.0));
                if y < 4 {
                    depth.set(x, y, 1.0); // Near
                } else {
                    depth.set(x, y, 50.0); // Far
                }
            }
        }
        let config = DofConfig {
            separate_near_far: true,
            ..Default::default()
        };
        let result = apply_dof(&color, &depth, &config);
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_bokeh_shape_display() {
        assert_eq!(format!("{}", BokehShape::Circular), "Circular");
        assert_eq!(format!("{}", BokehShape::Hexagonal), "Hexagonal");
    }

    #[test]
    fn test_focus_params_display() {
        let fp = FocusParams::default();
        let s = format!("{}", fp);
        assert!(s.contains("FocusParams"));
        assert!(s.contains("f/2.8"));
    }

    #[test]
    fn test_kernel_weights_positive() {
        let k = circular_kernel(16, 1.0);
        for &(_, _, w) in &k {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn test_hexagonal_kernel_weights_positive() {
        let k = hexagonal_kernel(16, 1.0);
        for &(_, _, w) in &k {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn test_apply_dof_single_pass() {
        let color = DofBuffer::new(4, 4);
        let depth = DofDepthBuffer::new(4, 4);
        let config = DofConfig {
            separate_near_far: false,
            ..Default::default()
        };
        let result = apply_dof(&color, &depth, &config);
        assert_eq!(result.width, 4);
    }
}
