//! Pixel-perfect rendering helpers: position snapping, sub-pixel accumulators,
//! pixel-aligned camera, upscale filter modes, reference-resolution scaling,
//! and Bresenham line drawing.
//!
//! Essential for pixel-art games where fractional-pixel movement causes
//! shimmering and texture bleeding.

// ── Basic Types ────────────────────────────────────────────────

/// An integer pixel coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelPos {
    pub x: i64,
    pub y: i64,
}

/// A floating-point position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubPixelPos {
    pub x: f64,
    pub y: f64,
}

// ── Snap to Pixel Grid ─────────────────────────────────────────

/// Snap a floating-point position to the nearest pixel grid point.
pub fn snap_to_pixel(x: f64, y: f64) -> PixelPos {
    PixelPos {
        x: x.round() as i64,
        y: y.round() as i64,
    }
}

/// Snap towards zero (floor for positive, ceil for negative).
pub fn snap_floor(x: f64, y: f64) -> PixelPos {
    PixelPos {
        x: x.floor() as i64,
        y: y.floor() as i64,
    }
}

// ── Sub-Pixel Motion Accumulator ───────────────────────────────

/// Accumulates fractional movement and emits whole-pixel steps.
///
/// Calling `accumulate(dx, dy)` returns the integer pixels to move *now*,
/// while carrying the fractional remainder internally.
#[derive(Debug, Clone, PartialEq)]
pub struct SubPixelAccumulator {
    remainder_x: f64,
    remainder_y: f64,
}

impl SubPixelAccumulator {
    pub fn new() -> Self {
        Self { remainder_x: 0.0, remainder_y: 0.0 }
    }

    /// Add a fractional movement delta. Returns the integer pixel delta
    /// to apply this frame.
    pub fn accumulate(&mut self, dx: f64, dy: f64) -> (i64, i64) {
        self.remainder_x += dx;
        self.remainder_y += dy;

        let move_x = self.remainder_x.trunc() as i64;
        let move_y = self.remainder_y.trunc() as i64;

        self.remainder_x -= move_x as f64;
        self.remainder_y -= move_y as f64;

        (move_x, move_y)
    }

    pub fn remainder(&self) -> (f64, f64) {
        (self.remainder_x, self.remainder_y)
    }

    pub fn reset(&mut self) {
        self.remainder_x = 0.0;
        self.remainder_y = 0.0;
    }
}

impl Default for SubPixelAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Pixel-Perfect Camera ───────────────────────────────────────

/// A camera that always aligns to integer pixel positions in the
/// reference resolution, preventing sub-pixel jitter.
#[derive(Debug, Clone, PartialEq)]
pub struct PixelCamera {
    /// Floating-point target position (smoothly interpolated).
    pub target_x: f64,
    pub target_y: f64,
    /// Snapped integer position actually used for rendering.
    pub snapped_x: i64,
    pub snapped_y: i64,
    /// Viewport size in reference-resolution pixels.
    pub viewport_w: u32,
    pub viewport_h: u32,
}

impl PixelCamera {
    pub fn new(vw: u32, vh: u32) -> Self {
        Self {
            target_x: 0.0,
            target_y: 0.0,
            snapped_x: 0,
            snapped_y: 0,
            viewport_w: vw,
            viewport_h: vh,
        }
    }

    /// Set the target position (e.g., follow the player).
    pub fn set_target(&mut self, x: f64, y: f64) {
        self.target_x = x;
        self.target_y = y;
        self.snapped_x = x.round() as i64;
        self.snapped_y = y.round() as i64;
    }

    /// Smoothly move towards a target with linear interpolation.
    pub fn lerp_towards(&mut self, x: f64, y: f64, t: f64) {
        let t = t.clamp(0.0, 1.0);
        self.target_x += (x - self.target_x) * t;
        self.target_y += (y - self.target_y) * t;
        self.snapped_x = self.target_x.round() as i64;
        self.snapped_y = self.target_y.round() as i64;
    }

    /// The sub-pixel error between target and snapped (useful for
    /// post-process shift).
    pub fn sub_pixel_offset(&self) -> (f64, f64) {
        (
            self.target_x - self.snapped_x as f64,
            self.target_y - self.snapped_y as f64,
        )
    }

    /// Top-left corner of the camera viewport in world pixels.
    pub fn top_left(&self) -> PixelPos {
        PixelPos {
            x: self.snapped_x - (self.viewport_w as i64) / 2,
            y: self.snapped_y - (self.viewport_h as i64) / 2,
        }
    }
}

// ── Upscale Filter Modes ───────────────────────────────────────

/// Upscale filter for rendering the low-res framebuffer to screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleFilter {
    /// Nearest-neighbor — crisp pixels, standard for pixel art.
    NearestNeighbor,
    /// Bilinear — smooth, generally unwanted for pixel art.
    Bilinear,
    /// Integer scaling only (black borders if not exact multiple).
    IntegerScale,
}

// ── Resolution Scaling ─────────────────────────────────────────

/// Handles mapping between a fixed "reference" resolution (e.g., 320×180)
/// and the actual display resolution (e.g., 1920×1080).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionScaler {
    pub ref_width: u32,
    pub ref_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub filter: UpscaleFilter,
}

impl ResolutionScaler {
    pub fn new(ref_w: u32, ref_h: u32, disp_w: u32, disp_h: u32, filter: UpscaleFilter) -> Self {
        Self {
            ref_width: ref_w,
            ref_height: ref_h,
            display_width: disp_w,
            display_height: disp_h,
            filter,
        }
    }

    /// Integer scale factor (largest integer that fits).
    pub fn integer_scale(&self) -> u32 {
        if self.ref_width == 0 || self.ref_height == 0 {
            return 1;
        }
        let sx = self.display_width / self.ref_width;
        let sy = self.display_height / self.ref_height;
        sx.min(sy).max(1)
    }

    /// Floating-point scale factor to fill the display (aspect-correct).
    pub fn fit_scale(&self) -> f64 {
        if self.ref_width == 0 || self.ref_height == 0 {
            return 1.0;
        }
        let sx = self.display_width as f64 / self.ref_width as f64;
        let sy = self.display_height as f64 / self.ref_height as f64;
        sx.min(sy)
    }

    /// Effective scale based on the chosen filter.
    pub fn effective_scale(&self) -> f64 {
        match self.filter {
            UpscaleFilter::IntegerScale => self.integer_scale() as f64,
            _ => self.fit_scale(),
        }
    }

    /// Viewport offset (black bar margins) for centering.
    pub fn viewport_offset(&self) -> (i32, i32) {
        let s = self.effective_scale();
        let rendered_w = self.ref_width as f64 * s;
        let rendered_h = self.ref_height as f64 * s;
        let ox = ((self.display_width as f64 - rendered_w) / 2.0).round() as i32;
        let oy = ((self.display_height as f64 - rendered_h) / 2.0).round() as i32;
        (ox, oy)
    }

    /// Convert a display-space coordinate to reference-resolution coordinate.
    pub fn display_to_ref(&self, disp_x: f64, disp_y: f64) -> (f64, f64) {
        let s = self.effective_scale();
        let (ox, oy) = self.viewport_offset();
        let rx = (disp_x - ox as f64) / s;
        let ry = (disp_y - oy as f64) / s;
        (rx, ry)
    }

    /// Convert reference coordinate to display coordinate.
    pub fn ref_to_display(&self, ref_x: f64, ref_y: f64) -> (f64, f64) {
        let s = self.effective_scale();
        let (ox, oy) = self.viewport_offset();
        let dx = ref_x * s + ox as f64;
        let dy = ref_y * s + oy as f64;
        (dx, dy)
    }
}

// ── Bresenham Line Drawing ─────────────────────────────────────

/// Pixel-aligned line drawing using Bresenham's algorithm.
/// Returns a list of pixel positions forming the line from `(x0,y0)` to
/// `(x1,y1)`.
pub fn bresenham_line(x0: i64, y0: i64, x1: i64, y1: i64) -> Vec<PixelPos> {
    let mut points = Vec::new();

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i64 = if x0 < x1 { 1 } else { -1 };
    let sy: i64 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let mut cx = x0;
    let mut cy = y0;

    loop {
        points.push(PixelPos { x: cx, y: cy });
        if cx == x1 && cy == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }

    points
}

/// Draw a pixel-aligned rectangle outline.
pub fn pixel_rect_outline(x: i64, y: i64, w: u32, h: u32) -> Vec<PixelPos> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let x1 = x + w as i64 - 1;
    let y1 = y + h as i64 - 1;
    let mut pts = Vec::new();
    // Top
    for px in x..=x1 {
        pts.push(PixelPos { x: px, y });
    }
    // Right (skip corners)
    for py in (y + 1)..y1 {
        pts.push(PixelPos { x: x1, y: py });
    }
    // Bottom
    if h > 1 {
        for px in (x..=x1).rev() {
            pts.push(PixelPos { x: px, y: y1 });
        }
    }
    // Left (skip corners)
    if w > 1 {
        for py in ((y + 1)..y1).rev() {
            pts.push(PixelPos { x, y: py });
        }
    }
    pts
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn snap_positive() {
        let p = snap_to_pixel(3.7, 4.2);
        assert_eq!(p, PixelPos { x: 4, y: 4 });
    }

    #[test]
    fn snap_negative() {
        let p = snap_to_pixel(-1.3, -2.8);
        assert_eq!(p, PixelPos { x: -1, y: -3 });
    }

    #[test]
    fn snap_floor_positive() {
        let p = snap_floor(3.9, 4.9);
        assert_eq!(p, PixelPos { x: 3, y: 4 });
    }

    #[test]
    fn snap_floor_negative() {
        let p = snap_floor(-0.1, -0.9);
        assert_eq!(p, PixelPos { x: -1, y: -1 });
    }

    #[test]
    fn accumulator_no_movement_below_pixel() {
        let mut acc = SubPixelAccumulator::new();
        let (mx, my) = acc.accumulate(0.3, 0.4);
        assert_eq!(mx, 0);
        assert_eq!(my, 0);
    }

    #[test]
    fn accumulator_accumulates_across_frames() {
        let mut acc = SubPixelAccumulator::new();
        acc.accumulate(0.3, 0.0);
        acc.accumulate(0.3, 0.0);
        acc.accumulate(0.3, 0.0);
        let (mx, _) = acc.accumulate(0.3, 0.0);
        // 4 × 0.3 = 1.2 → emits 1 so far, remainder ~0.2
        // Actually: 0.3+0.3=0.6→0, 0.6+0.3=0.9→0, 0.9+0.3=1.2→1
        assert_eq!(mx, 1);
    }

    #[test]
    fn accumulator_negative_movement() {
        let mut acc = SubPixelAccumulator::new();
        let (mx, _) = acc.accumulate(-1.5, 0.0);
        assert_eq!(mx, -1);
        let (r, _) = acc.remainder();
        assert!((r - (-0.5)).abs() < EPS);
    }

    #[test]
    fn accumulator_reset() {
        let mut acc = SubPixelAccumulator::new();
        acc.accumulate(0.7, 0.8);
        acc.reset();
        let (rx, ry) = acc.remainder();
        assert!((rx).abs() < EPS);
        assert!((ry).abs() < EPS);
    }

    #[test]
    fn pixel_camera_snap() {
        let mut cam = PixelCamera::new(320, 180);
        cam.set_target(100.3, 50.7);
        assert_eq!(cam.snapped_x, 100);
        assert_eq!(cam.snapped_y, 51);
    }

    #[test]
    fn pixel_camera_lerp() {
        let mut cam = PixelCamera::new(320, 180);
        cam.set_target(0.0, 0.0);
        cam.lerp_towards(100.0, 100.0, 0.5);
        assert!((cam.target_x - 50.0).abs() < EPS);
        assert!((cam.target_y - 50.0).abs() < EPS);
        assert_eq!(cam.snapped_x, 50);
        assert_eq!(cam.snapped_y, 50);
    }

    #[test]
    fn pixel_camera_sub_pixel_offset() {
        let mut cam = PixelCamera::new(320, 180);
        cam.set_target(10.3, 20.7);
        let (ox, oy) = cam.sub_pixel_offset();
        assert!((ox - 0.3).abs() < EPS);
        assert!((oy - (-0.3)).abs() < EPS);
    }

    #[test]
    fn pixel_camera_top_left() {
        let mut cam = PixelCamera::new(320, 180);
        cam.set_target(160.0, 90.0);
        let tl = cam.top_left();
        assert_eq!(tl, PixelPos { x: 0, y: 0 });
    }

    #[test]
    fn resolution_scaler_integer_scale() {
        let s = ResolutionScaler::new(320, 180, 1920, 1080, UpscaleFilter::IntegerScale);
        assert_eq!(s.integer_scale(), 6); // 1920/320=6, 1080/180=6
    }

    #[test]
    fn resolution_scaler_fit_scale() {
        let s = ResolutionScaler::new(320, 180, 1600, 900, UpscaleFilter::NearestNeighbor);
        assert!((s.fit_scale() - 5.0).abs() < EPS);
    }

    #[test]
    fn resolution_scaler_non_integer_fit() {
        let s = ResolutionScaler::new(320, 180, 1000, 600, UpscaleFilter::NearestNeighbor);
        // 1000/320=3.125, 600/180=3.333 → min=3.125
        assert!((s.fit_scale() - 3.125).abs() < EPS);
    }

    #[test]
    fn resolution_scaler_viewport_offset() {
        let s = ResolutionScaler::new(320, 180, 1920, 1080, UpscaleFilter::IntegerScale);
        let (ox, oy) = s.viewport_offset();
        assert_eq!(ox, 0);
        assert_eq!(oy, 0);
    }

    #[test]
    fn resolution_scaler_round_trip() {
        let s = ResolutionScaler::new(320, 180, 1920, 1080, UpscaleFilter::IntegerScale);
        let (dx, dy) = s.ref_to_display(100.0, 50.0);
        let (rx, ry) = s.display_to_ref(dx, dy);
        assert!((rx - 100.0).abs() < EPS);
        assert!((ry - 50.0).abs() < EPS);
    }

    #[test]
    fn resolution_scaler_zero_ref() {
        let s = ResolutionScaler::new(0, 0, 1920, 1080, UpscaleFilter::NearestNeighbor);
        assert_eq!(s.integer_scale(), 1);
        assert!((s.fit_scale() - 1.0).abs() < EPS);
    }

    #[test]
    fn bresenham_horizontal() {
        let pts = bresenham_line(0, 0, 5, 0);
        assert_eq!(pts.len(), 6);
        for (i, p) in pts.iter().enumerate() {
            assert_eq!(p.x, i as i64);
            assert_eq!(p.y, 0);
        }
    }

    #[test]
    fn bresenham_vertical() {
        let pts = bresenham_line(0, 0, 0, 4);
        assert_eq!(pts.len(), 5);
        for (i, p) in pts.iter().enumerate() {
            assert_eq!(p.x, 0);
            assert_eq!(p.y, i as i64);
        }
    }

    #[test]
    fn bresenham_diagonal() {
        let pts = bresenham_line(0, 0, 3, 3);
        assert_eq!(pts.len(), 4);
        for p in &pts {
            assert_eq!(p.x, p.y);
        }
    }

    #[test]
    fn bresenham_reverse() {
        let fwd = bresenham_line(0, 0, 4, 3);
        let rev = bresenham_line(4, 3, 0, 0);
        assert_eq!(fwd.len(), rev.len());
        // Both lines start and end at the correct endpoints
        assert_eq!(fwd.first().unwrap(), &PixelPos { x: 0, y: 0 });
        assert_eq!(fwd.last().unwrap(), &PixelPos { x: 4, y: 3 });
        assert_eq!(rev.first().unwrap(), &PixelPos { x: 4, y: 3 });
        assert_eq!(rev.last().unwrap(), &PixelPos { x: 0, y: 0 });
    }

    #[test]
    fn bresenham_single_point() {
        let pts = bresenham_line(5, 5, 5, 5);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0], PixelPos { x: 5, y: 5 });
    }

    #[test]
    fn pixel_rect_outline_basic() {
        let pts = pixel_rect_outline(0, 0, 4, 3);
        // 4×3 rect: top=4, right=1, bottom=4, left=1 = 10 unique pixels
        assert_eq!(pts.len(), 10);
    }

    #[test]
    fn pixel_rect_outline_1x1() {
        let pts = pixel_rect_outline(0, 0, 1, 1);
        assert_eq!(pts.len(), 1);
    }

    #[test]
    fn pixel_rect_outline_empty() {
        let pts = pixel_rect_outline(0, 0, 0, 5);
        assert!(pts.is_empty());
    }
}
