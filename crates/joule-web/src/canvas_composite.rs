//! Canvas compositing operations: Porter-Duff compositing modes (source-over,
//! source-in, source-out, source-atop, destination-*, lighter, xor, copy),
//! blend modes (multiply, screen, overlay, darken, lighten), global alpha.
//!
//! Pure math — no browser dependency.

use std::fmt;

// ── Color ──────────────────────────────────────────────────────

/// Premultiplied RGBA color, channels in 0.0–1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn transparent() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn clamp(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }

    /// Apply global alpha, scaling the alpha channel.
    pub fn with_global_alpha(self, alpha: f64) -> Self {
        Self {
            r: self.r,
            g: self.g,
            b: self.b,
            a: self.a * alpha,
        }
    }

    /// Convert to premultiplied form.
    pub fn premultiply(self) -> Self {
        Self {
            r: self.r * self.a,
            g: self.g * self.a,
            b: self.b * self.a,
            a: self.a,
        }
    }

    /// Convert from premultiplied to straight alpha.
    pub fn unpremultiply(self) -> Self {
        if self.a < 1e-10 {
            return Self::transparent();
        }
        Self {
            r: self.r / self.a,
            g: self.g / self.a,
            b: self.b / self.a,
            a: self.a,
        }
    }

    /// Luminance (BT.709).
    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "rgba({:.0}, {:.0}, {:.0}, {:.2})",
            self.r * 255.0,
            self.g * 255.0,
            self.b * 255.0,
            self.a
        )
    }
}

// ── Porter-Duff Compositing ────────────────────────────────────

/// Porter-Duff compositing operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeOp {
    SourceOver,
    SourceIn,
    SourceOut,
    SourceAtop,
    DestinationOver,
    DestinationIn,
    DestinationOut,
    DestinationAtop,
    Lighter,
    Copy,
    Xor,
    Clear,
}

impl CompositeOp {
    /// All compositing operations.
    pub fn all() -> &'static [CompositeOp] {
        &[
            CompositeOp::SourceOver,
            CompositeOp::SourceIn,
            CompositeOp::SourceOut,
            CompositeOp::SourceAtop,
            CompositeOp::DestinationOver,
            CompositeOp::DestinationIn,
            CompositeOp::DestinationOut,
            CompositeOp::DestinationAtop,
            CompositeOp::Lighter,
            CompositeOp::Copy,
            CompositeOp::Xor,
            CompositeOp::Clear,
        ]
    }
}

/// Apply a Porter-Duff compositing operation.
/// Both src and dst should be in straight (non-premultiplied) alpha.
pub fn composite(src: Color, dst: Color, op: CompositeOp) -> Color {
    let sa = src.a;
    let da = dst.a;

    match op {
        CompositeOp::Clear => Color::transparent(),

        CompositeOp::Copy => src,

        CompositeOp::SourceOver => {
            let oa = sa + da * (1.0 - sa);
            if oa < 1e-10 {
                return Color::transparent();
            }
            Color::new(
                (src.r * sa + dst.r * da * (1.0 - sa)) / oa,
                (src.g * sa + dst.g * da * (1.0 - sa)) / oa,
                (src.b * sa + dst.b * da * (1.0 - sa)) / oa,
                oa,
            )
        }

        CompositeOp::SourceIn => {
            let oa = sa * da;
            Color::new(src.r, src.g, src.b, oa)
        }

        CompositeOp::SourceOut => {
            let oa = sa * (1.0 - da);
            Color::new(src.r, src.g, src.b, oa)
        }

        CompositeOp::SourceAtop => {
            let oa = da;
            if oa < 1e-10 {
                return Color::transparent();
            }
            Color::new(
                src.r * sa + dst.r * (1.0 - sa),
                src.g * sa + dst.g * (1.0 - sa),
                src.b * sa + dst.b * (1.0 - sa),
                oa,
            )
        }

        CompositeOp::DestinationOver => {
            let oa = da + sa * (1.0 - da);
            if oa < 1e-10 {
                return Color::transparent();
            }
            Color::new(
                (dst.r * da + src.r * sa * (1.0 - da)) / oa,
                (dst.g * da + src.g * sa * (1.0 - da)) / oa,
                (dst.b * da + src.b * sa * (1.0 - da)) / oa,
                oa,
            )
        }

        CompositeOp::DestinationIn => {
            let oa = da * sa;
            Color::new(dst.r, dst.g, dst.b, oa)
        }

        CompositeOp::DestinationOut => {
            let oa = da * (1.0 - sa);
            Color::new(dst.r, dst.g, dst.b, oa)
        }

        CompositeOp::DestinationAtop => {
            let oa = sa;
            if oa < 1e-10 {
                return Color::transparent();
            }
            Color::new(
                dst.r * da + src.r * (1.0 - da),
                dst.g * da + src.g * (1.0 - da),
                dst.b * da + src.b * (1.0 - da),
                oa,
            )
        }

        CompositeOp::Xor => {
            let oa = sa * (1.0 - da) + da * (1.0 - sa);
            if oa < 1e-10 {
                return Color::transparent();
            }
            Color::new(
                (src.r * sa * (1.0 - da) + dst.r * da * (1.0 - sa)) / oa,
                (src.g * sa * (1.0 - da) + dst.g * da * (1.0 - sa)) / oa,
                (src.b * sa * (1.0 - da) + dst.b * da * (1.0 - sa)) / oa,
                oa,
            )
        }

        CompositeOp::Lighter => {
            Color::new(
                (src.r * sa + dst.r * da).min(1.0),
                (src.g * sa + dst.g * da).min(1.0),
                (src.b * sa + dst.b * da).min(1.0),
                (sa + da).min(1.0),
            )
        }
    }
}

// ── Blend Modes ────────────────────────────────────────────────

/// CSS/Canvas blend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
}

impl BlendMode {
    pub fn all() -> &'static [BlendMode] {
        &[
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
        ]
    }
}

/// Blend a single channel value using the given blend mode.
fn blend_channel(src: f64, dst: f64, mode: BlendMode) -> f64 {
    match mode {
        BlendMode::Normal => src,
        BlendMode::Multiply => src * dst,
        BlendMode::Screen => src + dst - src * dst,
        BlendMode::Overlay => {
            if dst < 0.5 {
                2.0 * src * dst
            } else {
                1.0 - 2.0 * (1.0 - src) * (1.0 - dst)
            }
        }
        BlendMode::Darken => src.min(dst),
        BlendMode::Lighten => src.max(dst),
        BlendMode::ColorDodge => {
            if src >= 1.0 {
                1.0
            } else {
                (dst / (1.0 - src)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if src <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - dst) / src).min(1.0)
            }
        }
        BlendMode::HardLight => {
            if src < 0.5 {
                2.0 * src * dst
            } else {
                1.0 - 2.0 * (1.0 - src) * (1.0 - dst)
            }
        }
        BlendMode::SoftLight => {
            if src <= 0.5 {
                dst - (1.0 - 2.0 * src) * dst * (1.0 - dst)
            } else {
                let d = if dst <= 0.25 {
                    ((16.0 * dst - 12.0) * dst + 4.0) * dst
                } else {
                    dst.sqrt()
                };
                dst + (2.0 * src - 1.0) * (d - dst)
            }
        }
        BlendMode::Difference => (src - dst).abs(),
        BlendMode::Exclusion => src + dst - 2.0 * src * dst,
    }
}

/// Blend two colors using the given blend mode, then composite with source-over.
pub fn blend(src: Color, dst: Color, mode: BlendMode) -> Color {
    let blended_r = blend_channel(src.r, dst.r, mode);
    let blended_g = blend_channel(src.g, dst.g, mode);
    let blended_b = blend_channel(src.b, dst.b, mode);

    // Composite: result = blended * src.a + dst * (1 - src.a)
    let sa = src.a;
    let da = dst.a;
    let oa = sa + da * (1.0 - sa);

    if oa < 1e-10 {
        return Color::transparent();
    }

    Color::new(
        (blended_r * sa + dst.r * da * (1.0 - sa)) / oa,
        (blended_g * sa + dst.g * da * (1.0 - sa)) / oa,
        (blended_b * sa + dst.b * da * (1.0 - sa)) / oa,
        oa,
    )
}

// ── Compositing Context ────────────────────────────────────────

/// A compositing context that tracks global alpha, composite op, and blend mode.
#[derive(Debug, Clone)]
pub struct CompositingContext {
    pub global_alpha: f64,
    pub composite_op: CompositeOp,
    pub blend_mode: BlendMode,
}

impl Default for CompositingContext {
    fn default() -> Self {
        Self {
            global_alpha: 1.0,
            composite_op: CompositeOp::SourceOver,
            blend_mode: BlendMode::Normal,
        }
    }
}

impl CompositingContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Draw `src` onto `dst` using the current composite and blend settings.
    pub fn draw(&self, src: Color, dst: Color) -> Color {
        let adjusted_src = src.with_global_alpha(self.global_alpha);

        if self.blend_mode == BlendMode::Normal {
            composite(adjusted_src, dst, self.composite_op)
        } else {
            // Apply blend, then composite
            let blended = blend(adjusted_src, dst, self.blend_mode);
            // For non-normal blend modes, we use source-over compositing on the blended result
            blended
        }
    }
}

// ── Image Compositing ──────────────────────────────────────────

/// A simple pixel buffer for compositing.
#[derive(Debug, Clone)]
pub struct PixelBuffer {
    pub width: usize,
    pub height: usize,
    pub data: Vec<Color>,
}

impl PixelBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![Color::transparent(); width * height],
        }
    }

    pub fn filled(width: usize, height: usize, color: Color) -> Self {
        Self {
            width,
            height,
            data: vec![color; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Color {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            Color::transparent()
        }
    }

    pub fn set(&mut self, x: usize, y: usize, c: Color) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = c;
        }
    }

    /// Composite `src` buffer onto `self` at offset (ox, oy).
    pub fn composite_buffer(&mut self, src: &PixelBuffer, ox: usize, oy: usize, ctx: &CompositingContext) {
        for y in 0..src.height {
            let dy = oy + y;
            if dy >= self.height {
                break;
            }
            for x in 0..src.width {
                let dx = ox + x;
                if dx >= self.width {
                    break;
                }
                let sc = src.get(x, y);
                let dc = self.get(dx, dy);
                self.set(dx, dy, ctx.draw(sc, dc));
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn red() -> Color {
        Color::new(1.0, 0.0, 0.0, 1.0)
    }

    fn green() -> Color {
        Color::new(0.0, 1.0, 0.0, 1.0)
    }

    fn blue_half() -> Color {
        Color::new(0.0, 0.0, 1.0, 0.5)
    }

    #[test]
    fn test_source_over_opaque() {
        let r = composite(red(), green(), CompositeOp::SourceOver);
        assert!((r.r - 1.0).abs() < 1e-10);
        assert!(r.g.abs() < 1e-10);
    }

    #[test]
    fn test_source_over_semitransparent() {
        let r = composite(blue_half(), red(), CompositeOp::SourceOver);
        assert!(r.a > 0.99);
        assert!(r.b > 0.0);
        assert!(r.r > 0.0);
    }

    #[test]
    fn test_source_in() {
        let r = composite(red(), blue_half(), CompositeOp::SourceIn);
        assert!((r.r - 1.0).abs() < 1e-10);
        assert!((r.a - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_source_out() {
        let r = composite(red(), Color::new(0.0, 1.0, 0.0, 0.0), CompositeOp::SourceOut);
        assert!((r.a - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_copy() {
        let r = composite(red(), green(), CompositeOp::Copy);
        assert!((r.r - 1.0).abs() < 1e-10);
        assert!(r.g.abs() < 1e-10);
    }

    #[test]
    fn test_clear() {
        let r = composite(red(), green(), CompositeOp::Clear);
        assert!(r.a.abs() < 1e-10);
    }

    #[test]
    fn test_xor_same_alpha() {
        let r = composite(
            Color::new(1.0, 0.0, 0.0, 0.5),
            Color::new(0.0, 1.0, 0.0, 0.5),
            CompositeOp::Xor,
        );
        assert!((r.a - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_blend_multiply() {
        let r = blend_channel(0.5, 0.5, BlendMode::Multiply);
        assert!((r - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_blend_screen() {
        let r = blend_channel(0.5, 0.5, BlendMode::Screen);
        assert!((r - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_blend_overlay_dark() {
        let r = blend_channel(0.3, 0.2, BlendMode::Overlay);
        assert!((r - 2.0 * 0.3 * 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_blend_darken() {
        let r = blend_channel(0.3, 0.7, BlendMode::Darken);
        assert!((r - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_blend_lighten() {
        let r = blend_channel(0.3, 0.7, BlendMode::Lighten);
        assert!((r - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_global_alpha() {
        let c = Color::new(1.0, 0.0, 0.0, 1.0).with_global_alpha(0.5);
        assert!((c.a - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compositing_context_default() {
        let ctx = CompositingContext::default();
        assert!((ctx.global_alpha - 1.0).abs() < 1e-10);
        assert_eq!(ctx.composite_op, CompositeOp::SourceOver);
        assert_eq!(ctx.blend_mode, BlendMode::Normal);
    }

    #[test]
    fn test_pixel_buffer_composite() {
        let mut dst = PixelBuffer::filled(4, 4, green());
        let src = PixelBuffer::filled(2, 2, red());
        let ctx = CompositingContext::default();
        dst.composite_buffer(&src, 1, 1, &ctx);
        // (1,1) should be red (source-over with opaque)
        let p = dst.get(1, 1);
        assert!((p.r - 1.0).abs() < 1e-10);
        assert!(p.g.abs() < 1e-10);
        // (0,0) should still be green
        let p0 = dst.get(0, 0);
        assert!((p0.g - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_premultiply_unpremultiply() {
        let c = Color::new(1.0, 0.5, 0.0, 0.5);
        let pm = c.premultiply();
        assert!((pm.r - 0.5).abs() < 1e-10);
        let un = pm.unpremultiply();
        assert!((un.r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_color_display() {
        let c = Color::new(1.0, 0.0, 0.0, 1.0);
        let s = format!("{c}");
        assert!(s.contains("255"));
    }

    #[test]
    fn test_blend_difference() {
        let r = blend_channel(0.8, 0.3, BlendMode::Difference);
        assert!((r - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_destination_over() {
        let r = composite(blue_half(), red(), CompositeOp::DestinationOver);
        // Red is opaque, so destination-over keeps red dominant
        assert!(r.r > 0.4);
    }
}
