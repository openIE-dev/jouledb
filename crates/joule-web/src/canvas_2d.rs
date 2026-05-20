// canvas_2d.rs — 2D canvas: RGBA pixel buffer, drawing primitives
// (line/rect/circle/fill), color with alpha blending, transform
// stack (push/pop), clipping rect, clear, pixel read/write.

/// An RGBA color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const RED: Self = Self::rgb(255, 0, 0);
    pub const GREEN: Self = Self::rgb(0, 255, 0);
    pub const BLUE: Self = Self::rgb(0, 0, 255);
    pub const TRANSPARENT: Self = Self::rgba(0, 0, 0, 0);

    /// Alpha-blend `self` (foreground) over `dst` (background).
    pub fn blend_over(self, dst: Color) -> Color {
        if self.a == 255 {
            return self;
        }
        if self.a == 0 {
            return dst;
        }
        let sa = self.a as u16;
        let da = dst.a as u16;
        let inv_sa = 255 - sa;
        let out_a = sa + (da * inv_sa) / 255;
        if out_a == 0 {
            return Color::TRANSPARENT;
        }
        let blend = |sc: u8, dc: u8| -> u8 {
            let s = sc as u32;
            let d = dc as u32;
            let sa32 = sa as u32;
            let da32 = da as u32;
            let inv_sa32 = inv_sa as u32;
            let out_a32 = out_a as u32;
            ((s * sa32 + d * da32 * inv_sa32 / 255) / out_a32) as u8
        };
        Color {
            r: blend(self.r, dst.r),
            g: blend(self.g, dst.g),
            b: blend(self.b, dst.b),
            a: out_a as u8,
        }
    }
}

/// A clipping rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipRect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl ClipRect {
    pub fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x
            && py >= self.y
            && px < self.x + self.w as i32
            && py < self.y + self.h as i32
    }
}

/// A 2D affine transform (translate only for simplicity; stack supports push/pop).
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub tx: i32,
    pub ty: i32,
}

impl Transform {
    pub fn identity() -> Self {
        Self { tx: 0, ty: 0 }
    }

    pub fn translate(tx: i32, ty: i32) -> Self {
        Self { tx, ty }
    }

    pub fn apply(&self, x: i32, y: i32) -> (i32, i32) {
        (x + self.tx, y + self.ty)
    }

    pub fn combine(&self, other: &Transform) -> Transform {
        Transform {
            tx: self.tx + other.tx,
            ty: self.ty + other.ty,
        }
    }
}

/// A 2D pixel-buffer canvas.
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pixels: Vec<Color>,
    clip: Option<ClipRect>,
    transform_stack: Vec<Transform>,
    current_transform: Transform,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        let count = (width as usize) * (height as usize);
        Self {
            width,
            height,
            pixels: vec![Color::TRANSPARENT; count],
            clip: None,
            transform_stack: Vec::new(),
            current_transform: Transform::identity(),
        }
    }

    // ---- Transform stack ----

    pub fn push_transform(&mut self, t: Transform) {
        self.transform_stack.push(self.current_transform);
        self.current_transform = self.current_transform.combine(&t);
    }

    pub fn pop_transform(&mut self) {
        if let Some(prev) = self.transform_stack.pop() {
            self.current_transform = prev;
        }
    }

    pub fn current_transform(&self) -> Transform {
        self.current_transform
    }

    // ---- Clipping ----

    pub fn set_clip(&mut self, clip: ClipRect) {
        self.clip = Some(clip);
    }

    pub fn clear_clip(&mut self) {
        self.clip = None;
    }

    // ---- Pixel access ----

    fn index_of(&self, x: u32, y: u32) -> Option<usize> {
        if x < self.width && y < self.height {
            Some((y as usize) * (self.width as usize) + (x as usize))
        } else {
            None
        }
    }

    fn is_clipped(&self, x: i32, y: i32) -> bool {
        if let Some(clip) = &self.clip {
            !clip.contains(x, y)
        } else {
            false
        }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> Color {
        self.index_of(x, y)
            .map(|i| self.pixels[i])
            .unwrap_or(Color::TRANSPARENT)
    }

    pub fn set_pixel(&mut self, x: i32, y: i32, color: Color) {
        let (tx, ty) = self.current_transform.apply(x, y);
        if self.is_clipped(tx, ty) {
            return;
        }
        if tx < 0 || ty < 0 {
            return;
        }
        let ux = tx as u32;
        let uy = ty as u32;
        if let Some(idx) = self.index_of(ux, uy) {
            let dst = self.pixels[idx];
            self.pixels[idx] = color.blend_over(dst);
        }
    }

    // ---- Clear ----

    pub fn clear(&mut self, color: Color) {
        for p in &mut self.pixels {
            *p = color;
        }
    }

    // ---- Drawing primitives ----

    /// Draw a horizontal line.
    pub fn draw_hline(&mut self, x1: i32, x2: i32, y: i32, color: Color) {
        let start = x1.min(x2);
        let end = x1.max(x2);
        for x in start..=end {
            self.set_pixel(x, y, color);
        }
    }

    /// Draw a vertical line.
    pub fn draw_vline(&mut self, x: i32, y1: i32, y2: i32, color: Color) {
        let start = y1.min(y2);
        let end = y1.max(y2);
        for y in start..=end {
            self.set_pixel(x, y, color);
        }
    }

    /// Bresenham line drawing.
    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: Color) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut cx = x0;
        let mut cy = y0;
        loop {
            self.set_pixel(cx, cy, color);
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
    }

    /// Draw an axis-aligned rectangle outline.
    pub fn draw_rect(&mut self, x: i32, y: i32, w: u32, h: u32, color: Color) {
        let w = w as i32;
        let h = h as i32;
        self.draw_hline(x, x + w - 1, y, color);
        self.draw_hline(x, x + w - 1, y + h - 1, color);
        self.draw_vline(x, y, y + h - 1, color);
        self.draw_vline(x + w - 1, y, y + h - 1, color);
    }

    /// Fill a rectangle.
    pub fn fill_rect(&mut self, x: i32, y: i32, w: u32, h: u32, color: Color) {
        for row in 0..h as i32 {
            for col in 0..w as i32 {
                self.set_pixel(x + col, y + row, color);
            }
        }
    }

    /// Draw a circle outline (midpoint algorithm).
    pub fn draw_circle(&mut self, cx: i32, cy: i32, radius: u32, color: Color) {
        let r = radius as i32;
        let mut x = r;
        let mut y = 0i32;
        let mut err = 1 - r;

        while x >= y {
            self.set_pixel(cx + x, cy + y, color);
            self.set_pixel(cx - x, cy + y, color);
            self.set_pixel(cx + x, cy - y, color);
            self.set_pixel(cx - x, cy - y, color);
            self.set_pixel(cx + y, cy + x, color);
            self.set_pixel(cx - y, cy + x, color);
            self.set_pixel(cx + y, cy - x, color);
            self.set_pixel(cx - y, cy - x, color);
            y += 1;
            if err < 0 {
                err += 2 * y + 1;
            } else {
                x -= 1;
                err += 2 * (y - x) + 1;
            }
        }
    }

    /// Fill a circle.
    pub fn fill_circle(&mut self, cx: i32, cy: i32, radius: u32, color: Color) {
        let r = radius as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    self.set_pixel(cx + dx, cy + dy, color);
                }
            }
        }
    }

    /// Flood fill (4-connected) from a seed pixel, replacing matching color.
    pub fn flood_fill(&mut self, x: i32, y: i32, fill_color: Color) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let target = self.get_pixel(x as u32, y as u32);
        if target == fill_color {
            return;
        }
        // Apply transform to check bounds, but flood_fill works in canvas coords.
        let mut stack = vec![(x, y)];
        while let Some((px, py)) = stack.pop() {
            if px < 0 || py < 0 || px >= self.width as i32 || py >= self.height as i32 {
                continue;
            }
            let idx = (py as usize) * (self.width as usize) + (px as usize);
            if self.pixels[idx] != target {
                continue;
            }
            self.pixels[idx] = fill_color;
            stack.push((px + 1, py));
            stack.push((px - 1, py));
            stack.push((px, py + 1));
            stack.push((px, py - 1));
        }
    }

    /// Total number of pixels.
    pub fn pixel_count(&self) -> usize {
        self.pixels.len()
    }

    /// Count pixels matching a given color.
    pub fn count_color(&self, color: Color) -> usize {
        self.pixels.iter().filter(|&&c| c == color).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_blend_opaque() {
        let fg = Color::RED;
        let bg = Color::BLUE;
        let result = fg.blend_over(bg);
        assert_eq!(result, Color::RED);
    }

    #[test]
    fn test_color_blend_transparent() {
        let fg = Color::TRANSPARENT;
        let bg = Color::GREEN;
        let result = fg.blend_over(bg);
        assert_eq!(result, Color::GREEN);
    }

    #[test]
    fn test_color_blend_semi() {
        let fg = Color::rgba(255, 0, 0, 128);
        let bg = Color::rgb(0, 0, 255);
        let result = fg.blend_over(bg);
        // Red channel should be dominant, blue somewhat present.
        assert!(result.r > 100);
        assert!(result.b > 50);
        assert!(result.a > 200);
    }

    #[test]
    fn test_canvas_new() {
        let c = Canvas::new(10, 10);
        assert_eq!(c.pixel_count(), 100);
        assert_eq!(c.get_pixel(0, 0), Color::TRANSPARENT);
    }

    #[test]
    fn test_canvas_clear() {
        let mut c = Canvas::new(5, 5);
        c.clear(Color::WHITE);
        assert_eq!(c.get_pixel(0, 0), Color::WHITE);
        assert_eq!(c.get_pixel(4, 4), Color::WHITE);
        assert_eq!(c.count_color(Color::WHITE), 25);
    }

    #[test]
    fn test_set_get_pixel() {
        let mut c = Canvas::new(10, 10);
        c.set_pixel(3, 4, Color::RED);
        assert_eq!(c.get_pixel(3, 4), Color::RED);
    }

    #[test]
    fn test_set_pixel_out_of_bounds() {
        let mut c = Canvas::new(5, 5);
        c.set_pixel(-1, 0, Color::RED);
        c.set_pixel(0, -1, Color::RED);
        c.set_pixel(5, 0, Color::RED);
        c.set_pixel(0, 5, Color::RED);
        // Nothing should crash; all pixels still transparent.
        assert_eq!(c.count_color(Color::TRANSPARENT), 25);
    }

    #[test]
    fn test_draw_hline() {
        let mut c = Canvas::new(10, 1);
        c.draw_hline(2, 5, 0, Color::RED);
        assert_eq!(c.get_pixel(2, 0), Color::RED);
        assert_eq!(c.get_pixel(5, 0), Color::RED);
        assert_eq!(c.get_pixel(1, 0), Color::TRANSPARENT);
    }

    #[test]
    fn test_draw_vline() {
        let mut c = Canvas::new(1, 10);
        c.draw_vline(0, 1, 4, Color::BLUE);
        assert_eq!(c.get_pixel(0, 1), Color::BLUE);
        assert_eq!(c.get_pixel(0, 4), Color::BLUE);
        assert_eq!(c.get_pixel(0, 0), Color::TRANSPARENT);
    }

    #[test]
    fn test_draw_line() {
        let mut c = Canvas::new(10, 10);
        c.draw_line(0, 0, 9, 9, Color::WHITE);
        assert_eq!(c.get_pixel(0, 0), Color::WHITE);
        assert_eq!(c.get_pixel(9, 9), Color::WHITE);
        // Diagonal: approximately 10 pixels lit.
        let lit = c.count_color(Color::WHITE);
        assert!(lit >= 10);
    }

    #[test]
    fn test_draw_rect() {
        let mut c = Canvas::new(10, 10);
        c.draw_rect(1, 1, 4, 4, Color::GREEN);
        assert_eq!(c.get_pixel(1, 1), Color::GREEN);
        assert_eq!(c.get_pixel(4, 1), Color::GREEN);
        assert_eq!(c.get_pixel(1, 4), Color::GREEN);
        assert_eq!(c.get_pixel(4, 4), Color::GREEN);
        // Interior should be transparent.
        assert_eq!(c.get_pixel(2, 2), Color::TRANSPARENT);
    }

    #[test]
    fn test_fill_rect() {
        let mut c = Canvas::new(10, 10);
        c.fill_rect(0, 0, 3, 3, Color::BLUE);
        assert_eq!(c.count_color(Color::BLUE), 9);
    }

    #[test]
    fn test_draw_circle() {
        let mut c = Canvas::new(20, 20);
        c.draw_circle(10, 10, 5, Color::RED);
        let lit = c.count_color(Color::RED);
        assert!(lit > 20, "circle should have visible pixels, got {lit}");
    }

    #[test]
    fn test_fill_circle() {
        let mut c = Canvas::new(20, 20);
        c.fill_circle(10, 10, 3, Color::GREEN);
        let filled = c.count_color(Color::GREEN);
        // pi * r^2 ~ 28.3, so expect roughly 25-35 pixels.
        assert!(filled >= 20 && filled <= 40, "got {filled}");
    }

    #[test]
    fn test_transform_translate() {
        let mut c = Canvas::new(10, 10);
        c.push_transform(Transform::translate(3, 3));
        c.set_pixel(0, 0, Color::RED);
        c.pop_transform();
        assert_eq!(c.get_pixel(3, 3), Color::RED);
    }

    #[test]
    fn test_transform_stack() {
        let mut c = Canvas::new(20, 20);
        c.push_transform(Transform::translate(5, 5));
        c.push_transform(Transform::translate(2, 2));
        c.set_pixel(0, 0, Color::BLUE);
        c.pop_transform();
        c.set_pixel(0, 0, Color::RED);
        c.pop_transform();

        assert_eq!(c.get_pixel(7, 7), Color::BLUE);
        assert_eq!(c.get_pixel(5, 5), Color::RED);
    }

    #[test]
    fn test_clip_rect() {
        let mut c = Canvas::new(10, 10);
        c.set_clip(ClipRect::new(2, 2, 4, 4));
        c.fill_rect(0, 0, 10, 10, Color::WHITE);
        // Only (2,2) to (5,5) should be filled.
        assert_eq!(c.get_pixel(0, 0), Color::TRANSPARENT);
        assert_eq!(c.get_pixel(2, 2), Color::WHITE);
        assert_eq!(c.get_pixel(5, 5), Color::WHITE);
        assert_eq!(c.get_pixel(6, 6), Color::TRANSPARENT);
        assert_eq!(c.count_color(Color::WHITE), 16);
    }

    #[test]
    fn test_clip_clear_clip() {
        let mut c = Canvas::new(5, 5);
        c.set_clip(ClipRect::new(0, 0, 1, 1));
        c.fill_rect(0, 0, 5, 5, Color::RED);
        assert_eq!(c.count_color(Color::RED), 1);
        c.clear_clip();
        c.fill_rect(0, 0, 5, 5, Color::GREEN);
        assert_eq!(c.count_color(Color::GREEN), 25);
    }

    #[test]
    fn test_flood_fill() {
        let mut c = Canvas::new(5, 5);
        c.clear(Color::WHITE);
        // Draw a border.
        c.draw_rect(1, 1, 3, 3, Color::BLACK);
        // Fill inside the border.
        c.flood_fill(2, 2, Color::RED);
        assert_eq!(c.get_pixel(2, 2), Color::RED);
        // The border should still be black.
        assert_eq!(c.get_pixel(1, 1), Color::BLACK);
        // Outside should still be white.
        assert_eq!(c.get_pixel(0, 0), Color::WHITE);
    }

    #[test]
    fn test_flood_fill_same_color() {
        let mut c = Canvas::new(3, 3);
        c.clear(Color::RED);
        c.flood_fill(1, 1, Color::RED);
        // No infinite loop; still all red.
        assert_eq!(c.count_color(Color::RED), 9);
    }

    #[test]
    fn test_clip_rect_contains() {
        let clip = ClipRect::new(5, 5, 10, 10);
        assert!(clip.contains(5, 5));
        assert!(clip.contains(14, 14));
        assert!(!clip.contains(4, 5));
        assert!(!clip.contains(15, 5));
    }

    #[test]
    fn test_transform_combine() {
        let a = Transform::translate(3, 4);
        let b = Transform::translate(1, 2);
        let c = a.combine(&b);
        assert_eq!(c.tx, 4);
        assert_eq!(c.ty, 6);
    }
}
