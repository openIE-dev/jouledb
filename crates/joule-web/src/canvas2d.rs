//! 2D Drawing API — pure-Rust command buffer that replaces Konva, fabric.js, p5.js.
//!
//! Records drawing commands into a buffer that can be replayed on any backend
//! (Canvas2D, SVG, software renderer). Includes a `to_svg()` method for
//! server-side rendering.

use std::fmt;

// ── Styles ─────────────────────────────────────────────────────

/// Fill style for shapes and text.
#[derive(Debug, Clone, PartialEq)]
pub enum FillStyle {
    Color(String),
    LinearGradient {
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        stops: Vec<(f64, String)>,
    },
    RadialGradient {
        cx: f64,
        cy: f64,
        r: f64,
        stops: Vec<(f64, String)>,
    },
}

/// Stroke style for outlines.
#[derive(Debug, Clone, PartialEq)]
pub enum StrokeStyle {
    Color(String),
    LinearGradient {
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        stops: Vec<(f64, String)>,
    },
    RadialGradient {
        cx: f64,
        cy: f64,
        r: f64,
        stops: Vec<(f64, String)>,
    },
}

/// Line cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineCap {
    Butt,
    Round,
    Square,
}

impl fmt::Display for LineCap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Butt => write!(f, "butt"),
            Self::Round => write!(f, "round"),
            Self::Square => write!(f, "square"),
        }
    }
}

/// Line join style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineJoin {
    Miter,
    Round,
    Bevel,
}

impl fmt::Display for LineJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Miter => write!(f, "miter"),
            Self::Round => write!(f, "round"),
            Self::Bevel => write!(f, "bevel"),
        }
    }
}

// ── Transform ──────────────────────────────────────────────────

/// 2D affine transformation matrix: [a c e; b d f; 0 0 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Transform2D {
    /// Identity matrix.
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Translation matrix.
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: tx,
            f: ty,
        }
    }

    /// Rotation matrix (angle in radians).
    pub fn rotation(angle: f64) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            a: cos,
            b: sin,
            c: -sin,
            d: cos,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Scaling matrix.
    pub fn scaling(sx: f64, sy: f64) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Multiply two transforms: self * other.
    pub fn multiply(&self, other: &Transform2D) -> Transform2D {
        Transform2D {
            a: self.a * other.a + self.c * other.b,
            b: self.b * other.a + self.d * other.b,
            c: self.a * other.c + self.c * other.d,
            d: self.b * other.c + self.d * other.d,
            e: self.a * other.e + self.c * other.f + self.e,
            f: self.b * other.e + self.d * other.f + self.f,
        }
    }

    /// Transform a point (x, y).
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }

    /// Compute the inverse, if the matrix is non-singular.
    pub fn inverse(&self) -> Option<Transform2D> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Transform2D {
            a: self.d * inv_det,
            b: -self.b * inv_det,
            c: -self.c * inv_det,
            d: self.a * inv_det,
            e: (self.c * self.f - self.d * self.e) * inv_det,
            f: (self.b * self.e - self.a * self.f) * inv_det,
        })
    }
}

// ── Draw commands ──────────────────────────────────────────────

/// A single drawing command in the command buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum DrawCommand {
    MoveTo(f64, f64),
    LineTo(f64, f64),
    QuadraticTo { cx: f64, cy: f64, x: f64, y: f64 },
    BezierTo { cx1: f64, cy1: f64, cx2: f64, cy2: f64, x: f64, y: f64 },
    ArcTo { x: f64, y: f64, radius: f64, start_angle: f64, end_angle: f64 },
    ClosePath,
    Rect { x: f64, y: f64, w: f64, h: f64 },
    Circle { cx: f64, cy: f64, radius: f64 },
    Ellipse { cx: f64, cy: f64, rx: f64, ry: f64 },
    FillStyle(FillStyle),
    StrokeStyle(StrokeStyle),
    LineWidth(f64),
    LineCap(LineCap),
    LineJoin(LineJoin),
    Fill,
    Stroke,
    FillAndStroke,
    Save,
    Restore,
    Translate(f64, f64),
    Rotate(f64),
    Scale(f64, f64),
    SetTransform(Transform2D),
    FillText { text: String, x: f64, y: f64, font: Option<String>, max_width: Option<f64> },
    StrokeText { text: String, x: f64, y: f64, font: Option<String> },
    Clip,
    GlobalAlpha(f64),
    ClearRect { x: f64, y: f64, w: f64, h: f64 },
    BeginPath,
}

// ── Canvas2D ───────────────────────────────────────────────────

/// A 2D drawing command buffer.
pub struct Canvas2D {
    commands: Vec<DrawCommand>,
    width: f64,
    height: f64,
}

impl Canvas2D {
    /// Create a new canvas with the given dimensions.
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            commands: Vec::new(),
            width,
            height,
        }
    }

    // ── Fluent path commands ──

    pub fn begin_path(&mut self) -> &mut Self {
        self.commands.push(DrawCommand::BeginPath);
        self
    }

    pub fn move_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.commands.push(DrawCommand::MoveTo(x, y));
        self
    }

    pub fn line_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.commands.push(DrawCommand::LineTo(x, y));
        self
    }

    pub fn quadratic_to(&mut self, cx: f64, cy: f64, x: f64, y: f64) -> &mut Self {
        self.commands.push(DrawCommand::QuadraticTo { cx, cy, x, y });
        self
    }

    pub fn bezier_to(&mut self, cx1: f64, cy1: f64, cx2: f64, cy2: f64, x: f64, y: f64) -> &mut Self {
        self.commands.push(DrawCommand::BezierTo { cx1, cy1, cx2, cy2, x, y });
        self
    }

    pub fn arc_to(&mut self, x: f64, y: f64, radius: f64, start_angle: f64, end_angle: f64) -> &mut Self {
        self.commands.push(DrawCommand::ArcTo { x, y, radius, start_angle, end_angle });
        self
    }

    pub fn close_path(&mut self) -> &mut Self {
        self.commands.push(DrawCommand::ClosePath);
        self
    }

    // ── Shape shortcuts ──

    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64) -> &mut Self {
        self.commands.push(DrawCommand::Rect { x, y, w, h });
        self
    }

    pub fn circle(&mut self, cx: f64, cy: f64, radius: f64) -> &mut Self {
        self.commands.push(DrawCommand::Circle { cx, cy, radius });
        self
    }

    pub fn ellipse(&mut self, cx: f64, cy: f64, rx: f64, ry: f64) -> &mut Self {
        self.commands.push(DrawCommand::Ellipse { cx, cy, rx, ry });
        self
    }

    // ── Style setters ──

    pub fn fill_style(&mut self, style: FillStyle) -> &mut Self {
        self.commands.push(DrawCommand::FillStyle(style));
        self
    }

    pub fn stroke_style(&mut self, style: StrokeStyle) -> &mut Self {
        self.commands.push(DrawCommand::StrokeStyle(style));
        self
    }

    pub fn line_width(&mut self, w: f64) -> &mut Self {
        self.commands.push(DrawCommand::LineWidth(w));
        self
    }

    pub fn line_cap(&mut self, cap: LineCap) -> &mut Self {
        self.commands.push(DrawCommand::LineCap(cap));
        self
    }

    pub fn line_join(&mut self, join: LineJoin) -> &mut Self {
        self.commands.push(DrawCommand::LineJoin(join));
        self
    }

    pub fn global_alpha(&mut self, alpha: f64) -> &mut Self {
        self.commands.push(DrawCommand::GlobalAlpha(alpha));
        self
    }

    // ── Drawing operations ──

    pub fn fill(&mut self) -> &mut Self {
        self.commands.push(DrawCommand::Fill);
        self
    }

    pub fn stroke(&mut self) -> &mut Self {
        self.commands.push(DrawCommand::Stroke);
        self
    }

    pub fn fill_and_stroke(&mut self) -> &mut Self {
        self.commands.push(DrawCommand::FillAndStroke);
        self
    }

    pub fn clip(&mut self) -> &mut Self {
        self.commands.push(DrawCommand::Clip);
        self
    }

    pub fn clear_rect(&mut self, x: f64, y: f64, w: f64, h: f64) -> &mut Self {
        self.commands.push(DrawCommand::ClearRect { x, y, w, h });
        self
    }

    // ── Transform operations ──

    pub fn save(&mut self) -> &mut Self {
        self.commands.push(DrawCommand::Save);
        self
    }

    pub fn restore(&mut self) -> &mut Self {
        self.commands.push(DrawCommand::Restore);
        self
    }

    pub fn translate(&mut self, x: f64, y: f64) -> &mut Self {
        self.commands.push(DrawCommand::Translate(x, y));
        self
    }

    pub fn rotate(&mut self, angle: f64) -> &mut Self {
        self.commands.push(DrawCommand::Rotate(angle));
        self
    }

    pub fn scale(&mut self, sx: f64, sy: f64) -> &mut Self {
        self.commands.push(DrawCommand::Scale(sx, sy));
        self
    }

    pub fn set_transform(&mut self, t: Transform2D) -> &mut Self {
        self.commands.push(DrawCommand::SetTransform(t));
        self
    }

    // ── Text ──

    pub fn fill_text(&mut self, text: &str, x: f64, y: f64) -> &mut Self {
        self.commands.push(DrawCommand::FillText {
            text: text.to_string(),
            x,
            y,
            font: None,
            max_width: None,
        });
        self
    }

    pub fn stroke_text(&mut self, text: &str, x: f64, y: f64) -> &mut Self {
        self.commands.push(DrawCommand::StrokeText {
            text: text.to_string(),
            x,
            y,
            font: None,
        });
        self
    }

    // ── Queries ──

    /// Clear all commands.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// View the command buffer.
    pub fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    /// Number of commands recorded.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    // ── SVG export ─────────────────────────────────────────────

    /// Render the command buffer to an SVG string.
    pub fn to_svg(&self) -> String {
        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\n",
            self.width, self.height
        );

        let mut path_data = String::new();
        let mut fill_color = String::from("black");
        let mut stroke_color = String::from("none");
        let mut stroke_width = 1.0_f64;
        let mut gradient_defs: Vec<String> = Vec::new();
        let mut gradient_id = 0usize;

        for cmd in &self.commands {
            match cmd {
                DrawCommand::BeginPath => {
                    path_data.clear();
                }
                DrawCommand::MoveTo(x, y) => {
                    path_data.push_str(&format!("M{x} {y} "));
                }
                DrawCommand::LineTo(x, y) => {
                    path_data.push_str(&format!("L{x} {y} "));
                }
                DrawCommand::QuadraticTo { cx, cy, x, y } => {
                    path_data.push_str(&format!("Q{cx} {cy} {x} {y} "));
                }
                DrawCommand::BezierTo { cx1, cy1, cx2, cy2, x, y } => {
                    path_data.push_str(&format!("C{cx1} {cy1} {cx2} {cy2} {x} {y} "));
                }
                DrawCommand::ClosePath => {
                    path_data.push_str("Z ");
                }
                DrawCommand::Rect { x, y, w, h } => {
                    svg.push_str(&format!(
                        "  <rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" fill=\"{fill_color}\" stroke=\"{stroke_color}\" stroke-width=\"{stroke_width}\"/>\n"
                    ));
                }
                DrawCommand::Circle { cx, cy, radius } => {
                    svg.push_str(&format!(
                        "  <circle cx=\"{cx}\" cy=\"{cy}\" r=\"{radius}\" fill=\"{fill_color}\" stroke=\"{stroke_color}\" stroke-width=\"{stroke_width}\"/>\n"
                    ));
                }
                DrawCommand::Ellipse { cx, cy, rx, ry } => {
                    svg.push_str(&format!(
                        "  <ellipse cx=\"{cx}\" cy=\"{cy}\" rx=\"{rx}\" ry=\"{ry}\" fill=\"{fill_color}\" stroke=\"{stroke_color}\" stroke-width=\"{stroke_width}\"/>\n"
                    ));
                }
                DrawCommand::FillStyle(style) => match style {
                    FillStyle::Color(c) => fill_color = c.clone(),
                    FillStyle::LinearGradient { x0, y0, x1, y1, stops } => {
                        let gid = format!("lg{gradient_id}");
                        gradient_id += 1;
                        let mut def = format!(
                            "<linearGradient id=\"{gid}\" x1=\"{x0}\" y1=\"{y0}\" x2=\"{x1}\" y2=\"{y1}\" gradientUnits=\"userSpaceOnUse\">"
                        );
                        for (offset, color) in stops {
                            def.push_str(&format!(
                                "<stop offset=\"{offset}\" stop-color=\"{color}\"/>"
                            ));
                        }
                        def.push_str("</linearGradient>");
                        gradient_defs.push(def);
                        fill_color = format!("url(#{gid})");
                    }
                    FillStyle::RadialGradient { cx, cy, r, stops } => {
                        let gid = format!("rg{gradient_id}");
                        gradient_id += 1;
                        let mut def = format!(
                            "<radialGradient id=\"{gid}\" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" gradientUnits=\"userSpaceOnUse\">"
                        );
                        for (offset, color) in stops {
                            def.push_str(&format!(
                                "<stop offset=\"{offset}\" stop-color=\"{color}\"/>"
                            ));
                        }
                        def.push_str("</radialGradient>");
                        gradient_defs.push(def);
                        fill_color = format!("url(#{gid})");
                    }
                },
                DrawCommand::StrokeStyle(style) => match style {
                    StrokeStyle::Color(c) => stroke_color = c.clone(),
                    StrokeStyle::LinearGradient { x0, y0, x1, y1, stops } => {
                        let gid = format!("slg{gradient_id}");
                        gradient_id += 1;
                        let mut def = format!(
                            "<linearGradient id=\"{gid}\" x1=\"{x0}\" y1=\"{y0}\" x2=\"{x1}\" y2=\"{y1}\" gradientUnits=\"userSpaceOnUse\">"
                        );
                        for (offset, color) in stops {
                            def.push_str(&format!(
                                "<stop offset=\"{offset}\" stop-color=\"{color}\"/>"
                            ));
                        }
                        def.push_str("</linearGradient>");
                        gradient_defs.push(def);
                        stroke_color = format!("url(#{gid})");
                    }
                    StrokeStyle::RadialGradient { cx, cy, r, stops } => {
                        let gid = format!("srg{gradient_id}");
                        gradient_id += 1;
                        let mut def = format!(
                            "<radialGradient id=\"{gid}\" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" gradientUnits=\"userSpaceOnUse\">"
                        );
                        for (offset, color) in stops {
                            def.push_str(&format!(
                                "<stop offset=\"{offset}\" stop-color=\"{color}\"/>"
                            ));
                        }
                        def.push_str("</radialGradient>");
                        gradient_defs.push(def);
                        stroke_color = format!("url(#{gid})");
                    }
                },
                DrawCommand::LineWidth(w) => stroke_width = *w,
                DrawCommand::FillText { text, x, y, font, .. } => {
                    let font_attr = font.as_ref().map_or(String::new(), |f| format!(" font-family=\"{f}\""));
                    svg.push_str(&format!(
                        "  <text x=\"{x}\" y=\"{y}\" fill=\"{fill_color}\"{font_attr}>{text}</text>\n"
                    ));
                }
                DrawCommand::StrokeText { text, x, y, font } => {
                    let font_attr = font.as_ref().map_or(String::new(), |f| format!(" font-family=\"{f}\""));
                    svg.push_str(&format!(
                        "  <text x=\"{x}\" y=\"{y}\" stroke=\"{stroke_color}\" fill=\"none\"{font_attr}>{text}</text>\n"
                    ));
                }
                DrawCommand::Fill | DrawCommand::Stroke | DrawCommand::FillAndStroke => {
                    if !path_data.is_empty() {
                        let fill = if matches!(cmd, DrawCommand::Stroke) { "none" } else { &fill_color };
                        let stroke_attr = if matches!(cmd, DrawCommand::Fill) { "none" } else { &stroke_color };
                        svg.push_str(&format!(
                            "  <path d=\"{}\" fill=\"{fill}\" stroke=\"{stroke_attr}\" stroke-width=\"{stroke_width}\"/>\n",
                            path_data.trim()
                        ));
                    }
                }
                // Commands that don't map directly to SVG are skipped.
                _ => {}
            }
        }

        // Insert gradient defs if any.
        if !gradient_defs.is_empty() {
            let mut final_svg = format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\n  <defs>\n",
                self.width, self.height
            );
            for def in &gradient_defs {
                final_svg.push_str(&format!("    {def}\n"));
            }
            final_svg.push_str("  </defs>\n");
            // Remove the opening svg tag from the original and append the rest.
            let body_start = svg.find('\n').map_or(0, |i| i + 1);
            final_svg.push_str(&svg[body_start..]);
            final_svg.push_str("</svg>\n");
            return final_svg;
        }

        svg.push_str("</svg>\n");
        svg
    }
}

// ── Path2D ─────────────────────────────────────────────────────

/// A reusable 2D path.
pub struct Path2D {
    segments: Vec<DrawCommand>,
}

impl Path2D {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn move_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.segments.push(DrawCommand::MoveTo(x, y));
        self
    }

    pub fn line_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.segments.push(DrawCommand::LineTo(x, y));
        self
    }

    pub fn bezier_to(&mut self, cx1: f64, cy1: f64, cx2: f64, cy2: f64, x: f64, y: f64) -> &mut Self {
        self.segments.push(DrawCommand::BezierTo { cx1, cy1, cx2, cy2, x, y });
        self
    }

    pub fn close(&mut self) -> &mut Self {
        self.segments.push(DrawCommand::ClosePath);
        self
    }

    /// Bounding box: (min_x, min_y, max_x, max_y).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for seg in &self.segments {
            let points: Vec<(f64, f64)> = match seg {
                DrawCommand::MoveTo(x, y) | DrawCommand::LineTo(x, y) => vec![(*x, *y)],
                DrawCommand::BezierTo { cx1, cy1, cx2, cy2, x, y } => {
                    vec![(*cx1, *cy1), (*cx2, *cy2), (*x, *y)]
                }
                DrawCommand::QuadraticTo { cx, cy, x, y } => {
                    vec![(*cx, *cy), (*x, *y)]
                }
                _ => vec![],
            };
            for (px, py) in points {
                if px < min_x { min_x = px; }
                if py < min_y { min_y = py; }
                if px > max_x { max_x = px; }
                if py > max_y { max_y = py; }
            }
        }

        if min_x == f64::INFINITY {
            return (0.0, 0.0, 0.0, 0.0);
        }
        (min_x, min_y, max_x, max_y)
    }

    /// Approximate path length (sum of segment lengths).
    pub fn length(&self) -> f64 {
        let mut total = 0.0;
        let mut cur_x = 0.0;
        let mut cur_y = 0.0;

        for seg in &self.segments {
            match seg {
                DrawCommand::MoveTo(x, y) => {
                    cur_x = *x;
                    cur_y = *y;
                }
                DrawCommand::LineTo(x, y) => {
                    let dx = *x - cur_x;
                    let dy = *y - cur_y;
                    total += (dx * dx + dy * dy).sqrt();
                    cur_x = *x;
                    cur_y = *y;
                }
                DrawCommand::BezierTo { cx1, cy1, cx2, cy2, x, y } => {
                    // Approximate with chord + control polygon average.
                    let chord = ((*x - cur_x).powi(2) + (*y - cur_y).powi(2)).sqrt();
                    let poly = ((cx1 - cur_x).powi(2) + (cy1 - cur_y).powi(2)).sqrt()
                        + ((cx2 - cx1).powi(2) + (cy2 - cy1).powi(2)).sqrt()
                        + ((x - cx2).powi(2) + (y - cy2).powi(2)).sqrt();
                    total += (chord + poly) / 2.0;
                    cur_x = *x;
                    cur_y = *y;
                }
                DrawCommand::QuadraticTo { cx, cy, x, y } => {
                    let chord = ((*x - cur_x).powi(2) + (*y - cur_y).powi(2)).sqrt();
                    let poly = ((cx - cur_x).powi(2) + (cy - cur_y).powi(2)).sqrt()
                        + ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                    total += (chord + poly) / 2.0;
                    cur_x = *x;
                    cur_y = *y;
                }
                _ => {}
            }
        }

        total
    }

    /// Access the segment commands.
    pub fn segments(&self) -> &[DrawCommand] {
        &self.segments
    }
}

impl Default for Path2D {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_buffer_records() {
        let mut c = Canvas2D::new(800.0, 600.0);
        c.begin_path().move_to(10.0, 20.0).line_to(30.0, 40.0).fill();
        assert_eq!(c.command_count(), 4);
        assert!(matches!(c.commands()[0], DrawCommand::BeginPath));
        assert!(matches!(c.commands()[1], DrawCommand::MoveTo(10.0, 20.0)));
    }

    #[test]
    fn svg_output_for_rect() {
        let mut c = Canvas2D::new(100.0, 100.0);
        c.rect(10.0, 20.0, 30.0, 40.0);
        let svg = c.to_svg();
        assert!(svg.contains("<rect"));
        assert!(svg.contains("x=\"10\""));
        assert!(svg.contains("width=\"30\""));
    }

    #[test]
    fn circle_to_svg() {
        let mut c = Canvas2D::new(200.0, 200.0);
        c.circle(100.0, 100.0, 50.0);
        let svg = c.to_svg();
        assert!(svg.contains("<circle"));
        assert!(svg.contains("r=\"50\""));
    }

    #[test]
    fn transform_multiply() {
        let t = Transform2D::translation(10.0, 20.0);
        let s = Transform2D::scaling(2.0, 3.0);
        let ts = t.multiply(&s);
        // Translation applied first, then scale: result should scale then translate.
        let (px, py) = ts.transform_point(1.0, 1.0);
        assert!((px - 12.0).abs() < 1e-10);
        assert!((py - 23.0).abs() < 1e-10);
    }

    #[test]
    fn transform_point() {
        let t = Transform2D::translation(5.0, 10.0);
        let (px, py) = t.transform_point(3.0, 7.0);
        assert!((px - 8.0).abs() < 1e-10);
        assert!((py - 17.0).abs() < 1e-10);
    }

    #[test]
    fn identity_inverse() {
        let id = Transform2D::identity();
        let inv = id.inverse().unwrap();
        assert!((inv.a - 1.0).abs() < 1e-10);
        assert!((inv.d - 1.0).abs() < 1e-10);
        assert!((inv.e).abs() < 1e-10);
    }

    #[test]
    fn path_bounds() {
        let mut p = Path2D::new();
        p.move_to(0.0, 0.0).line_to(10.0, 20.0).line_to(5.0, 15.0);
        let (min_x, min_y, max_x, max_y) = p.bounds();
        assert!((min_x).abs() < 1e-10);
        assert!((min_y).abs() < 1e-10);
        assert!((max_x - 10.0).abs() < 1e-10);
        assert!((max_y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn gradient_fill() {
        let mut c = Canvas2D::new(100.0, 100.0);
        c.fill_style(FillStyle::LinearGradient {
            x0: 0.0, y0: 0.0, x1: 100.0, y1: 0.0,
            stops: vec![(0.0, "red".into()), (1.0, "blue".into())],
        });
        c.rect(0.0, 0.0, 100.0, 100.0);
        let svg = c.to_svg();
        assert!(svg.contains("linearGradient"));
        assert!(svg.contains("red"));
        assert!(svg.contains("blue"));
    }

    #[test]
    fn save_restore_stack() {
        let mut c = Canvas2D::new(100.0, 100.0);
        c.save().translate(10.0, 10.0).restore();
        assert_eq!(c.command_count(), 3);
        assert!(matches!(c.commands()[0], DrawCommand::Save));
        assert!(matches!(c.commands()[2], DrawCommand::Restore));
    }

    #[test]
    fn clear_resets() {
        let mut c = Canvas2D::new(100.0, 100.0);
        c.rect(0.0, 0.0, 50.0, 50.0);
        assert_eq!(c.command_count(), 1);
        c.clear();
        assert_eq!(c.command_count(), 0);
    }

    #[test]
    fn linear_gradient_svg_has_defs() {
        let mut c = Canvas2D::new(200.0, 200.0);
        c.fill_style(FillStyle::LinearGradient {
            x0: 0.0, y0: 0.0, x1: 200.0, y1: 0.0,
            stops: vec![(0.0, "#000".into()), (1.0, "#fff".into())],
        });
        c.rect(0.0, 0.0, 200.0, 200.0);
        let svg = c.to_svg();
        assert!(svg.contains("<defs>"));
        assert!(svg.contains("url(#lg0)"));
    }

    #[test]
    fn path_length_approximation() {
        let mut p = Path2D::new();
        p.move_to(0.0, 0.0).line_to(3.0, 4.0);
        let len = p.length();
        assert!((len - 5.0).abs() < 1e-10);
    }

    #[test]
    fn transform_inverse_roundtrip() {
        let t = Transform2D::translation(7.0, 13.0)
            .multiply(&Transform2D::rotation(0.5));
        let inv = t.inverse().unwrap();
        let id = t.multiply(&inv);
        assert!((id.a - 1.0).abs() < 1e-8);
        assert!((id.d - 1.0).abs() < 1e-8);
        assert!((id.e).abs() < 1e-8);
        assert!((id.f).abs() < 1e-8);
    }
}
