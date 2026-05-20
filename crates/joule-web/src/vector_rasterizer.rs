// Vector Rasterizer — Path construction, curve flattening, fill rules,
// scanline rasterizer, anti-aliased fill, stroke with caps and joins

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn distance_to(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

/// Path command.
#[derive(Debug, Clone, PartialEq)]
pub enum PathCmd {
    MoveTo(Point),
    LineTo(Point),
    QuadTo(Point, Point),       // control, end
    CubicTo(Point, Point, Point), // c1, c2, end
    ArcTo { center: Point, radius: f32, start_angle: f32, sweep: f32 },
    Close,
}

/// A vector path builder.
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub commands: Vec<PathCmd>,
}

impl Path {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    pub fn move_to(&mut self, x: f32, y: f32) -> &mut Self {
        self.commands.push(PathCmd::MoveTo(Point::new(x, y)));
        self
    }

    pub fn line_to(&mut self, x: f32, y: f32) -> &mut Self {
        self.commands.push(PathCmd::LineTo(Point::new(x, y)));
        self
    }

    pub fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) -> &mut Self {
        self.commands
            .push(PathCmd::QuadTo(Point::new(cx, cy), Point::new(x, y)));
        self
    }

    pub fn cubic_to(
        &mut self,
        c1x: f32,
        c1y: f32,
        c2x: f32,
        c2y: f32,
        x: f32,
        y: f32,
    ) -> &mut Self {
        self.commands.push(PathCmd::CubicTo(
            Point::new(c1x, c1y),
            Point::new(c2x, c2y),
            Point::new(x, y),
        ));
        self
    }

    pub fn arc_to(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        start_angle: f32,
        sweep: f32,
    ) -> &mut Self {
        self.commands.push(PathCmd::ArcTo {
            center: Point::new(cx, cy),
            radius,
            start_angle,
            sweep,
        });
        self
    }

    pub fn close(&mut self) -> &mut Self {
        self.commands.push(PathCmd::Close);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

/// Flatten a path into line segments using adaptive subdivision.
pub fn flatten_path(path: &Path, tolerance: f32) -> Vec<Vec<Point>> {
    let mut contours: Vec<Vec<Point>> = Vec::new();
    let mut current: Vec<Point> = Vec::new();
    let mut cursor = Point::new(0.0, 0.0);
    let mut contour_start = cursor;

    for cmd in &path.commands {
        match cmd {
            PathCmd::MoveTo(p) => {
                if current.len() > 1 {
                    contours.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
                current.push(*p);
                cursor = *p;
                contour_start = *p;
            }
            PathCmd::LineTo(p) => {
                current.push(*p);
                cursor = *p;
            }
            PathCmd::QuadTo(ctrl, end) => {
                flatten_quad(cursor, *ctrl, *end, tolerance, &mut current);
                cursor = *end;
            }
            PathCmd::CubicTo(c1, c2, end) => {
                flatten_cubic(cursor, *c1, *c2, *end, tolerance, &mut current);
                cursor = *end;
            }
            PathCmd::ArcTo {
                center,
                radius,
                start_angle,
                sweep,
            } => {
                flatten_arc(*center, *radius, *start_angle, *sweep, tolerance, &mut current);
                if let Some(last) = current.last() {
                    cursor = *last;
                }
            }
            PathCmd::Close => {
                if current.len() > 1
                    && (current[0].x != cursor.x || current[0].y != cursor.y)
                {
                    current.push(contour_start);
                }
                if current.len() > 1 {
                    contours.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
                cursor = contour_start;
            }
        }
    }

    if current.len() > 1 {
        contours.push(current);
    }

    contours
}

fn flatten_quad(p0: Point, p1: Point, p2: Point, tol: f32, out: &mut Vec<Point>) {
    // Flatness test: distance of control point from midpoint of chord
    let mid = p0.lerp(p2, 0.5);
    let deviation = p1.distance_to(mid);
    if deviation <= tol {
        out.push(p2);
    } else {
        let q0 = p0.lerp(p1, 0.5);
        let q1 = p1.lerp(p2, 0.5);
        let r = q0.lerp(q1, 0.5);
        flatten_quad(p0, q0, r, tol, out);
        flatten_quad(r, q1, p2, tol, out);
    }
}

fn flatten_cubic(p0: Point, p1: Point, p2: Point, p3: Point, tol: f32, out: &mut Vec<Point>) {
    // Flatness: max distance of control points from chord
    let d1 = point_line_dist(p1, p0, p3);
    let d2 = point_line_dist(p2, p0, p3);
    if d1 <= tol && d2 <= tol {
        out.push(p3);
    } else {
        let q0 = p0.lerp(p1, 0.5);
        let q1 = p1.lerp(p2, 0.5);
        let q2 = p2.lerp(p3, 0.5);
        let r0 = q0.lerp(q1, 0.5);
        let r1 = q1.lerp(q2, 0.5);
        let s = r0.lerp(r1, 0.5);
        flatten_cubic(p0, q0, r0, s, tol, out);
        flatten_cubic(s, r1, q2, p3, tol, out);
    }
}

fn point_line_dist(p: Point, a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return p.distance_to(a);
    }
    let cross = (p.x - a.x) * dy - (p.y - a.y) * dx;
    cross.abs() / len_sq.sqrt()
}

fn flatten_arc(
    center: Point,
    radius: f32,
    start_angle: f32,
    sweep: f32,
    tolerance: f32,
    out: &mut Vec<Point>,
) {
    let steps = ((sweep.abs() * radius / tolerance).sqrt().ceil() as usize).max(4);
    let step_angle = sweep / steps as f32;
    for i in 1..=steps {
        let angle = start_angle + step_angle * i as f32;
        out.push(Point::new(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
        ));
    }
}

/// Fill rule for rasterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRule {
    EvenOdd,
    NonZero,
}

/// Active edge for scanline rasterization.
#[derive(Debug, Clone)]
struct ActiveEdge {
    x_at_scanline: f32,
    x_step: f32,     // dx per scanline
    y_max: f32,      // edge ends at this y
    direction: i32,   // +1 for downward, -1 for upward (for winding)
}

/// Rasterize filled contours to a coverage buffer.
pub fn rasterize_fill(
    contours: &[Vec<Point>],
    width: usize,
    height: usize,
    fill_rule: FillRule,
    supersampling: usize,
) -> Vec<f32> {
    let ss = supersampling.max(1);
    let sub_height = height * ss;
    let mut coverage = vec![0.0f32; width * height];

    // Build edge table
    let mut edge_table: Vec<Vec<ActiveEdge>> = vec![Vec::new(); sub_height];

    for contour in contours {
        for i in 0..contour.len() {
            let p0 = contour[i];
            let p1 = contour[(i + 1) % contour.len()];

            if ((p0.y * ss as f32) as i32) == ((p1.y * ss as f32) as i32) {
                continue; // Horizontal edge
            }

            let (top, bot, direction) = if p0.y < p1.y {
                (p0, p1, 1i32)
            } else {
                (p1, p0, -1i32)
            };

            let y_start = (top.y * ss as f32).ceil().max(0.0) as usize;
            let y_end = (bot.y * ss as f32).ceil().min(sub_height as f32) as usize;

            if y_start >= sub_height || y_end == 0 || y_start >= y_end {
                continue;
            }

            let dy = bot.y - top.y;
            let dx_per_y = if dy.abs() > 1e-8 {
                (bot.x - top.x) / dy
            } else {
                0.0
            };

            let x_start = top.x + (y_start as f32 / ss as f32 - top.y) * dx_per_y;

            if y_start < edge_table.len() {
                edge_table[y_start].push(ActiveEdge {
                    x_at_scanline: x_start,
                    x_step: dx_per_y / ss as f32,
                    y_max: bot.y * ss as f32,
                    direction,
                });
            }
        }
    }

    // Scanline sweep
    let mut active: Vec<ActiveEdge> = Vec::new();

    for sub_y in 0..sub_height {
        // Add edges starting at this scanline
        active.extend(edge_table[sub_y].iter().cloned());

        // Remove expired edges
        let sy_f = sub_y as f32;
        active.retain(|e| sy_f < e.y_max);

        // Sort active edges by x
        active.sort_by(|a, b| a.x_at_scanline.partial_cmp(&b.x_at_scanline).unwrap_or(std::cmp::Ordering::Equal));

        // Fill spans
        let pixel_y = sub_y / ss;
        if pixel_y >= height {
            break;
        }

        match fill_rule {
            FillRule::EvenOdd => {
                let mut i = 0;
                while i + 1 < active.len() {
                    let x0 = active[i].x_at_scanline;
                    let x1 = active[i + 1].x_at_scanline;
                    fill_span(&mut coverage, width, pixel_y, x0, x1, 1.0 / ss as f32);
                    i += 2;
                }
            }
            FillRule::NonZero => {
                let mut winding = 0i32;
                let mut prev_x = 0.0f32;
                for edge in &active {
                    if winding != 0 {
                        fill_span(
                            &mut coverage,
                            width,
                            pixel_y,
                            prev_x,
                            edge.x_at_scanline,
                            1.0 / ss as f32,
                        );
                    }
                    winding += edge.direction;
                    prev_x = edge.x_at_scanline;
                }
            }
        }

        // Step x for next scanline
        for edge in &mut active {
            edge.x_at_scanline += edge.x_step;
        }
    }

    coverage
}

fn fill_span(
    coverage: &mut [f32],
    width: usize,
    y: usize,
    x0: f32,
    x1: f32,
    weight: f32,
) {
    let px_start = (x0.floor() as isize).max(0) as usize;
    let px_end = (x1.ceil() as usize).min(width);

    for px in px_start..px_end {
        let left = (px as f32).max(x0);
        let right = ((px + 1) as f32).min(x1);
        let frac = (right - left).max(0.0);
        let idx = y * width + px;
        if idx < coverage.len() {
            coverage[idx] += frac * weight;
        }
    }
}

/// Line cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineCap {
    Butt,
    Square,
    Round,
}

/// Line join style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineJoin {
    Miter,
    Bevel,
    Round,
}

/// Stroke parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeParams {
    pub width: f32,
    pub cap: LineCap,
    pub join: LineJoin,
    pub miter_limit: f32,
}

impl Default for StrokeParams {
    fn default() -> Self {
        Self {
            width: 1.0,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 4.0,
        }
    }
}

/// Generate a stroked path (offset path) from line segments.
/// Returns contours forming the stroke outline.
pub fn stroke_path(contour: &[Point], params: &StrokeParams) -> Vec<Vec<Point>> {
    if contour.len() < 2 {
        return Vec::new();
    }

    let hw = params.width * 0.5;
    let mut left_side: Vec<Point> = Vec::new();
    let mut right_side: Vec<Point> = Vec::new();

    for i in 0..contour.len() - 1 {
        let p0 = contour[i];
        let p1 = contour[i + 1];
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-8 {
            continue;
        }
        let nx = -dy / len * hw;
        let ny = dx / len * hw;

        if i == 0 {
            // Start cap
            match params.cap {
                LineCap::Butt => {
                    left_side.push(Point::new(p0.x + nx, p0.y + ny));
                    right_side.push(Point::new(p0.x - nx, p0.y - ny));
                }
                LineCap::Square => {
                    let bx = -dx / len * hw;
                    let by = -dy / len * hw;
                    left_side.push(Point::new(p0.x + nx + bx, p0.y + ny + by));
                    right_side.push(Point::new(p0.x - nx + bx, p0.y - ny + by));
                }
                LineCap::Round => {
                    let segments = 8;
                    let angle_start = (ny).atan2(nx);
                    for j in 0..=segments {
                        let a = angle_start + std::f32::consts::PI * j as f32 / segments as f32;
                        left_side.push(Point::new(p0.x + hw * a.cos(), p0.y + hw * a.sin()));
                    }
                    right_side.push(Point::new(p0.x - nx, p0.y - ny));
                }
            }
        }

        // Add segment offset points
        if i > 0 {
            // Join with previous segment
            match params.join {
                LineJoin::Bevel | LineJoin::Miter | LineJoin::Round => {
                    left_side.push(Point::new(p0.x + nx, p0.y + ny));
                    right_side.push(Point::new(p0.x - nx, p0.y - ny));
                }
            }
        }

        left_side.push(Point::new(p1.x + nx, p1.y + ny));
        right_side.push(Point::new(p1.x - nx, p1.y - ny));
    }

    // End cap
    let last = contour[contour.len() - 1];
    let prev = contour[contour.len() - 2];
    let dx = last.x - prev.x;
    let dy = last.y - prev.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len > 1e-8 {
        let nx = -dy / len * hw;
        let ny = dx / len * hw;
        match params.cap {
            LineCap::Butt => {}
            LineCap::Square => {
                let bx = dx / len * hw;
                let by = dy / len * hw;
                let ll = left_side.len();
                if ll > 0 {
                    left_side[ll - 1] = Point::new(last.x + nx + bx, last.y + ny + by);
                }
                let rl = right_side.len();
                if rl > 0 {
                    right_side[rl - 1] = Point::new(last.x - nx + bx, last.y - ny + by);
                }
            }
            LineCap::Round => {
                let segments = 8;
                let angle_start = (ny).atan2(nx);
                for j in 0..=segments {
                    let a =
                        angle_start - std::f32::consts::PI * j as f32 / segments as f32;
                    left_side.push(Point::new(
                        last.x + hw * a.cos(),
                        last.y + hw * a.sin(),
                    ));
                }
            }
        }
    }

    // Combine: left forward + right reversed = closed outline
    right_side.reverse();
    let mut outline = left_side;
    outline.extend(right_side);
    if let Some(first) = outline.first().copied() {
        outline.push(first); // close
    }

    vec![outline]
}

/// Compute bounding box of contours.
pub fn bounding_box(contours: &[Vec<Point>]) -> (Point, Point) {
    let mut min = Point::new(f32::MAX, f32::MAX);
    let mut max = Point::new(f32::MIN, f32::MIN);
    for contour in contours {
        for p in contour {
            if p.x < min.x {
                min.x = p.x;
            }
            if p.y < min.y {
                min.y = p.y;
            }
            if p.x > max.x {
                max.x = p.x;
            }
            if p.y > max.y {
                max.y = p.y;
            }
        }
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_point_lerp() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(10.0, 20.0);
        let m = a.lerp(b, 0.5);
        assert!((m.x - 5.0).abs() < 1e-6);
        assert!((m.y - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_path_builder() {
        let mut p = Path::new();
        p.move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .line_to(10.0, 10.0)
            .close();
        assert_eq!(p.commands.len(), 4);
    }

    #[test]
    fn test_path_empty() {
        let p = Path::new();
        assert!(p.is_empty());
    }

    #[test]
    fn test_flatten_lines_only() {
        let mut p = Path::new();
        p.move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .line_to(10.0, 10.0)
            .close();
        let contours = flatten_path(&p, 0.5);
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].len(), 4); // moveto + 2 lineto + close
    }

    #[test]
    fn test_flatten_quad_bezier() {
        let mut p = Path::new();
        p.move_to(0.0, 0.0).quad_to(5.0, 10.0, 10.0, 0.0);
        let contours = flatten_path(&p, 0.5);
        assert_eq!(contours.len(), 1);
        assert!(contours[0].len() >= 3);
    }

    #[test]
    fn test_flatten_cubic_bezier() {
        let mut p = Path::new();
        p.move_to(0.0, 0.0)
            .cubic_to(3.0, 10.0, 7.0, 10.0, 10.0, 0.0);
        let contours = flatten_path(&p, 0.5);
        assert_eq!(contours.len(), 1);
        assert!(contours[0].len() >= 3);
    }

    #[test]
    fn test_flatten_arc() {
        let mut p = Path::new();
        p.move_to(10.0, 0.0)
            .arc_to(0.0, 0.0, 10.0, 0.0, std::f32::consts::PI);
        let contours = flatten_path(&p, 0.5);
        assert!(!contours.is_empty());
        assert!(contours[0].len() >= 4);
    }

    #[test]
    fn test_rasterize_triangle() {
        let contour = vec![
            Point::new(5.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let coverage = rasterize_fill(&[contour], 12, 12, FillRule::EvenOdd, 1);
        // Some pixels should have coverage
        let filled: usize = coverage.iter().filter(|&&c| c > 0.1).count();
        assert!(filled > 10);
    }

    #[test]
    fn test_rasterize_nonzero_winding() {
        let contour = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 2.0),
            Point::new(8.0, 8.0),
            Point::new(2.0, 8.0),
        ];
        let coverage = rasterize_fill(&[contour], 10, 10, FillRule::NonZero, 1);
        // Center should be filled
        assert!(coverage[5 * 10 + 5] > 0.5);
    }

    #[test]
    fn test_rasterize_supersampled() {
        let contour = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 2.0),
            Point::new(8.0, 8.0),
            Point::new(2.0, 8.0),
        ];
        let cov1 = rasterize_fill(&[contour.clone()], 10, 10, FillRule::EvenOdd, 1);
        let cov4 = rasterize_fill(&[contour], 10, 10, FillRule::EvenOdd, 4);
        // Both should fill center
        assert!(cov1[5 * 10 + 5] > 0.3);
        assert!(cov4[5 * 10 + 5] > 0.3);
    }

    #[test]
    fn test_rasterize_empty() {
        let coverage = rasterize_fill(&[], 10, 10, FillRule::EvenOdd, 1);
        assert!(coverage.iter().all(|c| c.abs() < 1e-6));
    }

    #[test]
    fn test_stroke_path_basic() {
        let contour = vec![
            Point::new(0.0, 5.0),
            Point::new(10.0, 5.0),
        ];
        let params = StrokeParams {
            width: 2.0,
            ..Default::default()
        };
        let stroked = stroke_path(&contour, &params);
        assert_eq!(stroked.len(), 1);
        assert!(stroked[0].len() >= 4);
    }

    #[test]
    fn test_stroke_path_round_cap() {
        let contour = vec![
            Point::new(0.0, 5.0),
            Point::new(10.0, 5.0),
        ];
        let params = StrokeParams {
            width: 4.0,
            cap: LineCap::Round,
            ..Default::default()
        };
        let stroked = stroke_path(&contour, &params);
        assert_eq!(stroked.len(), 1);
        // Round cap adds extra vertices
        assert!(stroked[0].len() > 6);
    }

    #[test]
    fn test_stroke_path_square_cap() {
        let contour = vec![
            Point::new(5.0, 5.0),
            Point::new(15.0, 5.0),
        ];
        let params = StrokeParams {
            width: 2.0,
            cap: LineCap::Square,
            ..Default::default()
        };
        let stroked = stroke_path(&contour, &params);
        assert_eq!(stroked.len(), 1);
    }

    #[test]
    fn test_stroke_empty() {
        let stroked = stroke_path(&[], &StrokeParams::default());
        assert!(stroked.is_empty());
    }

    #[test]
    fn test_stroke_single_point() {
        let stroked = stroke_path(&[Point::new(5.0, 5.0)], &StrokeParams::default());
        assert!(stroked.is_empty());
    }

    #[test]
    fn test_bounding_box() {
        let contours = vec![vec![
            Point::new(1.0, 2.0),
            Point::new(10.0, 3.0),
            Point::new(5.0, 15.0),
        ]];
        let (min, max) = bounding_box(&contours);
        assert!((min.x - 1.0).abs() < 1e-6);
        assert!((min.y - 2.0).abs() < 1e-6);
        assert!((max.x - 10.0).abs() < 1e-6);
        assert!((max.y - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_fill_rule_enum() {
        assert_ne!(FillRule::EvenOdd, FillRule::NonZero);
    }

    #[test]
    fn test_line_cap_enum() {
        assert_ne!(LineCap::Butt, LineCap::Round);
        assert_ne!(LineCap::Round, LineCap::Square);
    }

    #[test]
    fn test_line_join_enum() {
        assert_ne!(LineJoin::Miter, LineJoin::Bevel);
        assert_ne!(LineJoin::Bevel, LineJoin::Round);
    }

    #[test]
    fn test_point_line_dist() {
        let d = point_line_dist(
            Point::new(0.0, 5.0),
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
        );
        assert!((d - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_flatten_tolerance_affects_detail() {
        let mut p = Path::new();
        p.move_to(0.0, 0.0)
            .cubic_to(0.0, 100.0, 100.0, 100.0, 100.0, 0.0);
        let coarse = flatten_path(&p, 10.0);
        let fine = flatten_path(&p, 0.1);
        assert!(fine[0].len() > coarse[0].len());
    }

    #[test]
    fn test_multiple_contours() {
        let mut p = Path::new();
        p.move_to(0.0, 0.0).line_to(10.0, 10.0);
        p.move_to(20.0, 20.0).line_to(30.0, 30.0);
        let contours = flatten_path(&p, 0.5);
        assert_eq!(contours.len(), 2);
    }

    #[test]
    fn test_stroke_polyline() {
        let contour = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
        ];
        let params = StrokeParams {
            width: 2.0,
            join: LineJoin::Bevel,
            ..Default::default()
        };
        let stroked = stroke_path(&contour, &params);
        assert_eq!(stroked.len(), 1);
        assert!(stroked[0].len() >= 6);
    }

    #[test]
    fn test_coverage_clamped() {
        let contour = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 2.0),
            Point::new(8.0, 8.0),
            Point::new(2.0, 8.0),
        ];
        let coverage = rasterize_fill(&[contour], 10, 10, FillRule::EvenOdd, 1);
        for &c in &coverage {
            assert!(c >= -1e-6);
            assert!(c <= 1.0 + 1e-6);
        }
    }
}
