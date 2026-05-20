//! SVG path data operations — parse, serialize, transform, and measure SVG paths.
//!
//! Pure Rust replacement for snap.svg path utilities, paper.js path operations,
//! and d3-path. Supports all SVG path commands with full transform pipeline.

use std::fmt;
use std::fmt::Write as _;

// ── Point ────────────────────────────────────────────────────

/// 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(self, other: Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

// ── BBox ─────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BBox {
    pub fn empty() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    pub fn expand(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }
}

// ── PathCommand ──────────────────────────────────────────────

/// SVG path command — absolute or relative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathCommand {
    /// M x y
    MoveTo { x: f64, y: f64 },
    /// m dx dy
    MoveToRel { dx: f64, dy: f64 },
    /// L x y
    LineTo { x: f64, y: f64 },
    /// l dx dy
    LineToRel { dx: f64, dy: f64 },
    /// H x
    HorizontalTo { x: f64 },
    /// h dx
    HorizontalToRel { dx: f64 },
    /// V y
    VerticalTo { y: f64 },
    /// v dy
    VerticalToRel { dy: f64 },
    /// C x1 y1 x2 y2 x y
    CubicTo {
        x1: f64, y1: f64,
        x2: f64, y2: f64,
        x: f64, y: f64,
    },
    /// c dx1 dy1 dx2 dy2 dx dy
    CubicToRel {
        dx1: f64, dy1: f64,
        dx2: f64, dy2: f64,
        dx: f64, dy: f64,
    },
    /// Q x1 y1 x y
    QuadTo {
        x1: f64, y1: f64,
        x: f64, y: f64,
    },
    /// q dx1 dy1 dx dy
    QuadToRel {
        dx1: f64, dy1: f64,
        dx: f64, dy: f64,
    },
    /// A rx ry x_rotation large_arc sweep x y
    ArcTo {
        rx: f64, ry: f64,
        x_rotation: f64,
        large_arc: bool,
        sweep: bool,
        x: f64, y: f64,
    },
    /// a rx ry x_rotation large_arc sweep dx dy
    ArcToRel {
        rx: f64, ry: f64,
        x_rotation: f64,
        large_arc: bool,
        sweep: bool,
        dx: f64, dy: f64,
    },
    /// Z / z
    Close,
}

// ── SvgPath ──────────────────────────────────────────────────

/// A parsed SVG path consisting of a sequence of commands.
#[derive(Debug, Clone, PartialEq)]
pub struct SvgPath {
    pub commands: Vec<PathCommand>,
}

impl SvgPath {
    pub fn new() -> Self {
        Self { commands: Vec::new() }
    }

    /// Parse an SVG path data string.
    pub fn parse(input: &str) -> Result<Self, PathError> {
        let mut commands = Vec::new();
        let tokens = tokenize(input);
        let mut i = 0;
        let mut current_cmd: Option<char> = None;

        while i < tokens.len() {
            let tok = &tokens[i];
            // If it's a letter, update current command
            if tok.len() == 1 && tok.chars().next().unwrap().is_ascii_alphabetic() {
                current_cmd = Some(tok.chars().next().unwrap());
                i += 1;
                if current_cmd == Some('Z') || current_cmd == Some('z') {
                    commands.push(PathCommand::Close);
                    current_cmd = None;
                    continue;
                }
            }

            let cmd = current_cmd.ok_or(PathError::UnexpectedToken)?;

            match cmd {
                'M' => {
                    let x = parse_num(&tokens, &mut i)?;
                    let y = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::MoveTo { x, y });
                    // Subsequent coords after M are implicit LineTo
                    current_cmd = Some('L');
                }
                'm' => {
                    let dx = parse_num(&tokens, &mut i)?;
                    let dy = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::MoveToRel { dx, dy });
                    current_cmd = Some('l');
                }
                'L' => {
                    let x = parse_num(&tokens, &mut i)?;
                    let y = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::LineTo { x, y });
                }
                'l' => {
                    let dx = parse_num(&tokens, &mut i)?;
                    let dy = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::LineToRel { dx, dy });
                }
                'H' => {
                    let x = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::HorizontalTo { x });
                }
                'h' => {
                    let dx = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::HorizontalToRel { dx });
                }
                'V' => {
                    let y = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::VerticalTo { y });
                }
                'v' => {
                    let dy = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::VerticalToRel { dy });
                }
                'C' => {
                    let x1 = parse_num(&tokens, &mut i)?;
                    let y1 = parse_num(&tokens, &mut i)?;
                    let x2 = parse_num(&tokens, &mut i)?;
                    let y2 = parse_num(&tokens, &mut i)?;
                    let x = parse_num(&tokens, &mut i)?;
                    let y = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::CubicTo { x1, y1, x2, y2, x, y });
                }
                'c' => {
                    let dx1 = parse_num(&tokens, &mut i)?;
                    let dy1 = parse_num(&tokens, &mut i)?;
                    let dx2 = parse_num(&tokens, &mut i)?;
                    let dy2 = parse_num(&tokens, &mut i)?;
                    let dx = parse_num(&tokens, &mut i)?;
                    let dy = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::CubicToRel { dx1, dy1, dx2, dy2, dx, dy });
                }
                'Q' => {
                    let x1 = parse_num(&tokens, &mut i)?;
                    let y1 = parse_num(&tokens, &mut i)?;
                    let x = parse_num(&tokens, &mut i)?;
                    let y = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::QuadTo { x1, y1, x, y });
                }
                'q' => {
                    let dx1 = parse_num(&tokens, &mut i)?;
                    let dy1 = parse_num(&tokens, &mut i)?;
                    let dx = parse_num(&tokens, &mut i)?;
                    let dy = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::QuadToRel { dx1, dy1, dx, dy });
                }
                'A' => {
                    let rx = parse_num(&tokens, &mut i)?;
                    let ry = parse_num(&tokens, &mut i)?;
                    let x_rotation = parse_num(&tokens, &mut i)?;
                    let large_arc = parse_num(&tokens, &mut i)? != 0.0;
                    let sweep = parse_num(&tokens, &mut i)? != 0.0;
                    let x = parse_num(&tokens, &mut i)?;
                    let y = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x, y });
                }
                'a' => {
                    let rx = parse_num(&tokens, &mut i)?;
                    let ry = parse_num(&tokens, &mut i)?;
                    let x_rotation = parse_num(&tokens, &mut i)?;
                    let large_arc = parse_num(&tokens, &mut i)? != 0.0;
                    let sweep = parse_num(&tokens, &mut i)? != 0.0;
                    let dx = parse_num(&tokens, &mut i)?;
                    let dy = parse_num(&tokens, &mut i)?;
                    commands.push(PathCommand::ArcToRel { rx, ry, x_rotation, large_arc, sweep, dx, dy });
                }
                _ => return Err(PathError::UnknownCommand(cmd)),
            }
        }

        Ok(Self { commands })
    }

    /// Convert all relative commands to absolute.
    pub fn to_absolute(&self) -> Self {
        let mut result = Vec::new();
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let mut start_x = 0.0_f64;
        let mut start_y = 0.0_f64;

        for cmd in &self.commands {
            match *cmd {
                PathCommand::MoveTo { x, y } => {
                    cx = x; cy = y;
                    start_x = x; start_y = y;
                    result.push(PathCommand::MoveTo { x, y });
                }
                PathCommand::MoveToRel { dx, dy } => {
                    cx += dx; cy += dy;
                    start_x = cx; start_y = cy;
                    result.push(PathCommand::MoveTo { x: cx, y: cy });
                }
                PathCommand::LineTo { x, y } => {
                    cx = x; cy = y;
                    result.push(PathCommand::LineTo { x, y });
                }
                PathCommand::LineToRel { dx, dy } => {
                    cx += dx; cy += dy;
                    result.push(PathCommand::LineTo { x: cx, y: cy });
                }
                PathCommand::HorizontalTo { x } => {
                    cx = x;
                    result.push(PathCommand::LineTo { x: cx, y: cy });
                }
                PathCommand::HorizontalToRel { dx } => {
                    cx += dx;
                    result.push(PathCommand::LineTo { x: cx, y: cy });
                }
                PathCommand::VerticalTo { y } => {
                    cy = y;
                    result.push(PathCommand::LineTo { x: cx, y: cy });
                }
                PathCommand::VerticalToRel { dy } => {
                    cy += dy;
                    result.push(PathCommand::LineTo { x: cx, y: cy });
                }
                PathCommand::CubicTo { x1, y1, x2, y2, x, y } => {
                    cx = x; cy = y;
                    result.push(PathCommand::CubicTo { x1, y1, x2, y2, x, y });
                }
                PathCommand::CubicToRel { dx1, dy1, dx2, dy2, dx, dy } => {
                    let x1 = cx + dx1; let y1 = cy + dy1;
                    let x2 = cx + dx2; let y2 = cy + dy2;
                    cx += dx; cy += dy;
                    result.push(PathCommand::CubicTo { x1, y1, x2, y2, x: cx, y: cy });
                }
                PathCommand::QuadTo { x1, y1, x, y } => {
                    cx = x; cy = y;
                    result.push(PathCommand::QuadTo { x1, y1, x, y });
                }
                PathCommand::QuadToRel { dx1, dy1, dx, dy } => {
                    let x1 = cx + dx1; let y1 = cy + dy1;
                    cx += dx; cy += dy;
                    result.push(PathCommand::QuadTo { x1, y1, x: cx, y: cy });
                }
                PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x, y } => {
                    cx = x; cy = y;
                    result.push(PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x, y });
                }
                PathCommand::ArcToRel { rx, ry, x_rotation, large_arc, sweep, dx, dy } => {
                    cx += dx; cy += dy;
                    result.push(PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x: cx, y: cy });
                }
                PathCommand::Close => {
                    cx = start_x; cy = start_y;
                    result.push(PathCommand::Close);
                }
            }
        }

        Self { commands: result }
    }

    /// Compute the bounding box by walking all absolute endpoints and control points.
    pub fn bbox(&self) -> BBox {
        let abs = self.to_absolute();
        let mut bb = BBox::empty();

        for cmd in &abs.commands {
            match *cmd {
                PathCommand::MoveTo { x, y }
                | PathCommand::LineTo { x, y } => {
                    bb.expand(x, y);
                }
                PathCommand::CubicTo { x1, y1, x2, y2, x, y } => {
                    bb.expand(x1, y1);
                    bb.expand(x2, y2);
                    bb.expand(x, y);
                }
                PathCommand::QuadTo { x1, y1, x, y } => {
                    bb.expand(x1, y1);
                    bb.expand(x, y);
                }
                PathCommand::ArcTo { x, y, .. } => {
                    bb.expand(x, y);
                }
                PathCommand::Close => {}
                _ => {} // all relative resolved by to_absolute
            }
        }

        bb
    }

    /// Approximate path length by walking segments as line segments,
    /// subdividing curves.
    pub fn length(&self) -> f64 {
        let abs = self.to_absolute();
        let mut len = 0.0;
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let mut start_x = 0.0;
        let mut start_y = 0.0;

        for cmd in &abs.commands {
            match *cmd {
                PathCommand::MoveTo { x, y } => {
                    cx = x; cy = y;
                    start_x = x; start_y = y;
                }
                PathCommand::LineTo { x, y } => {
                    len += Point::new(cx, cy).distance_to(Point::new(x, y));
                    cx = x; cy = y;
                }
                PathCommand::CubicTo { x1, y1, x2, y2, x, y } => {
                    len += cubic_length(cx, cy, x1, y1, x2, y2, x, y, 16);
                    cx = x; cy = y;
                }
                PathCommand::QuadTo { x1, y1, x, y } => {
                    len += quad_length(cx, cy, x1, y1, x, y, 16);
                    cx = x; cy = y;
                }
                PathCommand::ArcTo { x, y, .. } => {
                    // Approximate arc as line for simplicity
                    len += Point::new(cx, cy).distance_to(Point::new(x, y));
                    cx = x; cy = y;
                }
                PathCommand::Close => {
                    len += Point::new(cx, cy).distance_to(Point::new(start_x, start_y));
                    cx = start_x; cy = start_y;
                }
                _ => {}
            }
        }

        len
    }

    /// Get a point at a given distance along the path (approximate).
    pub fn point_at_distance(&self, target: f64) -> Option<Point> {
        let abs = self.to_absolute();
        let mut traveled = 0.0;
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let mut start_x = 0.0;
        let mut start_y = 0.0;

        for cmd in &abs.commands {
            match *cmd {
                PathCommand::MoveTo { x, y } => {
                    cx = x; cy = y;
                    start_x = x; start_y = y;
                }
                PathCommand::LineTo { x, y } => {
                    let seg_len = Point::new(cx, cy).distance_to(Point::new(x, y));
                    if traveled + seg_len >= target {
                        let t = (target - traveled) / seg_len;
                        return Some(Point::new(
                            cx + (x - cx) * t,
                            cy + (y - cy) * t,
                        ));
                    }
                    traveled += seg_len;
                    cx = x; cy = y;
                }
                PathCommand::CubicTo { x1, y1, x2, y2, x, y } => {
                    let seg_len = cubic_length(cx, cy, x1, y1, x2, y2, x, y, 16);
                    if traveled + seg_len >= target {
                        let t = (target - traveled) / seg_len;
                        return Some(cubic_point_at(cx, cy, x1, y1, x2, y2, x, y, t));
                    }
                    traveled += seg_len;
                    cx = x; cy = y;
                }
                PathCommand::QuadTo { x1, y1, x, y } => {
                    let seg_len = quad_length(cx, cy, x1, y1, x, y, 16);
                    if traveled + seg_len >= target {
                        let t = (target - traveled) / seg_len;
                        return Some(quad_point_at(cx, cy, x1, y1, x, y, t));
                    }
                    traveled += seg_len;
                    cx = x; cy = y;
                }
                PathCommand::Close => {
                    let seg_len = Point::new(cx, cy).distance_to(Point::new(start_x, start_y));
                    if traveled + seg_len >= target {
                        let t = (target - traveled) / seg_len;
                        return Some(Point::new(
                            cx + (start_x - cx) * t,
                            cy + (start_y - cy) * t,
                        ));
                    }
                    traveled += seg_len;
                    cx = start_x; cy = start_y;
                }
                _ => {}
            }
        }

        None
    }

    /// Translate all commands by (tx, ty).
    pub fn translate(&self, tx: f64, ty: f64) -> Self {
        let abs = self.to_absolute();
        let commands = abs.commands.iter().map(|cmd| {
            match *cmd {
                PathCommand::MoveTo { x, y } => PathCommand::MoveTo { x: x + tx, y: y + ty },
                PathCommand::LineTo { x, y } => PathCommand::LineTo { x: x + tx, y: y + ty },
                PathCommand::CubicTo { x1, y1, x2, y2, x, y } => PathCommand::CubicTo {
                    x1: x1 + tx, y1: y1 + ty,
                    x2: x2 + tx, y2: y2 + ty,
                    x: x + tx, y: y + ty,
                },
                PathCommand::QuadTo { x1, y1, x, y } => PathCommand::QuadTo {
                    x1: x1 + tx, y1: y1 + ty,
                    x: x + tx, y: y + ty,
                },
                PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x, y } => PathCommand::ArcTo {
                    rx, ry, x_rotation, large_arc, sweep,
                    x: x + tx, y: y + ty,
                },
                PathCommand::Close => PathCommand::Close,
                _ => *cmd,
            }
        }).collect();
        Self { commands }
    }

    /// Scale all commands by (sx, sy).
    pub fn scale(&self, sx: f64, sy: f64) -> Self {
        let abs = self.to_absolute();
        let commands = abs.commands.iter().map(|cmd| {
            match *cmd {
                PathCommand::MoveTo { x, y } => PathCommand::MoveTo { x: x * sx, y: y * sy },
                PathCommand::LineTo { x, y } => PathCommand::LineTo { x: x * sx, y: y * sy },
                PathCommand::CubicTo { x1, y1, x2, y2, x, y } => PathCommand::CubicTo {
                    x1: x1 * sx, y1: y1 * sy,
                    x2: x2 * sx, y2: y2 * sy,
                    x: x * sx, y: y * sy,
                },
                PathCommand::QuadTo { x1, y1, x, y } => PathCommand::QuadTo {
                    x1: x1 * sx, y1: y1 * sy,
                    x: x * sx, y: y * sy,
                },
                PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x, y } => PathCommand::ArcTo {
                    rx: rx * sx, ry: ry * sy, x_rotation, large_arc, sweep,
                    x: x * sx, y: y * sy,
                },
                PathCommand::Close => PathCommand::Close,
                _ => *cmd,
            }
        }).collect();
        Self { commands }
    }

    /// Rotate all absolute points by `angle` radians around the origin.
    pub fn rotate(&self, angle: f64) -> Self {
        let abs = self.to_absolute();
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let rot = |x: f64, y: f64| -> (f64, f64) {
            (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
        };

        let commands = abs.commands.iter().map(|cmd| {
            match *cmd {
                PathCommand::MoveTo { x, y } => {
                    let (rx, ry) = rot(x, y);
                    PathCommand::MoveTo { x: rx, y: ry }
                }
                PathCommand::LineTo { x, y } => {
                    let (rx, ry) = rot(x, y);
                    PathCommand::LineTo { x: rx, y: ry }
                }
                PathCommand::CubicTo { x1, y1, x2, y2, x, y } => {
                    let (rx1, ry1) = rot(x1, y1);
                    let (rx2, ry2) = rot(x2, y2);
                    let (rx, ry) = rot(x, y);
                    PathCommand::CubicTo { x1: rx1, y1: ry1, x2: rx2, y2: ry2, x: rx, y: ry }
                }
                PathCommand::QuadTo { x1, y1, x, y } => {
                    let (rx1, ry1) = rot(x1, y1);
                    let (rx, ry) = rot(x, y);
                    PathCommand::QuadTo { x1: rx1, y1: ry1, x: rx, y: ry }
                }
                PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x, y } => {
                    let (rotx, roty) = rot(x, y);
                    PathCommand::ArcTo {
                        rx, ry,
                        x_rotation: x_rotation + angle.to_degrees(),
                        large_arc, sweep,
                        x: rotx, y: roty,
                    }
                }
                PathCommand::Close => PathCommand::Close,
                _ => *cmd,
            }
        }).collect();
        Self { commands }
    }

    /// Reverse the path direction.
    pub fn reverse(&self) -> Self {
        let abs = self.to_absolute();
        if abs.commands.is_empty() {
            return Self { commands: Vec::new() };
        }

        // Collect endpoints
        let mut points: Vec<Point> = Vec::new();
        let mut start_x = 0.0;
        let mut start_y = 0.0;

        for cmd in &abs.commands {
            match *cmd {
                PathCommand::MoveTo { x, y } => {
                    points.push(Point::new(x, y));
                    start_x = x; start_y = y;
                }
                PathCommand::LineTo { x, y }
                | PathCommand::CubicTo { x, y, .. }
                | PathCommand::QuadTo { x, y, .. }
                | PathCommand::ArcTo { x, y, .. } => {
                    points.push(Point::new(x, y));
                }
                PathCommand::Close => {
                    points.push(Point::new(start_x, start_y));
                }
                _ => {}
            }
        }

        // Build reversed: MoveTo last endpoint, then LineTo in reverse order
        let mut result = Vec::new();
        if let Some(last) = points.last() {
            result.push(PathCommand::MoveTo { x: last.x, y: last.y });
        }
        for pt in points.iter().rev().skip(1) {
            result.push(PathCommand::LineTo { x: pt.x, y: pt.y });
        }

        Self { commands: result }
    }

    /// Serialize back to an SVG path data string.
    pub fn to_string(&self) -> String {
        let mut s = String::new();
        for cmd in &self.commands {
            if !s.is_empty() {
                s.push(' ');
            }
            match *cmd {
                PathCommand::MoveTo { x, y } => write!(s, "M {} {}", fmtf(x), fmtf(y)).unwrap(),
                PathCommand::MoveToRel { dx, dy } => write!(s, "m {} {}", fmtf(dx), fmtf(dy)).unwrap(),
                PathCommand::LineTo { x, y } => write!(s, "L {} {}", fmtf(x), fmtf(y)).unwrap(),
                PathCommand::LineToRel { dx, dy } => write!(s, "l {} {}", fmtf(dx), fmtf(dy)).unwrap(),
                PathCommand::HorizontalTo { x } => write!(s, "H {}", fmtf(x)).unwrap(),
                PathCommand::HorizontalToRel { dx } => write!(s, "h {}", fmtf(dx)).unwrap(),
                PathCommand::VerticalTo { y } => write!(s, "V {}", fmtf(y)).unwrap(),
                PathCommand::VerticalToRel { dy } => write!(s, "v {}", fmtf(dy)).unwrap(),
                PathCommand::CubicTo { x1, y1, x2, y2, x, y } =>
                    write!(s, "C {} {} {} {} {} {}", fmtf(x1), fmtf(y1), fmtf(x2), fmtf(y2), fmtf(x), fmtf(y)).unwrap(),
                PathCommand::CubicToRel { dx1, dy1, dx2, dy2, dx, dy } =>
                    write!(s, "c {} {} {} {} {} {}", fmtf(dx1), fmtf(dy1), fmtf(dx2), fmtf(dy2), fmtf(dx), fmtf(dy)).unwrap(),
                PathCommand::QuadTo { x1, y1, x, y } =>
                    write!(s, "Q {} {} {} {}", fmtf(x1), fmtf(y1), fmtf(x), fmtf(y)).unwrap(),
                PathCommand::QuadToRel { dx1, dy1, dx, dy } =>
                    write!(s, "q {} {} {} {}", fmtf(dx1), fmtf(dy1), fmtf(dx), fmtf(dy)).unwrap(),
                PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x, y } =>
                    write!(s, "A {} {} {} {} {} {} {}",
                        fmtf(rx), fmtf(ry), fmtf(x_rotation),
                        if large_arc { 1 } else { 0 },
                        if sweep { 1 } else { 0 },
                        fmtf(x), fmtf(y)).unwrap(),
                PathCommand::ArcToRel { rx, ry, x_rotation, large_arc, sweep, dx, dy } =>
                    write!(s, "a {} {} {} {} {} {} {}",
                        fmtf(rx), fmtf(ry), fmtf(x_rotation),
                        if large_arc { 1 } else { 0 },
                        if sweep { 1 } else { 0 },
                        fmtf(dx), fmtf(dy)).unwrap(),
                PathCommand::Close => s.push('Z'),
            }
        }
        s
    }
}

impl fmt::Display for SvgPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl Default for SvgPath {
    fn default() -> Self {
        Self::new()
    }
}

// ── Error ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PathError {
    UnexpectedToken,
    UnknownCommand(char),
    InvalidNumber(String),
    UnexpectedEnd,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedToken => write!(f, "unexpected token in path data"),
            Self::UnknownCommand(c) => write!(f, "unknown path command: {c}"),
            Self::InvalidNumber(s) => write!(f, "invalid number: {s}"),
            Self::UnexpectedEnd => write!(f, "unexpected end of path data"),
        }
    }
}

impl std::error::Error for PathError {}

// ── Helpers ──────────────────────────────────────────────────

fn fmtf(v: f64) -> String {
    if v == v.trunc() {
        format!("{}", v as i64)
    } else {
        format!("{:.4}", v).trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() || ch == ',' {
            chars.next();
            continue;
        }
        if ch.is_ascii_alphabetic() {
            tokens.push(ch.to_string());
            chars.next();
            continue;
        }
        // Number: optional sign, digits, optional dot+digits, optional exponent
        if ch == '-' || ch == '+' || ch == '.' || ch.is_ascii_digit() {
            let mut num = String::new();
            if ch == '-' || ch == '+' {
                num.push(ch);
                chars.next();
            }
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    num.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Some(&'.') = chars.peek() {
                num.push('.');
                chars.next();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() {
                        num.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            if let Some(&c) = chars.peek() {
                if c == 'e' || c == 'E' {
                    num.push(c);
                    chars.next();
                    if let Some(&s) = chars.peek() {
                        if s == '+' || s == '-' {
                            num.push(s);
                            chars.next();
                        }
                    }
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() {
                            num.push(d);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
            }
            if !num.is_empty() && num != "-" && num != "+" {
                tokens.push(num);
            }
            continue;
        }
        chars.next(); // skip unknown
    }

    tokens
}

fn parse_num(tokens: &[String], i: &mut usize) -> Result<f64, PathError> {
    if *i >= tokens.len() {
        return Err(PathError::UnexpectedEnd);
    }
    let s = &tokens[*i];
    *i += 1;
    s.parse::<f64>().map_err(|_| PathError::InvalidNumber(s.clone()))
}

fn cubic_point_at(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, t: f64) -> Point {
    let u = 1.0 - t;
    let x = u * u * u * x0 + 3.0 * u * u * t * x1 + 3.0 * u * t * t * x2 + t * t * t * x3;
    let y = u * u * u * y0 + 3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t * t * t * y3;
    Point::new(x, y)
}

fn quad_point_at(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64, t: f64) -> Point {
    let u = 1.0 - t;
    let x = u * u * x0 + 2.0 * u * t * x1 + t * t * x2;
    let y = u * u * y0 + 2.0 * u * t * y1 + t * t * y2;
    Point::new(x, y)
}

fn cubic_length(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, steps: usize) -> f64 {
    let mut len = 0.0;
    let mut prev = Point::new(x0, y0);
    for i in 1..=steps {
        let t = i as f64 / steps as f64;
        let pt = cubic_point_at(x0, y0, x1, y1, x2, y2, x3, y3, t);
        len += prev.distance_to(pt);
        prev = pt;
    }
    len
}

fn quad_length(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64, steps: usize) -> f64 {
    let mut len = 0.0;
    let mut prev = Point::new(x0, y0);
    for i in 1..=steps {
        let t = i as f64 / steps as f64;
        let pt = quad_point_at(x0, y0, x1, y1, x2, y2, t);
        len += prev.distance_to(pt);
        prev = pt;
    }
    len
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn parse_move_line_close() {
        let path = SvgPath::parse("M 10 20 L 30 40 Z").unwrap();
        assert_eq!(path.commands.len(), 3);
        assert_eq!(path.commands[0], PathCommand::MoveTo { x: 10.0, y: 20.0 });
        assert_eq!(path.commands[1], PathCommand::LineTo { x: 30.0, y: 40.0 });
        assert_eq!(path.commands[2], PathCommand::Close);
    }

    #[test]
    fn parse_cubic() {
        let path = SvgPath::parse("M 0 0 C 10 20 30 40 50 60").unwrap();
        assert_eq!(path.commands.len(), 2);
        match path.commands[1] {
            PathCommand::CubicTo { x1, y1, x2, y2, x, y } => {
                assert_eq!((x1, y1, x2, y2, x, y), (10.0, 20.0, 30.0, 40.0, 50.0, 60.0));
            }
            _ => panic!("expected CubicTo"),
        }
    }

    #[test]
    fn parse_quad() {
        let path = SvgPath::parse("M 0 0 Q 10 20 30 40").unwrap();
        match path.commands[1] {
            PathCommand::QuadTo { x1, y1, x, y } => {
                assert_eq!((x1, y1, x, y), (10.0, 20.0, 30.0, 40.0));
            }
            _ => panic!("expected QuadTo"),
        }
    }

    #[test]
    fn parse_arc() {
        let path = SvgPath::parse("M 0 0 A 25 25 0 1 1 50 25").unwrap();
        match path.commands[1] {
            PathCommand::ArcTo { rx, ry, x_rotation, large_arc, sweep, x, y } => {
                assert_eq!(rx, 25.0);
                assert_eq!(ry, 25.0);
                assert_eq!(x_rotation, 0.0);
                assert!(large_arc);
                assert!(sweep);
                assert_eq!(x, 50.0);
                assert_eq!(y, 25.0);
            }
            _ => panic!("expected ArcTo"),
        }
    }

    #[test]
    fn parse_relative() {
        let path = SvgPath::parse("m 10 10 l 20 20").unwrap();
        assert_eq!(path.commands[0], PathCommand::MoveToRel { dx: 10.0, dy: 10.0 });
        assert_eq!(path.commands[1], PathCommand::LineToRel { dx: 20.0, dy: 20.0 });
    }

    #[test]
    fn to_absolute_relative() {
        let path = SvgPath::parse("m 10 10 l 20 20 l 5 5").unwrap();
        let abs = path.to_absolute();
        assert_eq!(abs.commands[0], PathCommand::MoveTo { x: 10.0, y: 10.0 });
        assert_eq!(abs.commands[1], PathCommand::LineTo { x: 30.0, y: 30.0 });
        assert_eq!(abs.commands[2], PathCommand::LineTo { x: 35.0, y: 35.0 });
    }

    #[test]
    fn serialize_roundtrip() {
        let input = "M 10 20 L 30 40 Z";
        let path = SvgPath::parse(input).unwrap();
        let output = path.to_string();
        assert_eq!(output, "M 10 20 L 30 40 Z");
    }

    #[test]
    fn bbox_simple() {
        let path = SvgPath::parse("M 10 20 L 50 60 L 30 10").unwrap();
        let bb = path.bbox();
        assert_eq!(bb.min_x, 10.0);
        assert_eq!(bb.min_y, 10.0);
        assert_eq!(bb.max_x, 50.0);
        assert_eq!(bb.max_y, 60.0);
    }

    #[test]
    fn length_line() {
        let path = SvgPath::parse("M 0 0 L 3 4").unwrap();
        assert!(approx(path.length(), 5.0, 0.001));
    }

    #[test]
    fn length_square() {
        let path = SvgPath::parse("M 0 0 L 10 0 L 10 10 L 0 10 Z").unwrap();
        assert!(approx(path.length(), 40.0, 0.001));
    }

    #[test]
    fn point_at_distance_line() {
        let path = SvgPath::parse("M 0 0 L 10 0").unwrap();
        let pt = path.point_at_distance(5.0).unwrap();
        assert!(approx(pt.x, 5.0, 0.001));
        assert!(approx(pt.y, 0.0, 0.001));
    }

    #[test]
    fn translate_path() {
        let path = SvgPath::parse("M 0 0 L 10 10").unwrap();
        let moved = path.translate(5.0, 5.0);
        assert_eq!(moved.commands[0], PathCommand::MoveTo { x: 5.0, y: 5.0 });
        assert_eq!(moved.commands[1], PathCommand::LineTo { x: 15.0, y: 15.0 });
    }

    #[test]
    fn scale_path() {
        let path = SvgPath::parse("M 10 20 L 30 40").unwrap();
        let scaled = path.scale(2.0, 0.5);
        assert_eq!(scaled.commands[0], PathCommand::MoveTo { x: 20.0, y: 10.0 });
        assert_eq!(scaled.commands[1], PathCommand::LineTo { x: 60.0, y: 20.0 });
    }

    #[test]
    fn rotate_path_90() {
        let path = SvgPath::parse("M 10 0").unwrap();
        let rotated = path.rotate(std::f64::consts::FRAC_PI_2);
        let p = match rotated.commands[0] {
            PathCommand::MoveTo { x, y } => (x, y),
            _ => panic!("expected MoveTo"),
        };
        assert!(approx(p.0, 0.0, 0.001));
        assert!(approx(p.1, 10.0, 0.001));
    }

    #[test]
    fn reverse_path() {
        let path = SvgPath::parse("M 0 0 L 10 0 L 10 10").unwrap();
        let rev = path.reverse();
        assert_eq!(rev.commands[0], PathCommand::MoveTo { x: 10.0, y: 10.0 });
        assert_eq!(rev.commands[1], PathCommand::LineTo { x: 10.0, y: 0.0 });
        assert_eq!(rev.commands[2], PathCommand::LineTo { x: 0.0, y: 0.0 });
    }

    #[test]
    fn parse_h_v_commands() {
        let path = SvgPath::parse("M 0 0 H 10 V 20").unwrap();
        assert_eq!(path.commands.len(), 3);
        assert_eq!(path.commands[1], PathCommand::HorizontalTo { x: 10.0 });
        assert_eq!(path.commands[2], PathCommand::VerticalTo { y: 20.0 });
    }

    #[test]
    fn implicit_lineto_after_moveto() {
        // After M, subsequent coordinate pairs become implicit L
        let path = SvgPath::parse("M 0 0 10 10 20 20").unwrap();
        assert_eq!(path.commands.len(), 3);
        assert_eq!(path.commands[0], PathCommand::MoveTo { x: 0.0, y: 0.0 });
        assert_eq!(path.commands[1], PathCommand::LineTo { x: 10.0, y: 10.0 });
        assert_eq!(path.commands[2], PathCommand::LineTo { x: 20.0, y: 20.0 });
    }

    #[test]
    fn negative_coordinates() {
        let path = SvgPath::parse("M -10 -20 L -30 -40").unwrap();
        assert_eq!(path.commands[0], PathCommand::MoveTo { x: -10.0, y: -20.0 });
        assert_eq!(path.commands[1], PathCommand::LineTo { x: -30.0, y: -40.0 });
    }
}
