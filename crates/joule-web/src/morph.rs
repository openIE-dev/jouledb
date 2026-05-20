//! Shape morphing engine — SVG path interpolation.
//!
//! Replaces flubber / GreenSock MorphSVGPlugin. Parses SVG path
//! commands, normalizes two paths to the same point count via
//! subdivision, and interpolates between them at time `t`.

use std::fmt;

// ── Point ──────────────────────────────────────────────────────

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorphPoint {
    pub x: f64,
    pub y: f64,
}

impl MorphPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn lerp(self, other: MorphPoint, t: f64) -> MorphPoint {
        MorphPoint {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    pub fn distance_to(self, other: MorphPoint) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Angle from origin to this point.
    pub fn angle(self) -> f64 {
        self.y.atan2(self.x)
    }
}

impl fmt::Display for MorphPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4},{:.4}", self.x, self.y)
    }
}

// ── SVG Path Commands ──────────────────────────────────────────

/// SVG path command (absolute).
#[derive(Debug, Clone, PartialEq)]
pub enum PathCommand {
    MoveTo(MorphPoint),
    LineTo(MorphPoint),
    HorizontalTo(f64),
    VerticalTo(f64),
    CubicTo(MorphPoint, MorphPoint, MorphPoint),
    QuadraticTo(MorphPoint, MorphPoint),
    Close,
}

// ── Parser ─────────────────────────────────────────────────────

/// Parse an SVG path data string into absolute path commands.
pub fn parse_svg_path(d: &str) -> Vec<PathCommand> {
    let mut commands = Vec::new();
    let mut cursor = MorphPoint::new(0.0, 0.0);
    let mut start = MorphPoint::new(0.0, 0.0);

    let tokens = tokenize_path(d);
    let mut i = 0;

    while i < tokens.len() {
        match tokens[i].as_str() {
            "M" => {
                let p = read_point(&tokens, i + 1);
                cursor = p;
                start = p;
                commands.push(PathCommand::MoveTo(p));
                i += 3;
            }
            "m" => {
                let dp = read_point(&tokens, i + 1);
                cursor = MorphPoint::new(cursor.x + dp.x, cursor.y + dp.y);
                start = cursor;
                commands.push(PathCommand::MoveTo(cursor));
                i += 3;
            }
            "L" => {
                let p = read_point(&tokens, i + 1);
                cursor = p;
                commands.push(PathCommand::LineTo(p));
                i += 3;
            }
            "l" => {
                let dp = read_point(&tokens, i + 1);
                cursor = MorphPoint::new(cursor.x + dp.x, cursor.y + dp.y);
                commands.push(PathCommand::LineTo(cursor));
                i += 3;
            }
            "H" => {
                let x = parse_num(&tokens[i + 1]);
                cursor.x = x;
                commands.push(PathCommand::HorizontalTo(x));
                i += 2;
            }
            "h" => {
                let dx = parse_num(&tokens[i + 1]);
                cursor.x += dx;
                commands.push(PathCommand::HorizontalTo(cursor.x));
                i += 2;
            }
            "V" => {
                let y = parse_num(&tokens[i + 1]);
                cursor.y = y;
                commands.push(PathCommand::VerticalTo(y));
                i += 2;
            }
            "v" => {
                let dy = parse_num(&tokens[i + 1]);
                cursor.y += dy;
                commands.push(PathCommand::VerticalTo(cursor.y));
                i += 2;
            }
            "C" => {
                let c1 = read_point(&tokens, i + 1);
                let c2 = read_point(&tokens, i + 3);
                let p = read_point(&tokens, i + 5);
                cursor = p;
                commands.push(PathCommand::CubicTo(c1, c2, p));
                i += 7;
            }
            "c" => {
                let c1 = MorphPoint::new(cursor.x + parse_num(&tokens[i + 1]), cursor.y + parse_num(&tokens[i + 2]));
                let c2 = MorphPoint::new(cursor.x + parse_num(&tokens[i + 3]), cursor.y + parse_num(&tokens[i + 4]));
                let p = MorphPoint::new(cursor.x + parse_num(&tokens[i + 5]), cursor.y + parse_num(&tokens[i + 6]));
                cursor = p;
                commands.push(PathCommand::CubicTo(c1, c2, p));
                i += 7;
            }
            "Q" => {
                let cp = read_point(&tokens, i + 1);
                let p = read_point(&tokens, i + 3);
                cursor = p;
                commands.push(PathCommand::QuadraticTo(cp, p));
                i += 5;
            }
            "q" => {
                let cp = MorphPoint::new(cursor.x + parse_num(&tokens[i + 1]), cursor.y + parse_num(&tokens[i + 2]));
                let p = MorphPoint::new(cursor.x + parse_num(&tokens[i + 3]), cursor.y + parse_num(&tokens[i + 4]));
                cursor = p;
                commands.push(PathCommand::QuadraticTo(cp, p));
                i += 5;
            }
            "Z" | "z" => {
                cursor = start;
                commands.push(PathCommand::Close);
                i += 1;
            }
            _ => {
                // Unknown token — skip.
                i += 1;
            }
        }
    }

    commands
}

fn tokenize_path(d: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in d.chars() {
        if ch.is_ascii_alphabetic() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
        } else if ch == ',' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else if ch == '-' && !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
            current.push(ch);
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn parse_num(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

fn read_point(tokens: &[String], idx: usize) -> MorphPoint {
    let x = if idx < tokens.len() { parse_num(&tokens[idx]) } else { 0.0 };
    let y = if idx + 1 < tokens.len() { parse_num(&tokens[idx + 1]) } else { 0.0 };
    MorphPoint::new(x, y)
}

// ── Path to Points ─────────────────────────────────────────────

/// Flatten path commands to a list of points.
pub fn commands_to_points(commands: &[PathCommand]) -> Vec<MorphPoint> {
    let mut points = Vec::new();
    let mut cursor = MorphPoint::new(0.0, 0.0);

    for cmd in commands {
        match cmd {
            PathCommand::MoveTo(p) => {
                cursor = *p;
                points.push(cursor);
            }
            PathCommand::LineTo(p) => {
                cursor = *p;
                points.push(cursor);
            }
            PathCommand::HorizontalTo(x) => {
                cursor.x = *x;
                points.push(cursor);
            }
            PathCommand::VerticalTo(y) => {
                cursor.y = *y;
                points.push(cursor);
            }
            PathCommand::CubicTo(c1, c2, p) => {
                // Sample the cubic at a few points.
                let p0 = cursor;
                for i in 1..=4 {
                    let t = i as f64 / 4.0;
                    let pt = cubic_at(p0, *c1, *c2, *p, t);
                    points.push(pt);
                }
                cursor = *p;
            }
            PathCommand::QuadraticTo(cp, p) => {
                let p0 = cursor;
                for i in 1..=3 {
                    let t = i as f64 / 3.0;
                    let a = p0.lerp(*cp, t);
                    let b = cp.lerp(*p, t);
                    points.push(a.lerp(b, t));
                }
                cursor = *p;
            }
            PathCommand::Close => {
                // Close doesn't add a new point.
            }
        }
    }

    points
}

fn cubic_at(p0: MorphPoint, c1: MorphPoint, c2: MorphPoint, p3: MorphPoint, t: f64) -> MorphPoint {
    let a = p0.lerp(c1, t);
    let b = c1.lerp(c2, t);
    let c = c2.lerp(p3, t);
    let d = a.lerp(b, t);
    let e = b.lerp(c, t);
    d.lerp(e, t)
}

/// Convert a point list back to an SVG path string (M + L commands).
pub fn points_to_svg_path(points: &[MorphPoint], closed: bool) -> String {
    if points.is_empty() {
        return String::new();
    }

    let mut d = format!("M {}", points[0]);
    for p in &points[1..] {
        d.push_str(&format!(" L {p}"));
    }
    if closed {
        d.push_str(" Z");
    }
    d
}

// ── Normalize ──────────────────────────────────────────────────

/// Normalize two point lists to the same length by subdividing the shorter one.
pub fn normalize_point_lists(
    a: &[MorphPoint],
    b: &[MorphPoint],
) -> (Vec<MorphPoint>, Vec<MorphPoint>) {
    let target_len = a.len().max(b.len());
    (
        resample_points(a, target_len),
        resample_points(b, target_len),
    )
}

/// Resample a point list to exactly `n` points via linear interpolation.
pub fn resample_points(points: &[MorphPoint], n: usize) -> Vec<MorphPoint> {
    if points.is_empty() || n == 0 {
        return Vec::new();
    }
    if points.len() == 1 || n == 1 {
        return vec![points[0]; n];
    }

    // Compute cumulative arc lengths.
    let mut lengths = vec![0.0_f64];
    for i in 1..points.len() {
        lengths.push(lengths[i - 1] + points[i - 1].distance_to(points[i]));
    }
    let total = *lengths.last().unwrap();

    if total < 1e-12 {
        return vec![points[0]; n];
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let target_dist = if n == 1 { 0.0 } else { total * i as f64 / (n - 1) as f64 };

        // Find the segment.
        let seg = match lengths.binary_search_by(|d| d.partial_cmp(&target_dist).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };

        if seg >= points.len() - 1 {
            result.push(*points.last().unwrap());
        } else {
            let seg_len = lengths[seg + 1] - lengths[seg];
            let t = if seg_len > 0.0 {
                (target_dist - lengths[seg]) / seg_len
            } else {
                0.0
            };
            result.push(points[seg].lerp(points[seg + 1], t));
        }
    }

    result
}

// ── Morph ──────────────────────────────────────────────────────

/// Interpolate between two normalized point lists at parameter t [0, 1].
pub fn interpolate_points(
    from: &[MorphPoint],
    to: &[MorphPoint],
    t: f64,
) -> Vec<MorphPoint> {
    let t = t.clamp(0.0, 1.0);
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| a.lerp(*b, t))
        .collect()
}

/// Interpolate with easing (quadratic ease-in-out).
pub fn interpolate_eased(
    from: &[MorphPoint],
    to: &[MorphPoint],
    t: f64,
) -> Vec<MorphPoint> {
    let t = t.clamp(0.0, 1.0);
    let eased = if t < 0.5 {
        2.0 * t * t
    } else {
        -1.0 + (4.0 - 2.0 * t) * t
    };
    interpolate_points(from, to, eased)
}

// ── Polygon Morphing ───────────────────────────────────────────

/// Match polygon vertices by angle from centroid, then morph.
pub fn polygon_morph(
    from: &[MorphPoint],
    to: &[MorphPoint],
    t: f64,
) -> Vec<MorphPoint> {
    let from_sorted = sort_by_angle(from);
    let to_sorted = sort_by_angle(to);
    let (norm_from, norm_to) = normalize_point_lists(&from_sorted, &to_sorted);
    interpolate_points(&norm_from, &norm_to, t)
}

fn centroid(points: &[MorphPoint]) -> MorphPoint {
    if points.is_empty() {
        return MorphPoint::new(0.0, 0.0);
    }
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.x).sum();
    let sy: f64 = points.iter().map(|p| p.y).sum();
    MorphPoint::new(sx / n, sy / n)
}

fn sort_by_angle(points: &[MorphPoint]) -> Vec<MorphPoint> {
    let c = centroid(points);
    let mut pts: Vec<MorphPoint> = points.to_vec();
    pts.sort_by(|a, b| {
        let angle_a = (a.y - c.y).atan2(a.x - c.x);
        let angle_b = (b.y - c.y).atan2(b.x - c.x);
        angle_a.partial_cmp(&angle_b).unwrap()
    });
    pts
}

// ── Full Morph Helpers ─────────────────────────────────────────

/// Parse two SVG path strings, normalize, and morph at t.
/// Returns the interpolated SVG path string.
pub fn morph_svg_paths(from_d: &str, to_d: &str, t: f64, closed: bool) -> String {
    let from_cmds = parse_svg_path(from_d);
    let to_cmds = parse_svg_path(to_d);
    let from_pts = commands_to_points(&from_cmds);
    let to_pts = commands_to_points(&to_cmds);
    let (norm_from, norm_to) = normalize_point_lists(&from_pts, &to_pts);
    let morphed = interpolate_points(&norm_from, &norm_to, t);
    points_to_svg_path(&morphed, closed)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_move_and_lines() {
        let cmds = parse_svg_path("M 0 0 L 100 0 L 100 100 Z");
        assert_eq!(cmds.len(), 4);
        assert!(matches!(cmds[0], PathCommand::MoveTo(p) if (p.x - 0.0).abs() < 1e-10));
        assert!(matches!(cmds[1], PathCommand::LineTo(p) if (p.x - 100.0).abs() < 1e-10));
        assert!(matches!(cmds[3], PathCommand::Close));
    }

    #[test]
    fn parse_relative_commands() {
        let cmds = parse_svg_path("m 10 20 l 30 0 l 0 40 z");
        assert_eq!(cmds.len(), 4);
        match &cmds[0] {
            PathCommand::MoveTo(p) => {
                assert!((p.x - 10.0).abs() < 1e-10);
                assert!((p.y - 20.0).abs() < 1e-10);
            }
            other => panic!("Expected MoveTo, got {other:?}"),
        }
        match &cmds[1] {
            PathCommand::LineTo(p) => {
                assert!((p.x - 40.0).abs() < 1e-10);
                assert!((p.y - 20.0).abs() < 1e-10);
            }
            other => panic!("Expected LineTo, got {other:?}"),
        }
    }

    #[test]
    fn parse_horizontal_vertical() {
        let cmds = parse_svg_path("M 0 0 H 50 V 50");
        assert_eq!(cmds.len(), 3);
        assert!(matches!(cmds[1], PathCommand::HorizontalTo(x) if (x - 50.0).abs() < 1e-10));
        assert!(matches!(cmds[2], PathCommand::VerticalTo(y) if (y - 50.0).abs() < 1e-10));
    }

    #[test]
    fn parse_cubic() {
        let cmds = parse_svg_path("M 0 0 C 10 20 30 40 50 60");
        assert_eq!(cmds.len(), 2);
        match &cmds[1] {
            PathCommand::CubicTo(c1, c2, p) => {
                assert!((c1.x - 10.0).abs() < 1e-10);
                assert!((c2.y - 40.0).abs() < 1e-10);
                assert!((p.x - 50.0).abs() < 1e-10);
            }
            other => panic!("Expected CubicTo, got {other:?}"),
        }
    }

    #[test]
    fn parse_quadratic() {
        let cmds = parse_svg_path("M 0 0 Q 50 100 100 0");
        assert_eq!(cmds.len(), 2);
        assert!(matches!(&cmds[1], PathCommand::QuadraticTo(_, _)));
    }

    #[test]
    fn commands_to_points_basic() {
        let cmds = parse_svg_path("M 0 0 L 100 0 L 100 100");
        let pts = commands_to_points(&cmds);
        assert_eq!(pts.len(), 3);
        assert!((pts[2].y - 100.0).abs() < 1e-10);
    }

    #[test]
    fn points_to_path_string() {
        let pts = vec![
            MorphPoint::new(0.0, 0.0),
            MorphPoint::new(100.0, 0.0),
            MorphPoint::new(100.0, 100.0),
        ];
        let d = points_to_svg_path(&pts, true);
        assert!(d.starts_with("M 0.0000,0.0000"));
        assert!(d.ends_with(" Z"));
    }

    #[test]
    fn normalize_makes_equal_length() {
        let a = vec![MorphPoint::new(0.0, 0.0), MorphPoint::new(10.0, 0.0)];
        let b = vec![
            MorphPoint::new(0.0, 0.0),
            MorphPoint::new(5.0, 0.0),
            MorphPoint::new(10.0, 0.0),
            MorphPoint::new(15.0, 0.0),
        ];
        let (na, nb) = normalize_point_lists(&a, &b);
        assert_eq!(na.len(), nb.len());
        assert_eq!(na.len(), 4);
    }

    #[test]
    fn interpolate_at_zero_is_from() {
        let from = vec![MorphPoint::new(0.0, 0.0), MorphPoint::new(10.0, 0.0)];
        let to = vec![MorphPoint::new(50.0, 50.0), MorphPoint::new(60.0, 50.0)];
        let result = interpolate_points(&from, &to, 0.0);
        assert!((result[0].x - 0.0).abs() < 1e-10);
        assert!((result[1].x - 10.0).abs() < 1e-10);
    }

    #[test]
    fn interpolate_at_one_is_to() {
        let from = vec![MorphPoint::new(0.0, 0.0)];
        let to = vec![MorphPoint::new(100.0, 200.0)];
        let result = interpolate_points(&from, &to, 1.0);
        assert!((result[0].x - 100.0).abs() < 1e-10);
        assert!((result[0].y - 200.0).abs() < 1e-10);
    }

    #[test]
    fn interpolate_midpoint() {
        let from = vec![MorphPoint::new(0.0, 0.0)];
        let to = vec![MorphPoint::new(100.0, 0.0)];
        let result = interpolate_points(&from, &to, 0.5);
        assert!((result[0].x - 50.0).abs() < 1e-10);
    }

    #[test]
    fn eased_interpolation() {
        let from = vec![MorphPoint::new(0.0, 0.0)];
        let to = vec![MorphPoint::new(100.0, 0.0)];
        let result = interpolate_eased(&from, &to, 0.5);
        // ease-in-out at 0.5 should be 0.5.
        assert!((result[0].x - 50.0).abs() < 0.01);
    }

    #[test]
    fn polygon_morph_works() {
        let triangle = vec![
            MorphPoint::new(50.0, 0.0),
            MorphPoint::new(100.0, 100.0),
            MorphPoint::new(0.0, 100.0),
        ];
        let square = vec![
            MorphPoint::new(0.0, 0.0),
            MorphPoint::new(100.0, 0.0),
            MorphPoint::new(100.0, 100.0),
            MorphPoint::new(0.0, 100.0),
        ];
        let result = polygon_morph(&triangle, &square, 0.5);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn morph_svg_paths_end_to_end() {
        let from = "M 0 0 L 100 0 L 100 100 Z";
        let to = "M 50 50 L 150 50 L 150 150 Z";
        let mid = morph_svg_paths(from, to, 0.5, true);
        assert!(mid.starts_with("M "));
        assert!(mid.ends_with(" Z"));
    }

    #[test]
    fn resample_preserves_endpoints() {
        let pts = vec![
            MorphPoint::new(0.0, 0.0),
            MorphPoint::new(50.0, 0.0),
            MorphPoint::new(100.0, 0.0),
        ];
        let resampled = resample_points(&pts, 5);
        assert_eq!(resampled.len(), 5);
        assert!((resampled[0].x - 0.0).abs() < 0.01);
        assert!((resampled[4].x - 100.0).abs() < 0.01);
    }
}
