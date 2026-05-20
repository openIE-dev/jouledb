//! 3D scatter plot with isometric and perspective projection, view rotation,
//! grid planes, and marker sizing.  Pure Rust SVG output.

use std::f64::consts::PI;

// ── Data types ───────────────────────────────────────────────────

/// A point in 3D space.
#[derive(Debug, Clone)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub size: f64,
    pub color: String,
    pub label: Option<String>,
    /// Optional group identifier for cluster coloring.
    pub group: Option<usize>,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            size: 4.0,
            color: "#3498db".into(),
            label: None,
            group: None,
        }
    }

    pub fn with_size(mut self, size: f64) -> Self {
        self.size = size;
        self
    }

    pub fn with_color(mut self, color: impl Into<String>) -> Self {
        self.color = color.into();
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_group(mut self, group: usize) -> Self {
        self.group = Some(group);
        self
    }
}

/// Axis configuration for one dimension.
#[derive(Debug, Clone)]
pub struct Axis3D {
    pub label: String,
    pub min: f64,
    pub max: f64,
    pub ticks: usize,
}

impl Axis3D {
    pub fn new(label: impl Into<String>, min: f64, max: f64, ticks: usize) -> Self {
        Self {
            label: label.into(),
            min,
            max,
            ticks,
        }
    }

    pub fn normalize(&self, value: f64) -> f64 {
        let range = (self.max - self.min).max(f64::EPSILON);
        (value - self.min) / range
    }
}

// ── Camera / view ────────────────────────────────────────────────

/// Camera view parameters.
#[derive(Debug, Clone, Copy)]
pub struct View3D {
    /// Yaw rotation in degrees (around Y axis).
    pub yaw_deg: f64,
    /// Pitch rotation in degrees (around X axis).
    pub pitch_deg: f64,
    /// Perspective focal length (0 = isometric).
    pub focal_length: f64,
}

impl Default for View3D {
    fn default() -> Self {
        Self {
            yaw_deg: 30.0,
            pitch_deg: 20.0,
            focal_length: 0.0, // isometric
        }
    }
}

impl View3D {
    pub fn perspective(yaw_deg: f64, pitch_deg: f64, focal_length: f64) -> Self {
        Self {
            yaw_deg,
            pitch_deg,
            focal_length,
        }
    }

    pub fn isometric(yaw_deg: f64, pitch_deg: f64) -> Self {
        Self {
            yaw_deg,
            pitch_deg,
            focal_length: 0.0,
        }
    }
}

// ── Rotation & projection ────────────────────────────────────────

/// Rotate a 3D point by yaw (Y-axis) then pitch (X-axis).
pub fn rotate(x: f64, y: f64, z: f64, yaw_deg: f64, pitch_deg: f64) -> (f64, f64, f64) {
    let yaw = yaw_deg.to_radians();
    let pitch = pitch_deg.to_radians();

    // Yaw (around Y)
    let x1 = x * yaw.cos() + z * yaw.sin();
    let y1 = y;
    let z1 = -x * yaw.sin() + z * yaw.cos();

    // Pitch (around X)
    let x2 = x1;
    let y2 = y1 * pitch.cos() - z1 * pitch.sin();
    let z2 = y1 * pitch.sin() + z1 * pitch.cos();

    (x2, y2, z2)
}

/// Projected 2D point with depth for z-sorting.
#[derive(Debug, Clone)]
pub struct Projected {
    pub screen_x: f64,
    pub screen_y: f64,
    pub depth: f64,
    pub index: usize,
}

/// Project a 3D point to 2D screen coordinates.
pub fn project(
    x: f64,
    y: f64,
    z: f64,
    view: &View3D,
    screen_cx: f64,
    screen_cy: f64,
    scale: f64,
) -> (f64, f64, f64) {
    let (rx, ry, rz) = rotate(x, y, z, view.yaw_deg, view.pitch_deg);

    if view.focal_length > 0.0 {
        // Perspective
        let d = view.focal_length;
        let factor = d / (d + rz);
        (
            screen_cx + rx * scale * factor,
            screen_cy - ry * scale * factor,
            rz,
        )
    } else {
        // Isometric
        (screen_cx + rx * scale, screen_cy - ry * scale, rz)
    }
}

// ── Grid planes ──────────────────────────────────────────────────

/// Which grid plane to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPlane {
    XY,
    XZ,
    YZ,
}

/// Generate SVG lines for a grid plane.
pub fn render_grid_plane(
    plane: GridPlane,
    divisions: usize,
    view: &View3D,
    screen_cx: f64,
    screen_cy: f64,
    scale: f64,
) -> String {
    let mut svg = String::new();
    let n = divisions.max(1);
    let step = 1.0 / n as f64;

    for i in 0..=n {
        let t = i as f64 * step;
        let (x1, y1, z1, x2, y2, z2) = match plane {
            GridPlane::XY => (t, 0.0, 0.0, t, 1.0, 0.0),
            GridPlane::XZ => (t, 0.0, 0.0, t, 0.0, 1.0),
            GridPlane::YZ => (0.0, t, 0.0, 0.0, t, 1.0),
        };
        let (sx1, sy1, _) = project(x1, y1, z1, view, screen_cx, screen_cy, scale);
        let (sx2, sy2, _) = project(x2, y2, z2, view, screen_cx, screen_cy, scale);
        svg.push_str(&format!(
            "<line x1=\"{sx1}\" y1=\"{sy1}\" x2=\"{sx2}\" y2=\"{sy2}\" stroke=\"#ddd\" stroke-width=\"0.5\" />"
        ));

        // Perpendicular lines
        let (x3, y3, z3, x4, y4, z4) = match plane {
            GridPlane::XY => (0.0, t, 0.0, 1.0, t, 0.0),
            GridPlane::XZ => (0.0, 0.0, t, 1.0, 0.0, t),
            GridPlane::YZ => (0.0, 0.0, t, 0.0, 1.0, t),
        };
        let (sx3, sy3, _) = project(x3, y3, z3, view, screen_cx, screen_cy, scale);
        let (sx4, sy4, _) = project(x4, y4, z4, view, screen_cx, screen_cy, scale);
        svg.push_str(&format!(
            "<line x1=\"{sx3}\" y1=\"{sy3}\" x2=\"{sx4}\" y2=\"{sy4}\" stroke=\"#ddd\" stroke-width=\"0.5\" />"
        ));
    }
    svg
}

// ── Cluster coloring ─────────────────────────────────────────────

const GROUP_COLORS: &[&str] = &[
    "#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
];

/// Pick a color for a group index.
pub fn group_color(group: usize) -> &'static str {
    GROUP_COLORS[group % GROUP_COLORS.len()]
}

// ── Config ───────────────────────────────────────────────────────

/// Configuration for a 3D scatter chart.
#[derive(Debug, Clone)]
pub struct Scatter3DConfig {
    pub width: f64,
    pub height: f64,
    pub view: View3D,
    pub x_axis: Axis3D,
    pub y_axis: Axis3D,
    pub z_axis: Axis3D,
    pub show_grid_xy: bool,
    pub show_grid_xz: bool,
    pub show_grid_yz: bool,
    pub grid_divisions: usize,
    pub font_size: f64,
}

impl Default for Scatter3DConfig {
    fn default() -> Self {
        Self {
            width: 500.0,
            height: 400.0,
            view: View3D::default(),
            x_axis: Axis3D::new("X", 0.0, 10.0, 5),
            y_axis: Axis3D::new("Y", 0.0, 10.0, 5),
            z_axis: Axis3D::new("Z", 0.0, 10.0, 5),
            show_grid_xy: true,
            show_grid_xz: false,
            show_grid_yz: false,
            grid_divisions: 5,
            font_size: 11.0,
        }
    }
}

// ── Rendering ────────────────────────────────────────────────────

/// Render a 3D scatter chart as SVG using painter's algorithm (z-sort).
pub fn render_scatter3d(points: &[Point3D], cfg: &Scatter3DConfig) -> String {
    let screen_cx = cfg.width / 2.0;
    let screen_cy = cfg.height / 2.0;
    let scale = cfg.width.min(cfg.height) * 0.3;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    // Grid planes
    if cfg.show_grid_xy {
        svg.push_str(&render_grid_plane(GridPlane::XY, cfg.grid_divisions, &cfg.view, screen_cx, screen_cy, scale));
    }
    if cfg.show_grid_xz {
        svg.push_str(&render_grid_plane(GridPlane::XZ, cfg.grid_divisions, &cfg.view, screen_cx, screen_cy, scale));
    }
    if cfg.show_grid_yz {
        svg.push_str(&render_grid_plane(GridPlane::YZ, cfg.grid_divisions, &cfg.view, screen_cx, screen_cy, scale));
    }

    // Axis lines
    let axes = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), &cfg.x_axis.label),
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), &cfg.y_axis.label),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), &cfg.z_axis.label),
    ];
    for ((ox, oy, oz), (ex, ey, ez), label) in &axes {
        let (sx1, sy1, _) = project(*ox, *oy, *oz, &cfg.view, screen_cx, screen_cy, scale);
        let (sx2, sy2, _) = project(*ex, *ey, *ez, &cfg.view, screen_cx, screen_cy, scale);
        svg.push_str(&format!(
            "<line x1=\"{sx1}\" y1=\"{sy1}\" x2=\"{sx2}\" y2=\"{sy2}\" stroke=\"#666\" stroke-width=\"1\" />"
        ));
        let fs = cfg.font_size;
        svg.push_str(&format!(
            "<text x=\"{sx2}\" y=\"{}\" font-size=\"{fs}\" fill=\"#333\">{label}</text>",
            sy2 - 5.0
        ));
    }

    // Project all points and z-sort (painter's algorithm — furthest first)
    let mut projected: Vec<Projected> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let nx = cfg.x_axis.normalize(p.x);
            let ny = cfg.y_axis.normalize(p.y);
            let nz = cfg.z_axis.normalize(p.z);
            let (sx, sy, depth) = project(nx, ny, nz, &cfg.view, screen_cx, screen_cy, scale);
            Projected {
                screen_x: sx,
                screen_y: sy,
                depth,
                index: i,
            }
        })
        .collect();

    projected.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));

    // Draw points
    for proj in &projected {
        let pt = &points[proj.index];
        let color = match pt.group {
            Some(g) => group_color(g),
            None => pt.color.as_str(),
        };
        let r = pt.size;
        svg.push_str(&format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"{r}\" fill=\"{color}\" fill-opacity=\"0.8\" />",
            proj.screen_x, proj.screen_y
        ));
        if let Some(ref label) = pt.label {
            let fs = cfg.font_size - 2.0;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" font-size=\"{fs}\" fill=\"#333\">{label}</text>",
                proj.screen_x + r + 2.0,
                proj.screen_y - 2.0
            ));
        }
    }

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_identity() {
        let (x, y, z) = rotate(1.0, 2.0, 3.0, 0.0, 0.0);
        assert!((x - 1.0).abs() < 1e-9);
        assert!((y - 2.0).abs() < 1e-9);
        assert!((z - 3.0).abs() < 1e-9);
    }

    #[test]
    fn rotate_yaw_90() {
        let (x, _y, z) = rotate(1.0, 0.0, 0.0, 90.0, 0.0);
        assert!(x.abs() < 1e-9);
        assert!((z - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn rotate_pitch_90() {
        let (_x, y, z) = rotate(0.0, 1.0, 0.0, 0.0, 90.0);
        assert!(y.abs() < 1e-9);
        assert!((z - 1.0).abs() < 1e-9);
    }

    #[test]
    fn project_isometric_center() {
        let view = View3D::isometric(0.0, 0.0);
        let (sx, sy, _) = project(0.0, 0.0, 0.0, &view, 100.0, 100.0, 50.0);
        assert!((sx - 100.0).abs() < 1e-9);
        assert!((sy - 100.0).abs() < 1e-9);
    }

    #[test]
    fn project_perspective_shrinks_distant() {
        let view = View3D::perspective(0.0, 0.0, 500.0);
        let (_, _, _) = project(1.0, 0.0, 0.0, &view, 200.0, 200.0, 100.0);
        // Just verify it doesn't panic; perspective projection is non-trivial
        let (sx_near, _, _) = project(1.0, 0.0, 0.0, &view, 200.0, 200.0, 100.0);
        let (sx_far, _, _) = project(1.0, 0.0, 10.0, &view, 200.0, 200.0, 100.0);
        // Near point should be further from center
        assert!((sx_near - 200.0).abs() >= (sx_far - 200.0).abs());
    }

    #[test]
    fn axis3d_normalize() {
        let axis = Axis3D::new("X", 10.0, 20.0, 5);
        assert!((axis.normalize(10.0)).abs() < 1e-9);
        assert!((axis.normalize(15.0) - 0.5).abs() < 1e-9);
        assert!((axis.normalize(20.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn group_color_wraps() {
        let c0 = group_color(0);
        let c10 = group_color(10);
        assert_eq!(c0, c10); // wraps around
    }

    #[test]
    fn render_scatter3d_empty() {
        let cfg = Scatter3DConfig::default();
        let svg = render_scatter3d(&[], &cfg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(!svg.contains("fill-opacity"));
    }

    #[test]
    fn render_scatter3d_basic() {
        let points = vec![
            Point3D::new(1.0, 2.0, 3.0).with_color("#f00"),
            Point3D::new(5.0, 5.0, 5.0).with_label("Center"),
            Point3D::new(9.0, 8.0, 7.0).with_group(2),
        ];
        let cfg = Scatter3DConfig::default();
        let svg = render_scatter3d(&points, &cfg);
        assert!(svg.contains("<circle"));
        assert!(svg.contains("Center"));
        assert_eq!(svg.matches("<circle").count(), 3);
    }

    #[test]
    fn render_scatter3d_with_grid() {
        let cfg = Scatter3DConfig {
            show_grid_xy: true,
            show_grid_xz: true,
            show_grid_yz: true,
            ..Scatter3DConfig::default()
        };
        let svg = render_scatter3d(&[], &cfg);
        assert!(svg.contains("<line"));
    }

    #[test]
    fn render_scatter3d_axis_labels() {
        let cfg = Scatter3DConfig::default();
        let svg = render_scatter3d(&[], &cfg);
        assert!(svg.contains("X"));
        assert!(svg.contains("Y"));
        assert!(svg.contains("Z"));
    }

    #[test]
    fn point3d_builder() {
        let p = Point3D::new(1.0, 2.0, 3.0)
            .with_size(8.0)
            .with_color("#abc")
            .with_label("test")
            .with_group(3);
        assert!((p.size - 8.0).abs() < 1e-9);
        assert_eq!(p.color, "#abc");
        assert_eq!(p.label.as_deref(), Some("test"));
        assert_eq!(p.group, Some(3));
    }

    #[test]
    fn render_grid_plane_produces_lines() {
        let view = View3D::default();
        let svg = render_grid_plane(GridPlane::XY, 3, &view, 200.0, 200.0, 100.0);
        assert!(svg.contains("<line"));
        // 4 lines per direction * 2 directions = 8 minimum
        assert!(svg.matches("<line").count() >= 8);
    }
}
