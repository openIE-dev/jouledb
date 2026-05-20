//! 3D surface, contour, and mesh visualization.
//!
//! Replaces MATLAB's surf/mesh/contour, Julia's Plots.surface,
//! Python's mpl_toolkits.mplot3d and plotly.graph_objects.Surface.
//!
//! Renders to SVG using isometric/perspective projection with
//! painter's algorithm for depth sorting. Also outputs raw vertex
//! data for WebGPU rendering paths.

use std::f64::consts::PI;

// ── Data types ─────────────────────────────────────────────────────

/// A 2D grid of Z values over an X-Y domain.
#[derive(Debug, Clone)]
pub struct SurfaceData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<Vec<f64>>,   // z[iy][ix]
}

impl SurfaceData {
    /// Create from a function f(x, y) evaluated over a grid.
    pub fn from_fn(
        x_range: (f64, f64),
        y_range: (f64, f64),
        nx: usize,
        ny: usize,
        f: impl Fn(f64, f64) -> f64,
    ) -> Self {
        let x: Vec<f64> = (0..nx).map(|i| x_range.0 + (x_range.1 - x_range.0) * i as f64 / (nx - 1) as f64).collect();
        let y: Vec<f64> = (0..ny).map(|j| y_range.0 + (y_range.1 - y_range.0) * j as f64 / (ny - 1) as f64).collect();
        let z: Vec<Vec<f64>> = y.iter().map(|&yv| x.iter().map(|&xv| f(xv, yv)).collect()).collect();
        Self { x, y, z }
    }

    pub fn z_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for row in &self.z {
            for &v in row {
                if v < min { min = v; }
                if v > max { max = v; }
            }
        }
        (min, max)
    }
}

/// Contour level specification.
#[derive(Debug, Clone)]
pub struct ContourLevel {
    pub value: f64,
    pub color: String,
    pub label: Option<String>,
}

/// 3D projection parameters.
#[derive(Debug, Clone)]
pub struct View3D {
    pub azimuth: f64,      // degrees, rotation around Z
    pub elevation: f64,    // degrees, tilt from XY plane
    pub distance: f64,     // camera distance
    pub width: f64,
    pub height: f64,
}

impl Default for View3D {
    fn default() -> Self {
        Self {
            azimuth: -60.0,
            elevation: 30.0,
            distance: 2.0,
            width: 600.0,
            height: 500.0,
        }
    }
}

/// Surface plot configuration.
#[derive(Debug, Clone)]
pub struct SurfaceConfig {
    pub view: View3D,
    pub title: Option<String>,
    pub colormap: Colormap,
    pub show_wireframe: bool,
    pub show_colorbar: bool,
    pub wireframe_color: String,
    pub opacity: f64,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub z_label: Option<String>,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            view: View3D::default(),
            title: None,
            colormap: Colormap::Viridis,
            show_wireframe: true,
            show_colorbar: true,
            wireframe_color: "#333".to_string(),
            opacity: 0.9,
            x_label: None,
            y_label: None,
            z_label: None,
        }
    }
}

// ── Colormaps ──────────────────────────────────────────────────────

/// Scientific colormaps (perceptually uniform).
#[derive(Debug, Clone, Copy)]
pub enum Colormap {
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Cividis,
    Turbo,
    Hot,
    Cool,
    Jet,
    Grayscale,
    BlueRed,
    GreenYellow,
}

impl Colormap {
    /// Map a normalized value [0, 1] to an RGB hex color.
    pub fn color(&self, t: f64) -> String {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => viridis(t),
            Colormap::Plasma => plasma(t),
            Colormap::Inferno => inferno(t),
            Colormap::Magma => magma(t),
            Colormap::Cividis => cividis(t),
            Colormap::Turbo => turbo(t),
            Colormap::Hot => hot(t),
            Colormap::Cool => cool(t),
            Colormap::Jet => jet(t),
            Colormap::Grayscale => {
                let v = (t * 255.0) as u8;
                format!("#{:02x}{:02x}{:02x}", v, v, v)
            }
            Colormap::BlueRed => lerp_color(t, (0, 0, 200), (200, 0, 0)),
            Colormap::GreenYellow => lerp_color(t, (0, 128, 0), (255, 255, 0)),
        }
    }

    /// Generate N evenly spaced colors from this colormap.
    pub fn palette(&self, n: usize) -> Vec<String> {
        (0..n).map(|i| self.color(i as f64 / (n.max(1) - 1).max(1) as f64)).collect()
    }
}

// ── 3D Projection ──────────────────────────────────────────────────

/// Project a 3D point to 2D screen coordinates.
fn project(x: f64, y: f64, z: f64, view: &View3D) -> (f64, f64, f64) {
    let az = view.azimuth.to_radians();
    let el = view.elevation.to_radians();

    // Rotate around Z axis (azimuth)
    let x1 = x * az.cos() - y * az.sin();
    let y1 = x * az.sin() + y * az.cos();
    let z1 = z;

    // Rotate around X axis (elevation)
    let y2 = y1 * el.cos() - z1 * el.sin();
    let z2 = y1 * el.sin() + z1 * el.cos();

    // Simple perspective
    let scale = view.distance / (view.distance + y2);
    let sx = view.width / 2.0 + x1 * scale * view.width * 0.3;
    let sy = view.height / 2.0 - z2 * scale * view.height * 0.3;

    (sx, sy, y2) // y2 is depth for sorting
}

// ── Surface rendering ──────────────────────────────────────────────

/// Render a 3D surface plot to SVG.
pub fn surface_svg(data: &SurfaceData, config: &SurfaceConfig) -> String {
    let (z_min, z_max) = data.z_range();
    let z_range = (z_max - z_min).max(1e-10);

    let nx = data.x.len();
    let ny = data.y.len();
    let x_min = data.x[0];
    let x_max = data.x[nx - 1];
    let y_min = data.y[0];
    let y_max = data.y[ny - 1];
    let x_range = (x_max - x_min).max(1e-10);
    let y_range = (y_max - y_min).max(1e-10);

    // Build quads with depth for painter's algorithm
    let mut quads: Vec<(f64, String)> = Vec::new(); // (depth, svg_polygon)

    for iy in 0..ny - 1 {
        for ix in 0..nx - 1 {
            let corners = [
                (data.x[ix], data.y[iy], data.z[iy][ix]),
                (data.x[ix + 1], data.y[iy], data.z[iy][ix + 1]),
                (data.x[ix + 1], data.y[iy + 1], data.z[iy + 1][ix + 1]),
                (data.x[ix], data.y[iy + 1], data.z[iy + 1][ix]),
            ];

            // Normalize to [-1, 1]
            let norm_corners: Vec<(f64, f64, f64)> = corners.iter().map(|&(cx, cy, cz)| {
                let nx = (cx - x_min) / x_range * 2.0 - 1.0;
                let ny = (cy - y_min) / y_range * 2.0 - 1.0;
                let nz = (cz - z_min) / z_range * 2.0 - 1.0;
                (nx, ny, nz)
            }).collect();

            // Project to screen
            let projected: Vec<(f64, f64, f64)> = norm_corners.iter()
                .map(|&(nx, ny, nz)| project(nx, ny, nz, &config.view))
                .collect();

            let avg_depth = projected.iter().map(|p| p.2).sum::<f64>() / 4.0;
            let avg_z = corners.iter().map(|c| c.2).sum::<f64>() / 4.0;
            let t = (avg_z - z_min) / z_range;
            let fill = config.colormap.color(t);

            let points: String = projected.iter()
                .map(|(sx, sy, _)| format!("{:.1},{:.1}", sx, sy))
                .collect::<Vec<_>>()
                .join(" ");

            let wireframe = if config.show_wireframe {
                format!(" stroke=\"{}\" stroke-width=\"0.5\"", config.wireframe_color)
            } else {
                String::new()
            };

            quads.push((avg_depth, format!(
                "<polygon points=\"{points}\" fill=\"{fill}\" opacity=\"{:.2}\"{wireframe}/>",
                config.opacity
            )));
        }
    }

    // Sort by depth (painter's algorithm: far first)
    quads.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
        config.view.width, config.view.height, config.view.width, config.view.height
    );
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#fafafa\"/>");

    if let Some(title) = &config.title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{}</text>",
            config.view.width / 2.0, title
        ));
    }

    for (_, polygon) in &quads {
        svg.push_str(polygon);
    }

    // Colorbar
    if config.show_colorbar {
        svg.push_str(&colorbar_svg(
            config.view.width - 60.0, 40.0, 20.0, config.view.height - 80.0,
            z_min, z_max, &config.colormap,
        ));
    }

    svg.push_str("</svg>");
    svg
}

// ── Contour rendering ──────────────────────────────────────────────

/// Generate contour levels automatically.
pub fn auto_contour_levels(data: &SurfaceData, n_levels: usize, colormap: &Colormap) -> Vec<ContourLevel> {
    let (z_min, z_max) = data.z_range();
    let z_range = z_max - z_min;
    (0..n_levels).map(|i| {
        let t = i as f64 / (n_levels - 1).max(1) as f64;
        let value = z_min + t * z_range;
        ContourLevel {
            value,
            color: colormap.color(t),
            label: Some(format!("{:.2}", value)),
        }
    }).collect()
}

/// Render a contour plot to SVG using marching squares.
pub fn contour_svg(
    data: &SurfaceData,
    levels: &[ContourLevel],
    width: f64,
    height: f64,
    title: Option<&str>,
) -> String {
    let nx = data.x.len();
    let ny = data.y.len();
    let padding = 50.0;
    let plot_w = width - 2.0 * padding;
    let plot_h = height - 2.0 * padding;

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">"
    );
    svg.push_str(&format!("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"));

    if let Some(t) = title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{t}</text>",
            width / 2.0
        ));
    }

    // For each level, trace contour lines via marching squares
    for level in levels {
        let mut paths = Vec::new();

        for iy in 0..ny - 1 {
            for ix in 0..nx - 1 {
                let z00 = data.z[iy][ix];
                let z10 = data.z[iy][ix + 1];
                let z01 = data.z[iy + 1][ix];
                let z11 = data.z[iy + 1][ix + 1];

                let threshold = level.value;

                // Marching squares case index
                let case = ((z00 >= threshold) as u8)
                    | (((z10 >= threshold) as u8) << 1)
                    | (((z11 >= threshold) as u8) << 2)
                    | (((z01 >= threshold) as u8) << 3);

                if case == 0 || case == 15 { continue; }

                // Interpolate edge crossings
                let lerp = |a: f64, b: f64| -> f64 {
                    if (b - a).abs() < 1e-15 { 0.5 } else { (threshold - a) / (b - a) }
                };

                let x0 = padding + ix as f64 / (nx - 1) as f64 * plot_w;
                let x1 = padding + (ix + 1) as f64 / (nx - 1) as f64 * plot_w;
                let y0 = padding + iy as f64 / (ny - 1) as f64 * plot_h;
                let y1 = padding + (iy + 1) as f64 / (ny - 1) as f64 * plot_h;

                // Edge midpoints
                let top = (x0 + lerp(z00, z10) * (x1 - x0), y0);
                let bottom = (x0 + lerp(z01, z11) * (x1 - x0), y1);
                let left = (x0, y0 + lerp(z00, z01) * (y1 - y0));
                let right = (x1, y0 + lerp(z10, z11) * (y1 - y0));

                // Draw line segments based on case
                let segments: Vec<((f64, f64), (f64, f64))> = match case {
                    1 | 14 => vec![(left, top)],
                    2 | 13 => vec![(top, right)],
                    3 | 12 => vec![(left, right)],
                    4 | 11 => vec![(right, bottom)],
                    5 => vec![(left, top), (right, bottom)], // saddle
                    6 | 9 => vec![(top, bottom)],
                    7 | 8 => vec![(left, bottom)],
                    10 => vec![(top, right), (left, bottom)], // saddle
                    _ => vec![],
                };

                for ((x1s, y1s), (x2s, y2s)) in segments {
                    paths.push(format!(
                        "<line x1=\"{x1s:.1}\" y1=\"{y1s:.1}\" x2=\"{x2s:.1}\" y2=\"{y2s:.1}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                        level.color
                    ));
                }
            }
        }

        for path in &paths {
            svg.push_str(path);
        }
    }

    svg.push_str("</svg>");
    svg
}

// ── Wireframe rendering ────────────────────────────────────────────

/// Render a 3D wireframe (mesh) plot to SVG.
pub fn wireframe_svg(data: &SurfaceData, view: &View3D, color: &str, title: Option<&str>) -> String {
    let (z_min, z_max) = data.z_range();
    let z_range = (z_max - z_min).max(1e-10);
    let nx = data.x.len();
    let ny = data.y.len();
    let x_min = data.x[0];
    let x_max = data.x[nx - 1];
    let y_min = data.y[0];
    let y_max = data.y[ny - 1];
    let x_range = (x_max - x_min).max(1e-10);
    let y_range = (y_max - y_min).max(1e-10);

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">",
        view.width, view.height
    );
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    if let Some(t) = title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{t}</text>",
            view.width / 2.0
        ));
    }

    // Draw grid lines in X direction
    for iy in 0..ny {
        let mut path = String::new();
        for ix in 0..nx {
            let nx = (data.x[ix] - x_min) / x_range * 2.0 - 1.0;
            let ny_norm = (data.y[iy] - y_min) / y_range * 2.0 - 1.0;
            let nz = (data.z[iy][ix] - z_min) / z_range * 2.0 - 1.0;
            let (sx, sy, _) = project(nx, ny_norm, nz, view);
            if ix == 0 {
                path.push_str(&format!("M{sx:.1},{sy:.1}"));
            } else {
                path.push_str(&format!(" L{sx:.1},{sy:.1}"));
            }
        }
        svg.push_str(&format!("<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"0.8\"/>"));
    }

    // Draw grid lines in Y direction
    for ix in 0..nx {
        let mut path = String::new();
        for iy in 0..ny {
            let nx = (data.x[ix] - x_min) / x_range * 2.0 - 1.0;
            let ny_norm = (data.y[iy] - y_min) / y_range * 2.0 - 1.0;
            let nz = (data.z[iy][ix] - z_min) / z_range * 2.0 - 1.0;
            let (sx, sy, _) = project(nx, ny_norm, nz, view);
            if iy == 0 {
                path.push_str(&format!("M{sx:.1},{sy:.1}"));
            } else {
                path.push_str(&format!(" L{sx:.1},{sy:.1}"));
            }
        }
        svg.push_str(&format!("<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"0.8\"/>"));
    }

    svg.push_str("</svg>");
    svg
}

// ── Colorbar ───────────────────────────────────────────────────────

fn colorbar_svg(x: f64, y: f64, w: f64, h: f64, min: f64, max: f64, cmap: &Colormap) -> String {
    let n = 64;
    let step = h / n as f64;
    let mut svg = String::new();

    for i in 0..n {
        let t = 1.0 - i as f64 / (n - 1) as f64;
        let color = cmap.color(t);
        let cy = y + i as f64 * step;
        svg.push_str(&format!(
            "<rect x=\"{x}\" y=\"{cy:.1}\" width=\"{w}\" height=\"{:.1}\" fill=\"{color}\"/>",
            step + 0.5
        ));
    }

    svg.push_str(&format!(
        "<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" fill=\"none\" stroke=\"#333\" stroke-width=\"0.5\"/>"
    ));

    // Labels
    let n_labels = 5;
    for i in 0..n_labels {
        let t = i as f64 / (n_labels - 1) as f64;
        let val = max - t * (max - min);
        let ly = y + t * h;
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{:.1}\" font-size=\"9\" dominant-baseline=\"middle\">{:.2}</text>",
            x + w + 4.0, ly, val
        ));
    }

    svg
}

// ── Colormap implementations ───────────────────────────────────────

fn lerp_color(t: f64, c0: (u8, u8, u8), c1: (u8, u8, u8)) -> String {
    let r = (c0.0 as f64 + t * (c1.0 as f64 - c0.0 as f64)) as u8;
    let g = (c0.1 as f64 + t * (c1.1 as f64 - c0.1 as f64)) as u8;
    let b = (c0.2 as f64 + t * (c1.2 as f64 - c0.2 as f64)) as u8;
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

fn lerp3(t: f64, stops: &[(f64, (u8, u8, u8))]) -> String {
    if stops.is_empty() { return "#000000".to_string(); }
    if t <= stops[0].0 { return format!("#{:02x}{:02x}{:02x}", stops[0].1.0, stops[0].1.1, stops[0].1.2); }
    for i in 1..stops.len() {
        if t <= stops[i].0 {
            let frac = (t - stops[i - 1].0) / (stops[i].0 - stops[i - 1].0);
            return lerp_color(frac, stops[i - 1].1, stops[i].1);
        }
    }
    let last = stops.last().unwrap().1;
    format!("#{:02x}{:02x}{:02x}", last.0, last.1, last.2)
}

fn viridis(t: f64) -> String {
    lerp3(t, &[
        (0.00, (68, 1, 84)), (0.13, (72, 35, 116)), (0.25, (64, 67, 135)),
        (0.38, (52, 94, 141)), (0.50, (33, 144, 140)), (0.63, (53, 183, 121)),
        (0.75, (109, 205, 89)), (0.88, (180, 222, 44)), (1.00, (253, 231, 37)),
    ])
}

fn plasma(t: f64) -> String {
    lerp3(t, &[
        (0.00, (13, 8, 135)), (0.13, (75, 3, 161)), (0.25, (125, 3, 168)),
        (0.38, (168, 34, 150)), (0.50, (203, 70, 121)), (0.63, (229, 107, 93)),
        (0.75, (248, 148, 65)), (0.88, (253, 195, 40)), (1.00, (240, 249, 33)),
    ])
}

fn inferno(t: f64) -> String {
    lerp3(t, &[
        (0.00, (0, 0, 4)), (0.13, (22, 11, 57)), (0.25, (66, 10, 104)),
        (0.38, (106, 23, 110)), (0.50, (147, 38, 103)), (0.63, (188, 55, 84)),
        (0.75, (221, 81, 58)), (0.88, (243, 131, 29)), (1.00, (252, 255, 164)),
    ])
}

fn magma(t: f64) -> String {
    lerp3(t, &[
        (0.00, (0, 0, 4)), (0.13, (18, 13, 50)), (0.25, (51, 16, 104)),
        (0.38, (89, 26, 130)), (0.50, (131, 37, 142)), (0.63, (173, 49, 144)),
        (0.75, (214, 72, 131)), (0.88, (244, 136, 128)), (1.00, (252, 253, 191)),
    ])
}

fn cividis(t: f64) -> String {
    lerp3(t, &[
        (0.00, (0, 32, 76)), (0.25, (57, 77, 107)), (0.50, (127, 127, 127)),
        (0.75, (186, 174, 87)), (1.00, (253, 231, 37)),
    ])
}

fn turbo(t: f64) -> String {
    lerp3(t, &[
        (0.00, (48, 18, 59)), (0.10, (67, 84, 190)), (0.20, (45, 152, 218)),
        (0.30, (32, 207, 162)), (0.40, (87, 238, 96)), (0.50, (167, 252, 39)),
        (0.60, (225, 234, 25)), (0.70, (254, 188, 43)), (0.80, (247, 127, 42)),
        (0.90, (209, 56, 21)), (1.00, (122, 4, 3)),
    ])
}

fn hot(t: f64) -> String {
    lerp3(t, &[
        (0.00, (10, 0, 0)), (0.33, (255, 0, 0)), (0.67, (255, 255, 0)), (1.00, (255, 255, 255)),
    ])
}

fn cool(t: f64) -> String {
    let r = (t * 255.0) as u8;
    let b = ((1.0 - t) * 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}", r, 255 - r / 2, b)
}

fn jet(t: f64) -> String {
    lerp3(t, &[
        (0.00, (0, 0, 127)), (0.11, (0, 0, 255)), (0.35, (0, 255, 255)),
        (0.50, (0, 255, 0)), (0.65, (255, 255, 0)), (0.89, (255, 0, 0)),
        (1.00, (127, 0, 0)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_from_fn() {
        let data = SurfaceData::from_fn((-1.0, 1.0), (-1.0, 1.0), 20, 20, |x, y| x * x + y * y);
        assert_eq!(data.x.len(), 20);
        assert_eq!(data.y.len(), 20);
        assert_eq!(data.z.len(), 20);
        assert_eq!(data.z[0].len(), 20);
        let (z_min, z_max) = data.z_range();
        assert!(z_min >= 0.0);
        assert!(z_max <= 2.0);
    }

    #[test]
    fn surface_svg_renders() {
        let data = SurfaceData::from_fn((-1.0, 1.0), (-1.0, 1.0), 10, 10, |x, y| x * x - y * y);
        let svg = surface_svg(&data, &SurfaceConfig::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("polygon"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn wireframe_svg_renders() {
        let data = SurfaceData::from_fn((0.0, 1.0), (0.0, 1.0), 5, 5, |x, y| (x + y).sin());
        let svg = wireframe_svg(&data, &View3D::default(), "#333", Some("Wire"));
        assert!(svg.contains("path"));
        assert!(svg.contains("Wire"));
    }

    #[test]
    fn contour_svg_renders() {
        let data = SurfaceData::from_fn((-2.0, 2.0), (-2.0, 2.0), 30, 30, |x, y| x * x + y * y);
        let levels = auto_contour_levels(&data, 8, &Colormap::Viridis);
        assert_eq!(levels.len(), 8);
        let svg = contour_svg(&data, &levels, 600.0, 500.0, Some("Contour"));
        assert!(svg.contains("line"));
    }

    #[test]
    fn colormaps_produce_valid_hex() {
        for cmap in &[Colormap::Viridis, Colormap::Plasma, Colormap::Inferno, Colormap::Magma,
                      Colormap::Cividis, Colormap::Turbo, Colormap::Hot, Colormap::Cool,
                      Colormap::Jet, Colormap::Grayscale, Colormap::BlueRed, Colormap::GreenYellow] {
            for i in 0..=10 {
                let c = cmap.color(i as f64 / 10.0);
                assert!(c.starts_with('#'), "colormap {:?} at {}: {}", cmap, i, c);
                assert_eq!(c.len(), 7, "colormap {:?} at {}: {}", cmap, i, c);
            }
        }
    }

    #[test]
    fn palette_generation() {
        let palette = Colormap::Viridis.palette(5);
        assert_eq!(palette.len(), 5);
        for c in &palette {
            assert!(c.starts_with('#'));
        }
    }

    #[test]
    fn colorbar_renders() {
        let svg = colorbar_svg(10.0, 10.0, 20.0, 200.0, 0.0, 100.0, &Colormap::Plasma);
        assert!(svg.contains("rect"));
        assert!(svg.contains("text"));
    }

    #[test]
    fn projection_center() {
        let view = View3D::default();
        let (sx, sy, _) = project(0.0, 0.0, 0.0, &view);
        // Center should be near the middle of the viewport
        assert!((sx - view.width / 2.0).abs() < view.width * 0.3);
        assert!((sy - view.height / 2.0).abs() < view.height * 0.3);
    }
}
