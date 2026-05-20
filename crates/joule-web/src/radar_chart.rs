//! Spider / radar chart.  Multiple data series plotted as polygons on radial
//! axes emanating from a common centre.  Axis labels, configurable scale rings,
//! multiple overlapping series with filled/outlined styles.  Pure Rust SVG output.

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;

// ── Axis ─────────────────────────────────────────────────────────

/// A single axis on the radar chart.
#[derive(Debug, Clone)]
pub struct RadarAxis {
    pub label: String,
    pub min: f64,
    pub max: f64,
}

impl RadarAxis {
    pub fn new(label: impl Into<String>, min: f64, max: f64) -> Self {
        Self {
            label: label.into(),
            min,
            max,
        }
    }

    /// Normalize `value` to [0, 1] within this axis range.
    pub fn normalize(&self, value: f64) -> f64 {
        let range = (self.max - self.min).max(f64::EPSILON);
        ((value - self.min) / range).clamp(0.0, 1.0)
    }

    /// Denormalize a [0,1] fraction back to data units.
    pub fn denormalize(&self, norm: f64) -> f64 {
        self.min + norm * (self.max - self.min)
    }
}

// ── Dataset ──────────────────────────────────────────────────────

/// Drawing style for a radar series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesStyle {
    /// Filled polygon with semi-transparent interior.
    Filled,
    /// Outline only (no fill).
    Outlined,
}

/// One dataset (polygon) on the radar chart.
#[derive(Debug, Clone)]
pub struct RadarDataset {
    pub name: String,
    /// One value per axis, in axis order.
    pub values: Vec<f64>,
    pub fill_color: String,
    pub stroke_color: String,
    pub stroke_width: f64,
    pub fill_opacity: f64,
    pub style: SeriesStyle,
    /// Whether to show dots on the vertices.
    pub show_dots: bool,
    pub dot_radius: f64,
}

impl RadarDataset {
    pub fn new(
        name: impl Into<String>,
        values: Vec<f64>,
        fill_color: impl Into<String>,
        stroke_color: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            values,
            fill_color: fill_color.into(),
            stroke_color: stroke_color.into(),
            stroke_width: 2.0,
            fill_opacity: 0.25,
            style: SeriesStyle::Filled,
            show_dots: true,
            dot_radius: 3.0,
        }
    }

    pub fn outlined(
        name: impl Into<String>,
        values: Vec<f64>,
        stroke_color: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            values,
            fill_color: String::new(),
            stroke_color: stroke_color.into(),
            stroke_width: 2.5,
            fill_opacity: 0.0,
            style: SeriesStyle::Outlined,
            show_dots: true,
            dot_radius: 3.0,
        }
    }

    pub fn with_stroke_width(mut self, w: f64) -> Self {
        self.stroke_width = w.max(0.5);
        self
    }

    pub fn with_fill_opacity(mut self, o: f64) -> Self {
        self.fill_opacity = o.clamp(0.0, 1.0);
        self
    }

    pub fn with_dots(mut self, show: bool, radius: f64) -> Self {
        self.show_dots = show;
        self.dot_radius = radius.max(0.5);
        self
    }
}

// ── Config ───────────────────────────────────────────────────────

/// Configuration for a radar chart.
#[derive(Debug, Clone)]
pub struct RadarChartConfig {
    pub width: f64,
    pub height: f64,
    /// Number of concentric grid rings.
    pub grid_levels: usize,
    pub font_size: f64,
    pub show_grid: bool,
    pub show_legend: bool,
    /// Scale labels on the first axis spoke.
    pub show_scale_labels: bool,
}

impl Default for RadarChartConfig {
    fn default() -> Self {
        Self {
            width: 400.0,
            height: 400.0,
            grid_levels: 5,
            font_size: 12.0,
            show_grid: true,
            show_legend: true,
            show_scale_labels: true,
        }
    }
}

impl RadarChartConfig {
    pub fn center(&self) -> (f64, f64) {
        (self.width / 2.0, self.height / 2.0)
    }

    pub fn radius(&self) -> f64 {
        self.width.min(self.height) / 2.0 * 0.72
    }
}

// ── Geometry helpers ─────────────────────────────────────────────

/// Compute the angle (in radians, clockwise from top) for axis index `i` out
/// of `n` axes.  Top-centre = -PI/2 so the first axis points straight up.
fn axis_angle(i: usize, n: usize) -> f64 {
    -PI / 2.0 + 2.0 * PI * i as f64 / n as f64
}

/// Compute the (x, y) vertices of a radar polygon for one dataset.
pub fn polygon_vertices(
    axes: &[RadarAxis],
    values: &[f64],
    cx: f64,
    cy: f64,
    radius: f64,
) -> Vec<(f64, f64)> {
    let n = axes.len();
    axes.iter()
        .zip(values.iter())
        .enumerate()
        .map(|(i, (ax, &v))| {
            let norm = ax.normalize(v);
            let angle = axis_angle(i, n);
            let r = norm * radius;
            (cx + r * angle.cos(), cy + r * angle.sin())
        })
        .collect()
}

/// Compute the area of a polygon given its vertices (shoelace formula).
pub fn polygon_area(vertices: &[(f64, f64)]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].0 * vertices[j].1;
        area -= vertices[j].0 * vertices[i].1;
    }
    area.abs() / 2.0
}

/// Compute perimeter of a polygon.
pub fn polygon_perimeter(vertices: &[(f64, f64)]) -> f64 {
    let n = vertices.len();
    if n < 2 {
        return 0.0;
    }
    let mut perimeter = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = vertices[j].0 - vertices[i].0;
        let dy = vertices[j].1 - vertices[i].1;
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    perimeter
}

/// Build an SVG polygon `points` attribute from vertices.
pub fn polygon_svg_points(vertices: &[(f64, f64)]) -> String {
    vertices
        .iter()
        .map(|(x, y)| format!("{x},{y}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Rendering ────────────────────────────────────────────────────

/// Render the grid (concentric polygons + radial spokes).
fn render_grid(axes: &[RadarAxis], cfg: &RadarChartConfig) -> String {
    if !cfg.show_grid || axes.is_empty() {
        return String::new();
    }
    let (cx, cy) = cfg.center();
    let r = cfg.radius();
    let n = axes.len();
    let mut svg = String::from("<g class=\"radar-grid\">");

    // Concentric rings
    for level in 1..=cfg.grid_levels {
        let frac = level as f64 / cfg.grid_levels as f64;
        let ring_r = frac * r;
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let a = axis_angle(i, n);
                (cx + ring_r * a.cos(), cy + ring_r * a.sin())
            })
            .collect();
        let points_str = polygon_svg_points(&pts);
        let _ = write!(
            svg,
            "<polygon points=\"{points_str}\" fill=\"none\" stroke=\"silver\" stroke-width=\"0.5\" />"
        );

        // Scale label on first spoke
        if cfg.show_scale_labels {
            let label_val = axes[0].denormalize(frac);
            let a0 = axis_angle(0, n);
            let lx = cx + ring_r * a0.cos() + 4.0;
            let ly = cy + ring_r * a0.sin();
            let fs = cfg.font_size * 0.75;
            let _ = write!(
                svg,
                "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" fill=\"gray\">{label_val:.0}</text>"
            );
        }
    }

    // Radial spokes + labels
    for (i, ax) in axes.iter().enumerate() {
        let a = axis_angle(i, n);
        let x2 = cx + r * a.cos();
        let y2 = cy + r * a.sin();
        let _ = write!(
            svg,
            "<line x1=\"{cx}\" y1=\"{cy}\" x2=\"{x2}\" y2=\"{y2}\" \
             stroke=\"silver\" stroke-width=\"0.5\" />"
        );
        let lx = cx + (r + 16.0) * a.cos();
        let ly = cy + (r + 16.0) * a.sin();
        let fs = cfg.font_size;
        let _ = write!(
            svg,
            "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" \
             text-anchor=\"middle\" dominant-baseline=\"middle\">{}</text>",
            ax.label
        );
    }

    svg.push_str("</g>");
    svg
}

/// Render a single dataset polygon.
fn render_dataset(ds: &RadarDataset, axes: &[RadarAxis], cfg: &RadarChartConfig) -> String {
    let (cx, cy) = cfg.center();
    let r = cfg.radius();
    let verts = polygon_vertices(axes, &ds.values, cx, cy, r);
    let pts = polygon_svg_points(&verts);
    let mut svg = String::new();

    match ds.style {
        SeriesStyle::Filled => {
            let _ = write!(
                svg,
                "<polygon points=\"{pts}\" fill=\"{}\" fill-opacity=\"{}\" \
                 stroke=\"{}\" stroke-width=\"{}\" />",
                ds.fill_color, ds.fill_opacity, ds.stroke_color, ds.stroke_width
            );
        }
        SeriesStyle::Outlined => {
            let _ = write!(
                svg,
                "<polygon points=\"{pts}\" fill=\"none\" \
                 stroke=\"{}\" stroke-width=\"{}\" />",
                ds.stroke_color, ds.stroke_width
            );
        }
    }

    // Dots on vertices
    if ds.show_dots {
        for (x, y) in &verts {
            let _ = write!(
                svg,
                "<circle cx=\"{x}\" cy=\"{y}\" r=\"{}\" fill=\"{}\" />",
                ds.dot_radius, ds.stroke_color
            );
        }
    }

    svg
}

/// Render a legend block.
fn render_legend(datasets: &[RadarDataset], cfg: &RadarChartConfig) -> String {
    if !cfg.show_legend || datasets.is_empty() {
        return String::new();
    }
    let mut svg = String::new();
    let lx = 10.0;
    let mut ly = 15.0;
    let fs = cfg.font_size * 0.9;

    for ds in datasets {
        let color = if ds.style == SeriesStyle::Filled {
            &ds.fill_color
        } else {
            &ds.stroke_color
        };
        let _ = write!(
            svg,
            "<rect x=\"{lx}\" y=\"{}\" width=\"10\" height=\"10\" fill=\"{color}\" />",
            ly - 8.0,
        );
        let _ = write!(
            svg,
            "<text x=\"{}\" y=\"{ly}\" font-size=\"{fs}\">{}</text>",
            lx + 14.0,
            ds.name
        );
        ly += fs + 6.0;
    }
    svg
}

/// Render a complete radar chart as an SVG string.
pub fn render_radar_chart(
    axes: &[RadarAxis],
    datasets: &[RadarDataset],
    cfg: &RadarChartConfig,
) -> String {
    let mut svg = String::with_capacity(4096);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">",
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    svg.push_str(&render_grid(axes, cfg));

    for ds in datasets {
        svg.push_str(&render_dataset(ds, axes, cfg));
    }

    svg.push_str(&render_legend(datasets, cfg));

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_axes() -> Vec<RadarAxis> {
        vec![
            RadarAxis::new("Speed", 0.0, 10.0),
            RadarAxis::new("Power", 0.0, 10.0),
            RadarAxis::new("Range", 0.0, 10.0),
            RadarAxis::new("Armor", 0.0, 10.0),
            RadarAxis::new("Stealth", 0.0, 10.0),
        ]
    }

    fn sample_dataset() -> RadarDataset {
        RadarDataset::new("Vehicle A", vec![8.0, 6.0, 7.0, 5.0, 9.0], "steelblue", "navy")
    }

    #[test]
    fn normalize_clamps() {
        let ax = RadarAxis::new("X", 0.0, 100.0);
        assert!((ax.normalize(50.0) - 0.5).abs() < 1e-9);
        assert!((ax.normalize(-10.0)).abs() < 1e-9);
        assert!((ax.normalize(200.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_zero_range() {
        let ax = RadarAxis::new("X", 5.0, 5.0);
        let _ = ax.normalize(5.0);
    }

    #[test]
    fn denormalize_roundtrip() {
        let ax = RadarAxis::new("X", 10.0, 50.0);
        let norm = ax.normalize(30.0);
        let back = ax.denormalize(norm);
        assert!((back - 30.0).abs() < 1e-9);
    }

    #[test]
    fn polygon_vertices_count() {
        let axes = sample_axes();
        let ds = sample_dataset();
        let verts = polygon_vertices(&axes, &ds.values, 200.0, 200.0, 100.0);
        assert_eq!(verts.len(), 5);
    }

    #[test]
    fn polygon_vertices_first_points_up() {
        let axes = vec![RadarAxis::new("A", 0.0, 10.0)];
        let verts = polygon_vertices(&axes, &[10.0], 200.0, 200.0, 100.0);
        let (x, y) = verts[0];
        assert!((x - 200.0).abs() < 1e-6);
        assert!((y - 100.0).abs() < 1e-6);
    }

    #[test]
    fn polygon_area_triangle() {
        let verts = vec![(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)];
        assert!((polygon_area(&verts) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn polygon_area_empty() {
        assert_eq!(polygon_area(&[]), 0.0);
        assert_eq!(polygon_area(&[(0.0, 0.0)]), 0.0);
    }

    #[test]
    fn polygon_perimeter_square() {
        let verts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!((polygon_perimeter(&verts) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn polygon_perimeter_empty() {
        assert_eq!(polygon_perimeter(&[]), 0.0);
        assert_eq!(polygon_perimeter(&[(0.0, 0.0)]), 0.0);
    }

    #[test]
    fn polygon_svg_points_format() {
        let pts = polygon_svg_points(&[(1.0, 2.0), (3.0, 4.0)]);
        assert_eq!(pts, "1,2 3,4");
    }

    #[test]
    fn render_radar_chart_has_polygons() {
        let axes = sample_axes();
        let datasets = vec![sample_dataset()];
        let cfg = RadarChartConfig::default();
        let svg = render_radar_chart(&axes, &datasets, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("steelblue"));
    }

    #[test]
    fn render_radar_chart_multiple_datasets() {
        let axes = sample_axes();
        let ds1 = sample_dataset();
        let ds2 = RadarDataset::new("Vehicle B", vec![4.0, 9.0, 3.0, 8.0, 2.0], "coral", "darkred");
        let cfg = RadarChartConfig::default();
        let svg = render_radar_chart(&axes, &[ds1, ds2], &cfg);
        // Grid polygons (5 levels) + 2 data polygons = 7
        assert!(svg.matches("<polygon").count() >= 7);
    }

    #[test]
    fn grid_labels_present() {
        let axes = sample_axes();
        let cfg = RadarChartConfig::default();
        let svg = render_radar_chart(&axes, &[], &cfg);
        assert!(svg.contains("Speed"));
        assert!(svg.contains("Stealth"));
    }

    #[test]
    fn area_comparison() {
        let axes = sample_axes();
        let r = 100.0;
        let v1 = polygon_vertices(&axes, &[10.0; 5], 0.0, 0.0, r);
        let v2 = polygon_vertices(&axes, &[5.0; 5], 0.0, 0.0, r);
        let a1 = polygon_area(&v1);
        let a2 = polygon_area(&v2);
        assert!((a1 / a2 - 4.0).abs() < 1e-6);
    }

    #[test]
    fn vertices_on_circle() {
        let axes = sample_axes();
        let r = 100.0;
        let cx = 200.0;
        let cy = 200.0;
        let verts = polygon_vertices(&axes, &[10.0; 5], cx, cy, r);
        for (x, y) in &verts {
            let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            assert!((dist - r).abs() < 1e-6);
        }
    }

    #[test]
    fn outlined_series_no_fill() {
        let axes = sample_axes();
        let ds = RadarDataset::outlined("Outline", vec![5.0; 5], "red");
        let cfg = RadarChartConfig::default();
        let svg = render_radar_chart(&axes, &[ds], &cfg);
        assert!(svg.contains("fill=\"none\""));
    }

    #[test]
    fn legend_rendered() {
        let axes = sample_axes();
        let ds = sample_dataset();
        let cfg = RadarChartConfig::default();
        let svg = render_radar_chart(&axes, &[ds], &cfg);
        assert!(svg.contains("Vehicle A"));
    }

    #[test]
    fn scale_labels_rendered() {
        let axes = sample_axes();
        let cfg = RadarChartConfig::default();
        let svg = render_radar_chart(&axes, &[], &cfg);
        // Should have scale labels (e.g., "2", "4", "6", "8", "10")
        assert!(svg.contains(">10</text>") || svg.contains(">10<"));
    }

    #[test]
    fn config_radius() {
        let cfg = RadarChartConfig {
            width: 400.0,
            height: 300.0,
            ..Default::default()
        };
        // min(400,300) / 2 * 0.72 = 108
        assert!((cfg.radius() - 108.0).abs() < 1e-9);
    }

    #[test]
    fn dataset_builder() {
        let ds = RadarDataset::new("A", vec![1.0], "blue", "navy")
            .with_stroke_width(3.0)
            .with_fill_opacity(0.5)
            .with_dots(false, 5.0);
        assert!((ds.stroke_width - 3.0).abs() < 1e-9);
        assert!((ds.fill_opacity - 0.5).abs() < 1e-9);
        assert!(!ds.show_dots);
    }
}
