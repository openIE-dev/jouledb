//! Polar coordinate chart: polar scatter, polar area (filled sectors), and rose
//! charts.  All output is SVG strings — no browser deps, fully testable headless.

use std::f64::consts::PI;

// ── Data types ───────────────────────────────────────────────────

/// A point in polar coordinates.
#[derive(Debug, Clone, Copy)]
pub struct PolarPoint {
    /// Angle in degrees (0–360).
    pub angle_deg: f64,
    /// Distance from origin.
    pub radius: f64,
}

impl PolarPoint {
    pub fn new(angle_deg: f64, radius: f64) -> Self {
        Self { angle_deg, radius }
    }

    /// Convert to cartesian (x, y) with given center and scale.
    pub fn to_cartesian(&self, cx: f64, cy: f64, radius_scale: f64) -> (f64, f64) {
        let rad = self.angle_deg.to_radians();
        let r = self.radius * radius_scale;
        (cx + r * rad.cos(), cy - r * rad.sin())
    }

    /// Convert from cartesian back to polar (angle_deg, radius).
    pub fn from_cartesian(x: f64, y: f64, cx: f64, cy: f64, radius_scale: f64) -> Self {
        let dx = x - cx;
        let dy = -(y - cy); // SVG y is inverted
        let r = (dx * dx + dy * dy).sqrt();
        let angle = dy.atan2(dx).to_degrees();
        let angle = if angle < 0.0 { angle + 360.0 } else { angle };
        Self {
            angle_deg: angle,
            radius: r / radius_scale.max(f64::EPSILON),
        }
    }
}

/// A named series of polar points.
#[derive(Debug, Clone)]
pub struct PolarSeries {
    pub name: String,
    pub points: Vec<PolarPoint>,
    pub color: String,
}

// ── Radial axis ─────────────────────────────────────────────────

/// Configuration for the radial (distance) axis.
#[derive(Debug, Clone)]
pub struct RadialAxis {
    pub min_radius: f64,
    pub max_radius: f64,
    /// Distance between concentric grid circles.
    pub tick_interval: f64,
}

impl RadialAxis {
    pub fn new(min_radius: f64, max_radius: f64, tick_interval: f64) -> Self {
        Self {
            min_radius,
            max_radius,
            tick_interval: tick_interval.max(f64::EPSILON),
        }
    }

    /// Scale a data radius to pixel radius.
    pub fn scale(&self, value: f64, pixel_radius: f64) -> f64 {
        let range = (self.max_radius - self.min_radius).max(f64::EPSILON);
        ((value - self.min_radius) / range) * pixel_radius
    }

    /// Generate tick values from min to max.
    pub fn ticks(&self) -> Vec<f64> {
        let mut v = Vec::new();
        let mut t = self.min_radius + self.tick_interval;
        while t <= self.max_radius + self.tick_interval * 0.001 {
            v.push(t);
            t += self.tick_interval;
        }
        v
    }
}

/// Configuration for the angular axis.
#[derive(Debug, Clone)]
pub struct AngularAxis {
    /// If empty, use default 0/30/60/.../330 degree labels.
    pub labels: Vec<String>,
}

impl Default for AngularAxis {
    fn default() -> Self {
        Self { labels: Vec::new() }
    }
}

impl AngularAxis {
    /// Custom labels evenly spaced around 360 degrees.
    pub fn with_labels(labels: Vec<String>) -> Self {
        Self { labels }
    }

    /// Return (angle_deg, label) pairs.
    pub fn label_positions(&self) -> Vec<(f64, String)> {
        if self.labels.is_empty() {
            (0..12)
                .map(|i| {
                    let deg = i as f64 * 30.0;
                    (deg, format!("{deg}°"))
                })
                .collect()
        } else {
            let n = self.labels.len();
            self.labels
                .iter()
                .enumerate()
                .map(|(i, l)| (i as f64 * 360.0 / n as f64, l.clone()))
                .collect()
        }
    }
}

// ── Chart config ────────────────────────────────────────────────

/// Configuration for a polar chart.
#[derive(Debug, Clone)]
pub struct PolarChartConfig {
    pub width: f64,
    pub height: f64,
    pub radial_axis: RadialAxis,
    pub angular_axis: AngularAxis,
    pub show_grid: bool,
    pub font_size: f64,
}

impl PolarChartConfig {
    pub fn center(&self) -> (f64, f64) {
        (self.width / 2.0, self.height / 2.0)
    }

    /// Pixel radius of the plot area (80 % of half the smaller dimension).
    pub fn plot_radius(&self) -> f64 {
        self.width.min(self.height) / 2.0 * 0.8
    }
}

// ── SVG arc path helper ─────────────────────────────────────────

/// Generate an SVG arc path string for a sector from `start_deg` to `end_deg`
/// at the given radius, centred at `(cx, cy)`.  Returns the `d` attribute.
pub fn arc_path(
    cx: f64,
    cy: f64,
    radius: f64,
    start_deg: f64,
    end_deg: f64,
    include_center: bool,
) -> String {
    let s = start_deg.to_radians();
    let e = end_deg.to_radians();
    let x1 = cx + radius * s.cos();
    let y1 = cy - radius * s.sin();
    let x2 = cx + radius * e.cos();
    let y2 = cy - radius * e.sin();
    let sweep = end_deg - start_deg;
    let large = if sweep.abs() > 180.0 { 1 } else { 0 };
    // SVG arc: sweep-flag 0 = counter-clockwise in screen coords (= clockwise
    // in math coords since y is flipped).  We draw CCW in math → sweep 0.
    if include_center {
        format!(
            "M {cx} {cy} L {x1} {y1} A {radius} {radius} 0 {large} 0 {x2} {y2} Z"
        )
    } else {
        format!(
            "M {x1} {y1} A {radius} {radius} 0 {large} 0 {x2} {y2}"
        )
    }
}

// ── Grid circles & radial lines ─────────────────────────────────

/// Render concentric grid circles + radial lines as SVG.
pub fn render_grid(cfg: &PolarChartConfig) -> String {
    if !cfg.show_grid {
        return String::new();
    }
    let (cx, cy) = cfg.center();
    let pr = cfg.plot_radius();
    let mut svg = String::from("<g class=\"polar-grid\">");

    // Concentric circles
    for tick in cfg.radial_axis.ticks() {
        let r = cfg.radial_axis.scale(tick, pr);
        svg.push_str(&format!(
            "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" fill=\"none\" stroke=\"#ddd\" />"
        ));
        // Tick label
        let fs = cfg.font_size;
        svg.push_str(&format!(
            "<text x=\"{cx}\" y=\"{}\" font-size=\"{fs}\" text-anchor=\"middle\" fill=\"#999\">{tick:.1}</text>",
            cy - r - 2.0
        ));
    }

    // Radial lines
    for (deg, label) in cfg.angular_axis.label_positions() {
        let rad = deg.to_radians();
        let x2 = cx + pr * rad.cos();
        let y2 = cy - pr * rad.sin();
        svg.push_str(&format!(
            "<line x1=\"{cx}\" y1=\"{cy}\" x2=\"{x2}\" y2=\"{y2}\" stroke=\"#eee\" />"
        ));
        let lx = cx + (pr + 14.0) * rad.cos();
        let ly = cy - (pr + 14.0) * rad.sin();
        let fs = cfg.font_size;
        svg.push_str(&format!(
            "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" text-anchor=\"middle\" dominant-baseline=\"middle\">{label}</text>"
        ));
    }

    svg.push_str("</g>");
    svg
}

// ── Polar scatter ───────────────────────────────────────────────

/// Render a polar scatter chart as SVG.
pub fn render_polar_scatter(series: &[PolarSeries], cfg: &PolarChartConfig) -> String {
    let (cx, cy) = cfg.center();
    let pr = cfg.plot_radius();
    let mut svg = svg_open(cfg);
    svg.push_str(&render_grid(cfg));

    for s in series {
        for pt in &s.points {
            let r = cfg.radial_axis.scale(pt.radius, pr);
            let rad = pt.angle_deg.to_radians();
            let x = cx + r * rad.cos();
            let y = cy - r * rad.sin();
            svg.push_str(&format!(
                "<circle cx=\"{x}\" cy=\"{y}\" r=\"4\" fill=\"{}\" />",
                s.color
            ));
        }
    }

    svg.push_str("</svg>");
    svg
}

// ── Polar area chart (filled sectors of equal angular width) ────

/// Sector for a polar area chart.
#[derive(Debug, Clone)]
pub struct PolarAreaSector {
    pub label: String,
    pub value: f64,
    pub color: String,
}

/// Render a polar area chart — equal-angle sectors with radius proportional to value.
pub fn render_polar_area(sectors: &[PolarAreaSector], cfg: &PolarChartConfig) -> String {
    let (cx, cy) = cfg.center();
    let pr = cfg.plot_radius();
    let mut svg = svg_open(cfg);
    svg.push_str(&render_grid(cfg));

    if sectors.is_empty() {
        svg.push_str("</svg>");
        return svg;
    }

    let n = sectors.len() as f64;
    let angle_width = 360.0 / n;

    for (i, sector) in sectors.iter().enumerate() {
        let start = i as f64 * angle_width;
        let end = start + angle_width;
        let r = cfg.radial_axis.scale(sector.value, pr);
        let d = arc_path(cx, cy, r, start, end, true);
        svg.push_str(&format!(
            "<path d=\"{d}\" fill=\"{}\" fill-opacity=\"0.7\" stroke=\"#fff\" />",
            sector.color
        ));
    }

    svg.push_str("</svg>");
    svg
}

// ── Rose chart (variable-width sectors) ─────────────────────────

/// Sector for a rose chart — both angle width and radius can vary.
#[derive(Debug, Clone)]
pub struct RoseSector {
    pub label: String,
    pub angle_width_deg: f64,
    pub value: f64,
    pub color: String,
}

/// Render a rose (Nightingale) chart.
pub fn render_rose_chart(sectors: &[RoseSector], cfg: &PolarChartConfig) -> String {
    let (cx, cy) = cfg.center();
    let pr = cfg.plot_radius();
    let mut svg = svg_open(cfg);
    svg.push_str(&render_grid(cfg));

    let mut angle = 0.0_f64;
    for sector in sectors {
        let end = angle + sector.angle_width_deg;
        let r = cfg.radial_axis.scale(sector.value, pr);
        let d = arc_path(cx, cy, r, angle, end, true);
        svg.push_str(&format!(
            "<path d=\"{d}\" fill=\"{}\" fill-opacity=\"0.7\" stroke=\"#fff\" />",
            sector.color
        ));
        angle = end;
    }

    svg.push_str("</svg>");
    svg
}

// ── helpers ─────────────────────────────────────────────────────

fn svg_open(cfg: &PolarChartConfig) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        cfg.width, cfg.height, cfg.width, cfg.height
    )
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> PolarChartConfig {
        PolarChartConfig {
            width: 400.0,
            height: 400.0,
            radial_axis: RadialAxis::new(0.0, 10.0, 2.0),
            angular_axis: AngularAxis::default(),
            show_grid: true,
            font_size: 12.0,
        }
    }

    #[test]
    fn polar_point_to_cartesian_origin() {
        let pt = PolarPoint::new(0.0, 0.0);
        let (x, y) = pt.to_cartesian(100.0, 100.0, 1.0);
        assert!((x - 100.0).abs() < 1e-9);
        assert!((y - 100.0).abs() < 1e-9);
    }

    #[test]
    fn polar_point_to_cartesian_east() {
        let pt = PolarPoint::new(0.0, 5.0);
        let (x, y) = pt.to_cartesian(0.0, 0.0, 2.0);
        assert!((x - 10.0).abs() < 1e-9);
        assert!(y.abs() < 1e-9);
    }

    #[test]
    fn polar_point_to_cartesian_north() {
        // 90 degrees → up in math, negative y in screen
        let pt = PolarPoint::new(90.0, 5.0);
        let (x, y) = pt.to_cartesian(0.0, 0.0, 1.0);
        assert!(x.abs() < 1e-9);
        assert!((y - (-5.0)).abs() < 1e-9);
    }

    #[test]
    fn polar_from_cartesian_roundtrip() {
        let original = PolarPoint::new(45.0, 7.0);
        let (x, y) = original.to_cartesian(100.0, 100.0, 2.0);
        let recovered = PolarPoint::from_cartesian(x, y, 100.0, 100.0, 2.0);
        assert!((recovered.angle_deg - 45.0).abs() < 1e-6);
        assert!((recovered.radius - 7.0).abs() < 1e-6);
    }

    #[test]
    fn radial_axis_scale() {
        let axis = RadialAxis::new(0.0, 100.0, 25.0);
        assert!((axis.scale(0.0, 200.0)).abs() < 1e-9);
        assert!((axis.scale(50.0, 200.0) - 100.0).abs() < 1e-9);
        assert!((axis.scale(100.0, 200.0) - 200.0).abs() < 1e-9);
    }

    #[test]
    fn radial_axis_ticks() {
        let axis = RadialAxis::new(0.0, 10.0, 2.0);
        let ticks = axis.ticks();
        assert_eq!(ticks.len(), 5); // 2, 4, 6, 8, 10
        assert!((ticks[0] - 2.0).abs() < 1e-9);
        assert!((ticks[4] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn angular_axis_default_12_labels() {
        let axis = AngularAxis::default();
        let positions = axis.label_positions();
        assert_eq!(positions.len(), 12);
        assert_eq!(positions[0].1, "0°");
    }

    #[test]
    fn angular_axis_custom_labels() {
        let axis = AngularAxis::with_labels(vec!["N".into(), "E".into(), "S".into(), "W".into()]);
        let positions = axis.label_positions();
        assert_eq!(positions.len(), 4);
        assert!((positions[1].0 - 90.0).abs() < 1e-9);
        assert_eq!(positions[1].1, "E");
    }

    #[test]
    fn arc_path_contains_arc_command() {
        let d = arc_path(100.0, 100.0, 50.0, 0.0, 90.0, true);
        assert!(d.contains('A'));
        assert!(d.contains('Z'));
        assert!(d.contains('M'));
    }

    #[test]
    fn render_grid_has_circles_and_lines() {
        let cfg = default_cfg();
        let svg = render_grid(&cfg);
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<line"));
        assert!(svg.contains("<text"));
    }

    #[test]
    fn render_grid_empty_when_disabled() {
        let mut cfg = default_cfg();
        cfg.show_grid = false;
        let svg = render_grid(&cfg);
        assert!(svg.is_empty());
    }

    #[test]
    fn polar_scatter_svg() {
        let cfg = default_cfg();
        let series = vec![PolarSeries {
            name: "Wind".into(),
            points: vec![
                PolarPoint::new(0.0, 5.0),
                PolarPoint::new(90.0, 8.0),
                PolarPoint::new(180.0, 3.0),
            ],
            color: "#3498db".into(),
        }];
        let svg = render_polar_scatter(&series, &cfg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("#3498db"));
    }

    #[test]
    fn polar_area_svg() {
        let cfg = default_cfg();
        let sectors = vec![
            PolarAreaSector { label: "A".into(), value: 5.0, color: "#f00".into() },
            PolarAreaSector { label: "B".into(), value: 8.0, color: "#0f0".into() },
            PolarAreaSector { label: "C".into(), value: 3.0, color: "#00f".into() },
        ];
        let svg = render_polar_area(&sectors, &cfg);
        assert!(svg.contains("<path"));
        // 3 sectors → 3 paths
        assert_eq!(svg.matches("<path").count(), 3);
    }

    #[test]
    fn polar_area_empty() {
        let cfg = default_cfg();
        let svg = render_polar_area(&[], &cfg);
        assert!(svg.contains("</svg>"));
        assert!(!svg.contains("<path"));
    }

    #[test]
    fn rose_chart_svg() {
        let cfg = default_cfg();
        let sectors = vec![
            RoseSector { label: "A".into(), angle_width_deg: 120.0, value: 6.0, color: "#e74c3c".into() },
            RoseSector { label: "B".into(), angle_width_deg: 120.0, value: 9.0, color: "#2ecc71".into() },
            RoseSector { label: "C".into(), angle_width_deg: 120.0, value: 4.0, color: "#3498db".into() },
        ];
        let svg = render_rose_chart(&sectors, &cfg);
        assert!(svg.contains("<path"));
        assert_eq!(svg.matches("<path").count(), 3);
    }

    #[test]
    fn plot_radius_is_reasonable() {
        let cfg = default_cfg();
        let r = cfg.plot_radius();
        assert!(r > 0.0);
        assert!(r < cfg.width / 2.0);
    }
}
