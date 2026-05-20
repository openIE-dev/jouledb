//! Gauge / dial chart: semicircular gauge with value needle, coloured zones
//! (green/yellow/red), min/max labels, current value display, tick marks.
//! Pure Rust SVG output — no browser dependency.

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// A coloured zone on the gauge arc.
#[derive(Debug, Clone)]
pub struct GaugeZone {
    /// Start value in data units.
    pub from: f64,
    /// End value in data units.
    pub to: f64,
    /// Fill color (use named SVG colors only).
    pub color: String,
    /// Optional label for this zone.
    pub label: String,
}

impl GaugeZone {
    pub fn new(from: f64, to: f64, color: impl Into<String>) -> Self {
        Self {
            from,
            to,
            color: color.into(),
            label: String::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Span in data units.
    pub fn span(&self) -> f64 {
        (self.to - self.from).abs()
    }
}

/// Standard three-zone scheme: green / yellow / red.
pub fn standard_zones(min: f64, max: f64) -> Vec<GaugeZone> {
    let range = max - min;
    let t1 = min + range * 0.6;
    let t2 = min + range * 0.8;
    vec![
        GaugeZone::new(min, t1, "green").with_label("Normal"),
        GaugeZone::new(t1, t2, "gold").with_label("Warning"),
        GaugeZone::new(t2, max, "red").with_label("Critical"),
    ]
}

// ── Config ──────────────────────────────────────────────────────

/// Tick mark configuration.
#[derive(Debug, Clone)]
pub struct TickConfig {
    /// Major tick interval in data units.
    pub major_interval: f64,
    /// Minor ticks between each pair of major ticks.
    pub minor_count: usize,
    /// Length of major tick line in pixels.
    pub major_length: f64,
    /// Length of minor tick line in pixels.
    pub minor_length: f64,
}

impl Default for TickConfig {
    fn default() -> Self {
        Self {
            major_interval: 10.0,
            minor_count: 4,
            major_length: 12.0,
            minor_length: 6.0,
        }
    }
}

/// Configuration for the gauge chart.
#[derive(Debug, Clone)]
pub struct GaugeChartConfig {
    pub width: f64,
    pub height: f64,
    pub min: f64,
    pub max: f64,
    pub value: f64,
    /// Gauge arc angular range: start angle in degrees (0 = right, CCW).
    pub start_angle_deg: f64,
    /// End angle in degrees.
    pub end_angle_deg: f64,
    /// Radius of the gauge arc.
    pub radius: f64,
    /// Width of the arc band.
    pub arc_width: f64,
    pub zones: Vec<GaugeZone>,
    pub ticks: TickConfig,
    pub font_size: f64,
    /// Color of the needle.
    pub needle_color: String,
    /// Whether to show the numeric value below the gauge.
    pub show_value: bool,
    /// Label text (e.g., unit name).
    pub label: String,
}

impl Default for GaugeChartConfig {
    fn default() -> Self {
        Self {
            width: 400.0,
            height: 260.0,
            min: 0.0,
            max: 100.0,
            value: 50.0,
            start_angle_deg: 180.0,
            end_angle_deg: 0.0,
            radius: 150.0,
            arc_width: 24.0,
            zones: standard_zones(0.0, 100.0),
            ticks: TickConfig::default(),
            font_size: 12.0,
            needle_color: "dimgray".into(),
            show_value: true,
            label: String::new(),
        }
    }
}

impl GaugeChartConfig {
    pub fn center(&self) -> (f64, f64) {
        (self.width / 2.0, self.height - 30.0)
    }

    /// Map a data value to an angle in radians.
    pub fn value_to_angle(&self, v: f64) -> f64 {
        let start_rad = self.start_angle_deg.to_radians();
        let end_rad = self.end_angle_deg.to_radians();
        let range = (self.max - self.min).max(f64::EPSILON);
        let frac = ((v - self.min) / range).clamp(0.0, 1.0);
        start_rad + frac * (end_rad - start_rad)
    }

    /// Clamp value within min..max.
    pub fn clamped_value(&self) -> f64 {
        self.value.clamp(self.min, self.max)
    }

    /// Percentage of the gauge range.
    pub fn percent(&self) -> f64 {
        let range = (self.max - self.min).max(f64::EPSILON);
        ((self.value - self.min) / range * 100.0).clamp(0.0, 100.0)
    }
}

// ── SVG helpers ─────────────────────────────────────────────────

/// SVG arc path from angle a0 to a1 at given radius, with arc width.
fn arc_band_path(a0: f64, a1: f64, r_inner: f64, r_outer: f64, cx: f64, cy: f64) -> String {
    let x0_o = cx + r_outer * a0.cos();
    let y0_o = cy - r_outer * a0.sin();
    let x1_o = cx + r_outer * a1.cos();
    let y1_o = cy - r_outer * a1.sin();
    let x1_i = cx + r_inner * a1.cos();
    let y1_i = cy - r_inner * a1.sin();
    let x0_i = cx + r_inner * a0.cos();
    let y0_i = cy - r_inner * a0.sin();

    let span = (a0 - a1).abs();
    let large = if span > PI { 1 } else { 0 };
    // For sweep: going from a0 to a1 where a0 > a1 means clockwise in SVG coords
    let sweep = if a0 > a1 { 1 } else { 0 };
    let sweep_rev = if sweep == 1 { 0 } else { 1 };

    format!(
        "M{x0_o},{y0_o} A{r_outer},{r_outer} 0 {large},{sweep} {x1_o},{y1_o} \
         L{x1_i},{y1_i} A{r_inner},{r_inner} 0 {large},{sweep_rev} {x0_i},{y0_i} Z"
    )
}

// ── Rendering ───────────────────────────────────────────────────

/// Render zone arc bands.
fn render_zones(cfg: &GaugeChartConfig) -> String {
    let (cx, cy) = cfg.center();
    let r_outer = cfg.radius;
    let r_inner = cfg.radius - cfg.arc_width;
    let mut svg = String::new();

    for zone in &cfg.zones {
        let a0 = cfg.value_to_angle(zone.from);
        let a1 = cfg.value_to_angle(zone.to);
        let path = arc_band_path(a0, a1, r_inner, r_outer, cx, cy);
        let _ = write!(
            svg,
            "<path d=\"{path}\" fill=\"{}\" stroke=\"white\" stroke-width=\"0.5\" />",
            zone.color
        );
    }
    svg
}

/// Render tick marks.
fn render_ticks(cfg: &GaugeChartConfig) -> String {
    let (cx, cy) = cfg.center();
    let mut svg = String::new();
    let interval = cfg.ticks.major_interval;
    if interval <= 0.0 {
        return svg;
    }

    let mut v = cfg.min;
    while v <= cfg.max + interval * 0.001 {
        let angle = cfg.value_to_angle(v);
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Major tick
        let r0 = cfg.radius + 2.0;
        let r1 = cfg.radius + 2.0 + cfg.ticks.major_length;
        let x0 = cx + r0 * cos_a;
        let y0 = cy - r0 * sin_a;
        let x1 = cx + r1 * cos_a;
        let y1 = cy - r1 * sin_a;
        let _ = write!(
            svg,
            "<line x1=\"{x0}\" y1=\"{y0}\" x2=\"{x1}\" y2=\"{y1}\" \
             stroke=\"gray\" stroke-width=\"2\" />"
        );

        // Major tick label
        let lr = r1 + 10.0;
        let lx = cx + lr * cos_a;
        let ly = cy - lr * sin_a;
        let fs = cfg.font_size * 0.85;
        let _ = write!(
            svg,
            "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" \
             text-anchor=\"middle\" dominant-baseline=\"middle\">{v:.0}</text>"
        );

        // Minor ticks
        if v + interval <= cfg.max + interval * 0.001 {
            let minor_step = interval / (cfg.ticks.minor_count as f64 + 1.0);
            for mi in 1..=cfg.ticks.minor_count {
                let mv = v + minor_step * mi as f64;
                if mv > cfg.max {
                    break;
                }
                let ma = cfg.value_to_angle(mv);
                let mr0 = cfg.radius + 2.0;
                let mr1 = cfg.radius + 2.0 + cfg.ticks.minor_length;
                let mx0 = cx + mr0 * ma.cos();
                let my0 = cy - mr0 * ma.sin();
                let mx1 = cx + mr1 * ma.cos();
                let my1 = cy - mr1 * ma.sin();
                let _ = write!(
                    svg,
                    "<line x1=\"{mx0}\" y1=\"{my0}\" x2=\"{mx1}\" y2=\"{my1}\" \
                     stroke=\"silver\" stroke-width=\"1\" />"
                );
            }
        }

        v += interval;
    }
    svg
}

/// Render the needle.
fn render_needle(cfg: &GaugeChartConfig) -> String {
    let (cx, cy) = cfg.center();
    let angle = cfg.value_to_angle(cfg.clamped_value());
    let needle_len = cfg.radius - cfg.arc_width - 8.0;
    let tip_x = cx + needle_len * angle.cos();
    let tip_y = cy - needle_len * angle.sin();

    // Needle triangle base
    let base_r = 6.0;
    let perp = angle + PI / 2.0;
    let bx0 = cx + base_r * perp.cos();
    let by0 = cy - base_r * perp.sin();
    let bx1 = cx - base_r * perp.cos();
    let by1 = cy + base_r * perp.sin();

    let mut svg = String::new();
    let _ = write!(
        svg,
        "<polygon points=\"{tip_x},{tip_y} {bx0},{by0} {bx1},{by1}\" fill=\"{}\" />",
        cfg.needle_color
    );
    // Center cap
    let _ = write!(
        svg,
        "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"8\" fill=\"{}\" />",
        cfg.needle_color
    );
    let _ = write!(
        svg,
        "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"4\" fill=\"white\" />"
    );
    svg
}

/// Render the complete gauge chart as an SVG string.
pub fn render_gauge_chart(cfg: &GaugeChartConfig) -> String {
    let (cx, cy) = cfg.center();
    let mut svg = String::with_capacity(4096);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">",
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    // Background arc
    let bg_path = arc_band_path(
        cfg.value_to_angle(cfg.min),
        cfg.value_to_angle(cfg.max),
        cfg.radius - cfg.arc_width,
        cfg.radius,
        cx,
        cy,
    );
    let _ = write!(
        svg,
        "<path d=\"{bg_path}\" fill=\"gainsboro\" stroke=\"none\" />"
    );

    // Zones
    svg.push_str(&render_zones(cfg));

    // Ticks
    svg.push_str(&render_ticks(cfg));

    // Needle
    svg.push_str(&render_needle(cfg));

    // Value display
    if cfg.show_value {
        let vy = cy + 20.0;
        let fs = cfg.font_size * 1.8;
        let _ = write!(
            svg,
            "<text x=\"{cx}\" y=\"{vy}\" font-size=\"{fs}\" \
             text-anchor=\"middle\" font-weight=\"bold\">{:.1}</text>",
            cfg.value
        );
    }

    // Label
    if !cfg.label.is_empty() {
        let ly = cy + 40.0;
        let fs = cfg.font_size;
        let _ = write!(
            svg,
            "<text x=\"{cx}\" y=\"{ly}\" font-size=\"{fs}\" \
             text-anchor=\"middle\" fill=\"gray\">{}</text>",
            cfg.label
        );
    }

    // Min / Max labels
    let min_angle = cfg.value_to_angle(cfg.min);
    let max_angle = cfg.value_to_angle(cfg.max);
    let label_r = cfg.radius + 28.0;
    let fs = cfg.font_size;

    let min_x = cx + label_r * min_angle.cos();
    let min_y = cy - label_r * min_angle.sin();
    let _ = write!(
        svg,
        "<text x=\"{min_x}\" y=\"{min_y}\" font-size=\"{fs}\" text-anchor=\"end\">{:.0}</text>",
        cfg.min
    );

    let max_x = cx + label_r * max_angle.cos();
    let max_y = cy - label_r * max_angle.sin();
    let _ = write!(
        svg,
        "<text x=\"{max_x}\" y=\"{max_y}\" font-size=\"{fs}\" text-anchor=\"start\">{:.0}</text>",
        cfg.max
    );

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_new() {
        let z = GaugeZone::new(0.0, 50.0, "green");
        assert_eq!(z.from, 0.0);
        assert_eq!(z.to, 50.0);
        assert_eq!(z.color, "green");
    }

    #[test]
    fn zone_span() {
        let z = GaugeZone::new(10.0, 30.0, "blue");
        assert!((z.span() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn zone_with_label() {
        let z = GaugeZone::new(0.0, 50.0, "green").with_label("Safe");
        assert_eq!(z.label, "Safe");
    }

    #[test]
    fn standard_zones_cover_range() {
        let zones = standard_zones(0.0, 100.0);
        assert_eq!(zones.len(), 3);
        assert!((zones[0].from).abs() < 1e-9);
        assert!((zones[2].to - 100.0).abs() < 1e-9);
        // Zones are contiguous
        assert!((zones[0].to - zones[1].from).abs() < 1e-9);
        assert!((zones[1].to - zones[2].from).abs() < 1e-9);
    }

    #[test]
    fn config_default_sane() {
        let cfg = GaugeChartConfig::default();
        assert!(cfg.width > 0.0);
        assert!(cfg.height > 0.0);
        assert!(cfg.min < cfg.max);
        assert!(cfg.radius > cfg.arc_width);
    }

    #[test]
    fn value_to_angle_min() {
        let cfg = GaugeChartConfig::default();
        let a = cfg.value_to_angle(cfg.min);
        assert!((a - PI).abs() < 1e-9); // 180 deg
    }

    #[test]
    fn value_to_angle_max() {
        let cfg = GaugeChartConfig::default();
        let a = cfg.value_to_angle(cfg.max);
        assert!(a.abs() < 1e-9); // 0 deg
    }

    #[test]
    fn value_to_angle_mid() {
        let cfg = GaugeChartConfig::default();
        let a = cfg.value_to_angle(50.0);
        assert!((a - PI / 2.0).abs() < 1e-9); // 90 deg
    }

    #[test]
    fn clamped_value_within() {
        let mut cfg = GaugeChartConfig::default();
        cfg.value = 75.0;
        assert!((cfg.clamped_value() - 75.0).abs() < 1e-9);
    }

    #[test]
    fn clamped_value_below() {
        let mut cfg = GaugeChartConfig::default();
        cfg.value = -10.0;
        assert!((cfg.clamped_value() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn clamped_value_above() {
        let mut cfg = GaugeChartConfig::default();
        cfg.value = 200.0;
        assert!((cfg.clamped_value() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn percent_at_half() {
        let mut cfg = GaugeChartConfig::default();
        cfg.value = 50.0;
        assert!((cfg.percent() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn render_produces_svg() {
        let cfg = GaugeChartConfig::default();
        let svg = render_gauge_chart(&cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn render_contains_zones() {
        let cfg = GaugeChartConfig::default();
        let svg = render_gauge_chart(&cfg);
        assert!(svg.contains("green"));
        assert!(svg.contains("gold"));
        assert!(svg.contains("red"));
    }

    #[test]
    fn render_contains_needle() {
        let cfg = GaugeChartConfig::default();
        let svg = render_gauge_chart(&cfg);
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("dimgray"));
    }

    #[test]
    fn render_contains_value_text() {
        let cfg = GaugeChartConfig::default();
        let svg = render_gauge_chart(&cfg);
        assert!(svg.contains("50.0"));
    }

    #[test]
    fn render_with_label() {
        let mut cfg = GaugeChartConfig::default();
        cfg.label = "RPM".into();
        let svg = render_gauge_chart(&cfg);
        assert!(svg.contains("RPM"));
    }

    #[test]
    fn render_ticks_present() {
        let cfg = GaugeChartConfig::default();
        let svg = render_gauge_chart(&cfg);
        assert!(svg.contains("<line"));
    }
}
