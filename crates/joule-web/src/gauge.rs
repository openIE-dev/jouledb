//! Gauge / meter visualization: arc gauges, progress rings, multi-needle,
//! digital readouts.  Pure Rust SVG output.

use std::f64::consts::PI;

// ── Color band ───────────────────────────────────────────────────

/// A coloured band on the gauge arc.
#[derive(Debug, Clone)]
pub struct ColorBand {
    /// Start value (data units).
    pub from: f64,
    /// End value (data units).
    pub to: f64,
    pub color: String,
}

// ── Gauge config ─────────────────────────────────────────────────

/// Configuration for a standard arc gauge.
#[derive(Debug, Clone)]
pub struct GaugeConfig {
    pub min: f64,
    pub max: f64,
    pub value: f64,
    /// Arc start angle in degrees (0 = 3 o'clock, counter-clockwise).
    pub start_angle_deg: f64,
    /// Arc end angle in degrees.
    pub end_angle_deg: f64,
    pub color_bands: Vec<ColorBand>,
    pub width: f64,
    pub height: f64,
    pub arc_width: f64,
    pub font_size: f64,
    /// Major tick spacing in data units.
    pub major_tick_interval: f64,
    /// Minor ticks between each pair of major ticks.
    pub minor_ticks_per_major: usize,
}

impl Default for GaugeConfig {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 100.0,
            value: 50.0,
            start_angle_deg: 225.0,
            end_angle_deg: -45.0,
            color_bands: vec![
                ColorBand { from: 0.0, to: 33.0, color: "#2ecc71".into() },
                ColorBand { from: 33.0, to: 66.0, color: "#f1c40f".into() },
                ColorBand { from: 66.0, to: 100.0, color: "#e74c3c".into() },
            ],
            width: 300.0,
            height: 200.0,
            arc_width: 20.0,
            font_size: 12.0,
            major_tick_interval: 20.0,
            minor_ticks_per_major: 4,
        }
    }
}

impl GaugeConfig {
    pub fn center(&self) -> (f64, f64) {
        (self.width / 2.0, self.height * 0.75)
    }

    pub fn radius(&self) -> f64 {
        self.width.min(self.height) * 0.4
    }
}

// ── Value to angle ───────────────────────────────────────────────

/// Map a data value to an angle (degrees) on the gauge arc.
pub fn value_to_angle(value: f64, cfg: &GaugeConfig) -> f64 {
    let range = (cfg.max - cfg.min).max(f64::EPSILON);
    let t = ((value - cfg.min) / range).clamp(0.0, 1.0);
    cfg.start_angle_deg + t * (cfg.end_angle_deg - cfg.start_angle_deg)
}

/// Compute the (x, y) position of the pointer tip at a given value.
pub fn pointer_position(value: f64, cfg: &GaugeConfig, length: f64) -> (f64, f64) {
    let angle = value_to_angle(value, cfg).to_radians();
    let (cx, cy) = cfg.center();
    (cx + length * angle.cos(), cy - length * angle.sin())
}

// ── Tick marks ───────────────────────────────────────────────────

/// A tick mark with its position and whether it is a major tick.
#[derive(Debug, Clone)]
pub struct TickMark {
    pub value: f64,
    pub angle_deg: f64,
    pub is_major: bool,
}

/// Generate tick marks for the gauge.
pub fn tick_marks(cfg: &GaugeConfig) -> Vec<TickMark> {
    let mut ticks = Vec::new();
    let interval = cfg.major_tick_interval.max(f64::EPSILON);
    let minor_step = interval / (cfg.minor_ticks_per_major as f64 + 1.0);
    let mut v = cfg.min;

    while v <= cfg.max + interval * 0.001 {
        ticks.push(TickMark {
            value: v,
            angle_deg: value_to_angle(v, cfg),
            is_major: true,
        });
        // Minor ticks
        for m in 1..=cfg.minor_ticks_per_major {
            let mv = v + m as f64 * minor_step;
            if mv < cfg.max + minor_step * 0.001 && mv < v + interval - minor_step * 0.5 {
                ticks.push(TickMark {
                    value: mv,
                    angle_deg: value_to_angle(mv, cfg),
                    is_major: false,
                });
            }
        }
        v += interval;
    }
    ticks
}

/// Compute a label position on the arc at a given angle.
pub fn label_position_on_arc(
    angle_deg: f64,
    cx: f64,
    cy: f64,
    radius: f64,
    offset: f64,
) -> (f64, f64) {
    let rad = angle_deg.to_radians();
    (
        cx + (radius + offset) * rad.cos(),
        cy - (radius + offset) * rad.sin(),
    )
}

// ── Digital readout ──────────────────────────────────────────────

/// Format a value for digital display (fixed 1 decimal, padded).
pub fn digital_readout(value: f64, unit: &str) -> String {
    format!("{value:.1} {unit}")
}

// ── Progress ring ────────────────────────────────────────────────

/// A thin arc gauge (progress ring).
#[derive(Debug, Clone)]
pub struct ProgressRing {
    pub value: f64,
    pub max: f64,
    pub radius: f64,
    pub stroke_width: f64,
    pub color: String,
    pub background_color: String,
}

impl Default for ProgressRing {
    fn default() -> Self {
        Self {
            value: 0.0,
            max: 100.0,
            radius: 50.0,
            stroke_width: 8.0,
            color: "#3498db".into(),
            background_color: "#eee".into(),
        }
    }
}

/// Render a progress ring as SVG.
pub fn render_progress_ring(ring: &ProgressRing, cx: f64, cy: f64) -> String {
    let circumference = 2.0 * PI * ring.radius;
    let t = (ring.value / ring.max.max(f64::EPSILON)).clamp(0.0, 1.0);
    let dash = circumference * t;
    let gap = circumference - dash;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" />",
        ring.radius, ring.background_color, ring.stroke_width
    ));
    svg.push_str(&format!(
        "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" \
         stroke-dasharray=\"{dash} {gap}\" stroke-dashoffset=\"{}\" \
         transform=\"rotate(-90 {cx} {cy})\" />",
        ring.radius,
        ring.color,
        ring.stroke_width,
        0.0
    ));
    svg
}

// ── Multi-needle gauge ───────────────────────────────────────────

/// A needle on a multi-needle gauge.
#[derive(Debug, Clone)]
pub struct Needle {
    pub value: f64,
    pub color: String,
    pub length_ratio: f64,
}

/// Render SVG lines for multiple needles on a gauge.
pub fn render_needles(needles: &[Needle], cfg: &GaugeConfig) -> String {
    let (cx, cy) = cfg.center();
    let r = cfg.radius();
    let mut svg = String::new();
    for n in needles {
        let length = r * n.length_ratio;
        let (tx, ty) = pointer_position(n.value, cfg, length);
        svg.push_str(&format!(
            "<line x1=\"{cx}\" y1=\"{cy}\" x2=\"{tx}\" y2=\"{ty}\" stroke=\"{}\" stroke-width=\"2\" stroke-linecap=\"round\" />",
            n.color
        ));
    }
    // Centre dot
    svg.push_str(&format!(
        "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"4\" fill=\"#333\" />"
    ));
    svg
}

// ── Full gauge rendering ─────────────────────────────────────────

/// Render a complete gauge as an SVG string.
pub fn render_gauge(cfg: &GaugeConfig) -> String {
    let (cx, cy) = cfg.center();
    let r = cfg.radius();
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    // Colour bands
    for band in &cfg.color_bands {
        let sa = value_to_angle(band.from.max(cfg.min), cfg);
        let ea = value_to_angle(band.to.min(cfg.max), cfg);
        let d = arc_path_thick(cx, cy, r, cfg.arc_width, sa, ea);
        svg.push_str(&format!("<path d=\"{d}\" fill=\"{}\" />", band.color));
    }

    // Tick marks
    for tick in tick_marks(cfg) {
        let rad = tick.angle_deg.to_radians();
        let len = if tick.is_major { 10.0 } else { 5.0 };
        let inner = r - cfg.arc_width / 2.0 - len;
        let outer = r - cfg.arc_width / 2.0;
        let x1 = cx + inner * rad.cos();
        let y1 = cy - inner * rad.sin();
        let x2 = cx + outer * rad.cos();
        let y2 = cy - outer * rad.sin();
        let sw = if tick.is_major { "1.5" } else { "0.75" };
        svg.push_str(&format!(
            "<line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" stroke=\"#333\" stroke-width=\"{sw}\" />"
        ));
        if tick.is_major {
            let (lx, ly) = label_position_on_arc(tick.angle_deg, cx, cy, r - cfg.arc_width / 2.0, -20.0);
            let fs = cfg.font_size;
            svg.push_str(&format!(
                "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" text-anchor=\"middle\" dominant-baseline=\"middle\">{:.0}</text>",
                tick.value
            ));
        }
    }

    // Pointer
    let needle_len = r - cfg.arc_width / 2.0 - 5.0;
    let (tx, ty) = pointer_position(cfg.value, cfg, needle_len);
    svg.push_str(&format!(
        "<line x1=\"{cx}\" y1=\"{cy}\" x2=\"{tx}\" y2=\"{ty}\" stroke=\"#333\" stroke-width=\"2\" stroke-linecap=\"round\" />"
    ));
    svg.push_str(&format!(
        "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"4\" fill=\"#333\" />"
    ));

    // Digital readout
    let fs = cfg.font_size + 4.0;
    svg.push_str(&format!(
        "<text x=\"{cx}\" y=\"{}\" font-size=\"{fs}\" text-anchor=\"middle\" font-weight=\"bold\">{:.1}</text>",
        cy + 20.0, cfg.value
    ));

    svg.push_str("</svg>");
    svg
}

/// Generate an arc path for a thick band (annular sector).
fn arc_path_thick(
    cx: f64,
    cy: f64,
    outer_r: f64,
    width: f64,
    start_deg: f64,
    end_deg: f64,
) -> String {
    let inner_r = outer_r - width;
    let s = start_deg.to_radians();
    let e = end_deg.to_radians();
    let sweep = (end_deg - start_deg).abs();
    let large = if sweep > 180.0 { 1 } else { 0 };

    // The direction for the arc depends on whether end < start
    let (sweep_outer, sweep_inner) = if end_deg < start_deg {
        (1, 0) // clockwise outer, counter-clockwise inner
    } else {
        (0, 1)
    };

    let ox1 = cx + outer_r * s.cos();
    let oy1 = cy - outer_r * s.sin();
    let ox2 = cx + outer_r * e.cos();
    let oy2 = cy - outer_r * e.sin();
    let ix1 = cx + inner_r * e.cos();
    let iy1 = cy - inner_r * e.sin();
    let ix2 = cx + inner_r * s.cos();
    let iy2 = cy - inner_r * s.sin();

    format!(
        "M {ox1} {oy1} A {outer_r} {outer_r} 0 {large} {sweep_outer} {ox2} {oy2} \
         L {ix1} {iy1} A {inner_r} {inner_r} 0 {large} {sweep_inner} {ix2} {iy2} Z"
    )
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_to_angle_min_max() {
        let cfg = GaugeConfig::default();
        let a_min = value_to_angle(cfg.min, &cfg);
        let a_max = value_to_angle(cfg.max, &cfg);
        assert!((a_min - cfg.start_angle_deg).abs() < 1e-9);
        assert!((a_max - cfg.end_angle_deg).abs() < 1e-9);
    }

    #[test]
    fn value_to_angle_mid() {
        let cfg = GaugeConfig::default();
        let a = value_to_angle(50.0, &cfg);
        let expected = (cfg.start_angle_deg + cfg.end_angle_deg) / 2.0;
        assert!((a - expected).abs() < 1e-9);
    }

    #[test]
    fn value_to_angle_clamps() {
        let cfg = GaugeConfig::default();
        let below = value_to_angle(-50.0, &cfg);
        let above = value_to_angle(200.0, &cfg);
        assert!((below - cfg.start_angle_deg).abs() < 1e-9);
        assert!((above - cfg.end_angle_deg).abs() < 1e-9);
    }

    #[test]
    fn pointer_position_at_start() {
        let cfg = GaugeConfig::default();
        let (x, y) = pointer_position(cfg.min, &cfg, 100.0);
        let (cx, cy) = cfg.center();
        let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        assert!((dist - 100.0).abs() < 1e-6);
    }

    #[test]
    fn tick_marks_include_min_and_max() {
        let cfg = GaugeConfig::default();
        let ticks = tick_marks(&cfg);
        let major_values: Vec<f64> = ticks.iter().filter(|t| t.is_major).map(|t| t.value).collect();
        assert!(major_values.iter().any(|v| (v - cfg.min).abs() < 1e-9));
        assert!(major_values.iter().any(|v| (v - cfg.max).abs() < 1e-9));
    }

    #[test]
    fn tick_marks_has_minor() {
        let cfg = GaugeConfig::default();
        let ticks = tick_marks(&cfg);
        assert!(ticks.iter().any(|t| !t.is_major));
    }

    #[test]
    fn label_position_offset() {
        let (x, y) = label_position_on_arc(0.0, 100.0, 100.0, 50.0, 10.0);
        assert!((x - 160.0).abs() < 1e-9);
        assert!((y - 100.0).abs() < 1e-9);
    }

    #[test]
    fn digital_readout_format() {
        let s = digital_readout(72.345, "km/h");
        assert_eq!(s, "72.3 km/h");
    }

    #[test]
    fn progress_ring_svg() {
        let ring = ProgressRing {
            value: 75.0,
            max: 100.0,
            ..ProgressRing::default()
        };
        let svg = render_progress_ring(&ring, 60.0, 60.0);
        assert!(svg.contains("<circle"));
        assert!(svg.contains("stroke-dasharray"));
        assert!(svg.contains(&ring.color));
    }

    #[test]
    fn progress_ring_zero() {
        let ring = ProgressRing {
            value: 0.0,
            ..ProgressRing::default()
        };
        let svg = render_progress_ring(&ring, 50.0, 50.0);
        assert!(svg.contains("stroke-dasharray=\"0"));
    }

    #[test]
    fn render_needles_multiple() {
        let cfg = GaugeConfig::default();
        let needles = vec![
            Needle { value: 30.0, color: "#f00".into(), length_ratio: 0.9 },
            Needle { value: 70.0, color: "#00f".into(), length_ratio: 0.6 },
        ];
        let svg = render_needles(&needles, &cfg);
        assert_eq!(svg.matches("<line").count(), 2);
        assert!(svg.contains("#f00"));
        assert!(svg.contains("#00f"));
        // Centre dot
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn render_gauge_full() {
        let cfg = GaugeConfig::default();
        let svg = render_gauge(&cfg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<path"));  // colour bands
        assert!(svg.contains("<line"));  // ticks + pointer
        assert!(svg.contains("<text"));  // labels + readout
    }

    #[test]
    fn render_gauge_custom_value() {
        let cfg = GaugeConfig {
            value: 85.0,
            ..GaugeConfig::default()
        };
        let svg = render_gauge(&cfg);
        assert!(svg.contains("85.0"));
    }
}
