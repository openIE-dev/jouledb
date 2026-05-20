//! Pure-Rust SVG chart generation for web dashboards.
//!
//! Replaces D3, Chart.js, Recharts. All functions return SVG strings
//! that can be embedded directly in HTML or rendered via the VDOM.

use std::f64::consts::PI;

// ── Data types ───────────────────────────────────────────────────

/// A single data point.
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
}

/// A named series of data points.
#[derive(Debug, Clone)]
pub struct Series {
    pub name: String,
    pub data: Vec<DataPoint>,
    pub color: String,
}

// ── ChartConfig ──────────────────────────────────────────────────

/// Padding around the plot area.
#[derive(Debug, Clone, Copy)]
pub struct ChartPadding {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Default for ChartPadding {
    fn default() -> Self {
        Self {
            top: 40.0,
            right: 20.0,
            bottom: 40.0,
            left: 50.0,
        }
    }
}

/// Configuration shared across chart types.
#[derive(Debug, Clone)]
pub struct ChartConfig {
    pub width: f64,
    pub height: f64,
    pub padding: ChartPadding,
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub show_grid: bool,
    pub show_legend: bool,
    pub font_size: f64,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            width: 600.0,
            height: 400.0,
            padding: ChartPadding::default(),
            title: None,
            x_label: None,
            y_label: None,
            show_grid: true,
            show_legend: true,
            font_size: 12.0,
        }
    }
}

impl ChartConfig {
    fn plot_width(&self) -> f64 {
        self.width - self.padding.left - self.padding.right
    }

    fn plot_height(&self) -> f64 {
        self.height - self.padding.top - self.padding.bottom
    }
}

// ── LinearScale ──────────────────────────────────────────────────

/// Maps a data range to a pixel range (linear interpolation).
#[derive(Debug, Clone, Copy)]
pub struct LinearScale {
    pub min: f64,
    pub max: f64,
    pub pixel_min: f64,
    pub pixel_max: f64,
}

impl LinearScale {
    /// Auto-range from data values.
    pub fn from_data(values: &[f64], pixel_range: (f64, f64)) -> Self {
        let min = values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let (min, max) = if (max - min).abs() < f64::EPSILON {
            (min - 1.0, max + 1.0)
        } else {
            (min, max)
        };
        Self {
            min,
            max,
            pixel_min: pixel_range.0,
            pixel_max: pixel_range.1,
        }
    }

    /// Map a data value to pixel coordinates.
    pub fn map(&self, value: f64) -> f64 {
        let range = self.max - self.min;
        if range.abs() < f64::EPSILON {
            return (self.pixel_min + self.pixel_max) / 2.0;
        }
        let t = (value - self.min) / range;
        self.pixel_min + t * (self.pixel_max - self.pixel_min)
    }

    /// Map a pixel value back to data coordinates.
    pub fn invert(&self, pixel: f64) -> f64 {
        let range = self.pixel_max - self.pixel_min;
        if range.abs() < f64::EPSILON {
            return (self.min + self.max) / 2.0;
        }
        let t = (pixel - self.pixel_min) / range;
        self.min + t * (self.max - self.min)
    }

    /// Generate nice tick values.
    pub fn ticks(&self, count: usize) -> Vec<f64> {
        if count == 0 {
            return Vec::new();
        }
        let step = (self.max - self.min) / count as f64;
        if step.abs() < f64::EPSILON {
            return vec![self.min];
        }
        let nice_step = nice_step(step);
        let start = (self.min / nice_step).ceil() * nice_step;
        let mut ticks = Vec::new();
        let mut v = start;
        while v <= self.max + nice_step * 0.001 {
            ticks.push(v);
            v += nice_step;
        }
        ticks
    }
}

/// Round a step size to a "nice" value (1, 2, 5, 10, 20, 50, ...).
fn nice_step(rough: f64) -> f64 {
    let exp = rough.abs().log10().floor();
    let frac = rough / 10.0_f64.powf(exp);
    let nice = if frac <= 1.5 {
        1.0
    } else if frac <= 3.5 {
        2.0
    } else if frac <= 7.5 {
        5.0
    } else {
        10.0
    };
    nice * 10.0_f64.powf(exp)
}

// ── Axis rendering ───────────────────────────────────────────────

pub struct Axis;

impl Axis {
    /// Render an X axis SVG group.
    pub fn render_x(scale: &LinearScale, config: &ChartConfig) -> String {
        let y = config.padding.top + config.plot_height();
        let x1 = config.padding.left;
        let x2 = config.padding.left + config.plot_width();
        let mut svg = format!(
            "<g class=\"axis x-axis\"><line x1=\"{x1}\" y1=\"{y}\" x2=\"{x2}\" y2=\"{y}\" stroke=\"#666\" />"
        );
        for tick in scale.ticks(5) {
            let px = scale.map(tick);
            let y2 = y + 5.0;
            svg.push_str(&format!(
                "<line x1=\"{px}\" y1=\"{y}\" x2=\"{px}\" y2=\"{y2}\" stroke=\"#999\" />"
            ));
            let ty = y + 18.0;
            let fs = config.font_size;
            svg.push_str(&format!(
                "<text x=\"{px}\" y=\"{ty}\" text-anchor=\"middle\" font-size=\"{fs}\">{tick:.1}</text>"
            ));
        }
        if let Some(ref label) = config.x_label {
            let lx = config.padding.left + config.plot_width() / 2.0;
            let ly = y + 35.0;
            let fs = config.font_size;
            svg.push_str(&format!(
                "<text x=\"{lx}\" y=\"{ly}\" text-anchor=\"middle\" font-size=\"{fs}\">{label}</text>"
            ));
        }
        svg.push_str("</g>");
        svg
    }

    /// Render a Y axis SVG group.
    pub fn render_y(scale: &LinearScale, config: &ChartConfig) -> String {
        let x = config.padding.left;
        let y1 = config.padding.top;
        let y2 = config.padding.top + config.plot_height();
        let mut svg = format!(
            "<g class=\"axis y-axis\"><line x1=\"{x}\" y1=\"{y1}\" x2=\"{x}\" y2=\"{y2}\" stroke=\"#666\" />"
        );
        for tick in scale.ticks(5) {
            let py = scale.map(tick);
            let lx = x - 5.0;
            svg.push_str(&format!(
                "<line x1=\"{lx}\" y1=\"{py}\" x2=\"{x}\" y2=\"{py}\" stroke=\"#999\" />"
            ));
            let tx = x - 8.0;
            let fs = config.font_size;
            svg.push_str(&format!(
                "<text x=\"{tx}\" y=\"{py}\" text-anchor=\"end\" dominant-baseline=\"middle\" font-size=\"{fs}\">{tick:.1}</text>"
            ));
        }
        if let Some(ref label) = config.y_label {
            let ly = config.padding.top + config.plot_height() / 2.0;
            let fs = config.font_size;
            svg.push_str(&format!(
                "<text x=\"15\" y=\"{ly}\" text-anchor=\"middle\" font-size=\"{fs}\" transform=\"rotate(-90,15,{ly})\">{label}</text>"
            ));
        }
        svg.push_str("</g>");
        svg
    }
}

// ── Grid ─────────────────────────────────────────────────────────

fn render_grid(
    x_scale: &LinearScale,
    y_scale: &LinearScale,
    config: &ChartConfig,
) -> String {
    if !config.show_grid {
        return String::new();
    }
    let mut svg = String::from("<g class=\"grid\">");
    let gx1 = config.padding.left;
    let gx2 = config.padding.left + config.plot_width();
    for tick in y_scale.ticks(5) {
        let py = y_scale.map(tick);
        svg.push_str(&format!(
            "<line x1=\"{gx1}\" y1=\"{py}\" x2=\"{gx2}\" y2=\"{py}\" stroke=\"#eee\" />"
        ));
    }
    let gy1 = config.padding.top;
    let gy2 = config.padding.top + config.plot_height();
    for tick in x_scale.ticks(5) {
        let px = x_scale.map(tick);
        svg.push_str(&format!(
            "<line x1=\"{px}\" y1=\"{gy1}\" x2=\"{px}\" y2=\"{gy2}\" stroke=\"#eee\" />"
        ));
    }
    svg.push_str("</g>");
    svg
}

// ── Legend ────────────────────────────────────────────────────────

fn render_legend(series: &[Series], config: &ChartConfig) -> String {
    if !config.show_legend || series.is_empty() {
        return String::new();
    }
    let mut svg = format!(
        r#"<g class="legend" transform="translate({},{})">"#,
        config.padding.left,
        config.padding.top - 20.0
    );
    for (i, s) in series.iter().enumerate() {
        let x = i as f64 * 120.0;
        svg.push_str(&format!(
            r#"<rect x="{x}" y="0" width="12" height="12" fill="{}" />"#,
            s.color
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="10" font-size="{}">{}</text>"#,
            x + 16.0,
            config.font_size,
            s.name
        ));
    }
    svg.push_str("</g>");
    svg
}

// ── Title ────────────────────────────────────────────────────────

fn render_title(config: &ChartConfig) -> String {
    config.title.as_ref().map_or_else(String::new, |t| {
        format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-size="{}" font-weight="bold">{t}</text>"#,
            config.width / 2.0,
            config.padding.top / 2.0,
            config.font_size + 4.0
        )
    })
}

// ── SVG wrapper ──────────────────────────────────────────────────

fn svg_open(config: &ChartConfig) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        config.width, config.height, config.width, config.height
    )
}

// ── Scales from series ───────────────────────────────────────────

fn scales_from_series(series: &[Series], config: &ChartConfig) -> (LinearScale, LinearScale) {
    let xs: Vec<f64> = series.iter().flat_map(|s| s.data.iter().map(|d| d.x)).collect();
    let ys: Vec<f64> = series.iter().flat_map(|s| s.data.iter().map(|d| d.y)).collect();
    let x_scale = if xs.is_empty() {
        LinearScale { min: 0.0, max: 1.0, pixel_min: config.padding.left, pixel_max: config.padding.left + config.plot_width() }
    } else {
        LinearScale::from_data(&xs, (config.padding.left, config.padding.left + config.plot_width()))
    };
    // Y axis is inverted (pixel 0 is top)
    let y_scale = if ys.is_empty() {
        LinearScale { min: 0.0, max: 1.0, pixel_min: config.padding.top + config.plot_height(), pixel_max: config.padding.top }
    } else {
        LinearScale::from_data(&ys, (config.padding.top + config.plot_height(), config.padding.top))
    };
    (x_scale, y_scale)
}

// ── Chart renderers ──────────────────────────────────────────────

/// Render a line chart as an SVG string.
pub fn render_line_chart(series: &[Series], config: &ChartConfig) -> String {
    let (x_scale, y_scale) = scales_from_series(series, config);
    let mut svg = svg_open(config);
    svg.push_str(&render_title(config));
    svg.push_str(&render_grid(&x_scale, &y_scale, config));
    svg.push_str(&Axis::render_x(&x_scale, config));
    svg.push_str(&Axis::render_y(&y_scale, config));

    for s in series {
        if s.data.is_empty() {
            continue;
        }
        let points: String = s
            .data
            .iter()
            .map(|d| format!("{},{}", x_scale.map(d.x), y_scale.map(d.y)))
            .collect::<Vec<_>>()
            .join(" ");
        svg.push_str(&format!(
            r#"<polyline points="{points}" fill="none" stroke="{}" stroke-width="2" />"#,
            s.color
        ));
    }

    svg.push_str(&render_legend(series, config));
    svg.push_str("</svg>");
    svg
}

/// Render a bar chart as an SVG string.
pub fn render_bar_chart(series: &[Series], config: &ChartConfig) -> String {
    let (x_scale, y_scale) = scales_from_series(series, config);
    let mut svg = svg_open(config);
    svg.push_str(&render_title(config));
    svg.push_str(&render_grid(&x_scale, &y_scale, config));
    svg.push_str(&Axis::render_x(&x_scale, config));
    svg.push_str(&Axis::render_y(&y_scale, config));

    let n_series = series.len().max(1);
    let bar_group_width = if series.is_empty() || series[0].data.is_empty() {
        20.0
    } else {
        config.plot_width() / series[0].data.len() as f64 * 0.8
    };
    let bar_width = bar_group_width / n_series as f64;
    let baseline = y_scale.map(0.0_f64.max(y_scale.min));

    for (si, s) in series.iter().enumerate() {
        for d in &s.data {
            let cx = x_scale.map(d.x);
            let x = cx - bar_group_width / 2.0 + si as f64 * bar_width;
            let y_top = y_scale.map(d.y);
            let (rect_y, rect_h) = if y_top < baseline {
                (y_top, baseline - y_top)
            } else {
                (baseline, y_top - baseline)
            };
            svg.push_str(&format!(
                r#"<rect x="{x}" y="{rect_y}" width="{bar_width}" height="{rect_h}" fill="{}" />"#,
                s.color
            ));
        }
    }

    svg.push_str(&render_legend(series, config));
    svg.push_str("</svg>");
    svg
}

/// Render a scatter chart as an SVG string.
pub fn render_scatter_chart(series: &[Series], config: &ChartConfig) -> String {
    let (x_scale, y_scale) = scales_from_series(series, config);
    let mut svg = svg_open(config);
    svg.push_str(&render_title(config));
    svg.push_str(&render_grid(&x_scale, &y_scale, config));
    svg.push_str(&Axis::render_x(&x_scale, config));
    svg.push_str(&Axis::render_y(&y_scale, config));

    for s in series {
        for d in &s.data {
            let cx = x_scale.map(d.x);
            let cy = y_scale.map(d.y);
            svg.push_str(&format!(
                r#"<circle cx="{cx}" cy="{cy}" r="4" fill="{}" />"#,
                s.color
            ));
        }
    }

    svg.push_str(&render_legend(series, config));
    svg.push_str("</svg>");
    svg
}

/// Render an area chart as an SVG string.
pub fn render_area_chart(series: &[Series], config: &ChartConfig) -> String {
    let (x_scale, y_scale) = scales_from_series(series, config);
    let mut svg = svg_open(config);
    svg.push_str(&render_title(config));
    svg.push_str(&render_grid(&x_scale, &y_scale, config));
    svg.push_str(&Axis::render_x(&x_scale, config));
    svg.push_str(&Axis::render_y(&y_scale, config));

    let baseline = y_scale.map(y_scale.min.max(0.0));

    for s in series {
        if s.data.is_empty() {
            continue;
        }
        let mut points = String::new();
        // Start at baseline
        let first_x = x_scale.map(s.data[0].x);
        points.push_str(&format!("{first_x},{baseline} "));
        for d in &s.data {
            points.push_str(&format!("{},{} ", x_scale.map(d.x), y_scale.map(d.y)));
        }
        let last_x = x_scale.map(s.data.last().unwrap().x);
        points.push_str(&format!("{last_x},{baseline}"));
        svg.push_str(&format!(
            r#"<polygon points="{points}" fill="{}" fill-opacity="0.3" stroke="{}" stroke-width="2" />"#,
            s.color, s.color
        ));
    }

    svg.push_str(&render_legend(series, config));
    svg.push_str("</svg>");
    svg
}

/// Render a pie chart from `(value, label, color)` slices.
pub fn render_pie_chart(slices: &[(f64, String, String)], config: &ChartConfig) -> String {
    render_pie_or_donut(slices, config, 0.0)
}

/// Render a donut chart (pie with inner cutout).
pub fn render_donut_chart(slices: &[(f64, String, String)], config: &ChartConfig) -> String {
    render_pie_or_donut(slices, config, 0.6)
}

fn render_pie_or_donut(
    slices: &[(f64, String, String)],
    config: &ChartConfig,
    inner_ratio: f64,
) -> String {
    let mut svg = svg_open(config);
    svg.push_str(&render_title(config));

    let cx = config.width / 2.0;
    let cy = config.padding.top + config.plot_height() / 2.0;
    let r = config.plot_width().min(config.plot_height()) / 2.0 * 0.9;
    let inner_r = r * inner_ratio;

    let total: f64 = slices.iter().map(|(v, _, _)| v).sum();
    if total <= 0.0 {
        svg.push_str("</svg>");
        return svg;
    }

    let mut angle = -PI / 2.0;
    for (value, label, color) in slices {
        let sweep = value / total * 2.0 * PI;
        let start_angle = angle;
        let end_angle = angle + sweep;

        let large_arc = if sweep > PI { 1 } else { 0 };

        if inner_ratio > 0.0 {
            // Donut: path with arc and inner cutout
            let x1 = cx + r * start_angle.cos();
            let y1 = cy + r * start_angle.sin();
            let x2 = cx + r * end_angle.cos();
            let y2 = cy + r * end_angle.sin();
            let ix1 = cx + inner_r * end_angle.cos();
            let iy1 = cy + inner_r * end_angle.sin();
            let ix2 = cx + inner_r * start_angle.cos();
            let iy2 = cy + inner_r * start_angle.sin();

            svg.push_str(&format!(
                r#"<path d="M {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2} L {ix1} {iy1} A {inner_r} {inner_r} 0 {large_arc} 0 {ix2} {iy2} Z" fill="{color}" />"#,
            ));
        } else {
            // Pie: path with arc to center
            let x1 = cx + r * start_angle.cos();
            let y1 = cy + r * start_angle.sin();
            let x2 = cx + r * end_angle.cos();
            let y2 = cy + r * end_angle.sin();

            svg.push_str(&format!(
                r#"<path d="M {cx} {cy} L {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2} Z" fill="{color}" />"#,
            ));
        }

        // Percentage label
        let mid_angle = start_angle + sweep / 2.0;
        let label_r = if inner_ratio > 0.0 {
            (r + inner_r) / 2.0
        } else {
            r * 0.65
        };
        let lx = cx + label_r * mid_angle.cos();
        let ly = cy + label_r * mid_angle.sin();
        let pct = value / total * 100.0;
        svg.push_str(&format!(
            r#"<text x="{lx}" y="{ly}" text-anchor="middle" dominant-baseline="middle" font-size="{}">{label} {pct:.0}%</text>"#,
            config.font_size
        ));

        angle = end_angle;
    }

    svg.push_str("</svg>");
    svg
}

/// Render a sparkline (minimal line chart, no axes).
pub fn render_sparkline(values: &[f64], width: f64, height: f64, color: &str) -> String {
    if values.is_empty() {
        return format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>"#
        );
    }
    let x_step = if values.len() > 1 {
        width / (values.len() - 1) as f64
    } else {
        width
    };
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < f64::EPSILON {
        1.0
    } else {
        max - min
    };

    let points: String = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = i as f64 * x_step;
            let y = height - (v - min) / range * height;
            format!("{x},{y}")
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" /></svg>"#
    )
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_series() -> Vec<Series> {
        vec![Series {
            name: "Sales".into(),
            data: vec![
                DataPoint { x: 1.0, y: 10.0, label: None },
                DataPoint { x: 2.0, y: 20.0, label: None },
                DataPoint { x: 3.0, y: 15.0, label: None },
                DataPoint { x: 4.0, y: 30.0, label: None },
            ],
            color: "#3498db".into(),
        }]
    }

    fn default_config() -> ChartConfig {
        ChartConfig::default()
    }

    #[test]
    fn linear_scale_maps_correctly() {
        let scale = LinearScale {
            min: 0.0,
            max: 100.0,
            pixel_min: 0.0,
            pixel_max: 500.0,
        };
        assert_eq!(scale.map(0.0), 0.0);
        assert_eq!(scale.map(100.0), 500.0);
        assert_eq!(scale.map(50.0), 250.0);
    }

    #[test]
    fn linear_scale_invert() {
        let scale = LinearScale {
            min: 0.0,
            max: 100.0,
            pixel_min: 0.0,
            pixel_max: 500.0,
        };
        assert!((scale.invert(250.0) - 50.0).abs() < 0.001);
        assert!((scale.invert(0.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn ticks_are_reasonable() {
        let scale = LinearScale {
            min: 0.0,
            max: 100.0,
            pixel_min: 0.0,
            pixel_max: 500.0,
        };
        let ticks = scale.ticks(5);
        assert!(!ticks.is_empty());
        assert!(ticks.len() <= 10);
        for &t in &ticks {
            assert!(t >= 0.0 && t <= 100.0 + 0.1);
        }
    }

    #[test]
    fn line_chart_has_polyline() {
        let svg = render_line_chart(&sample_series(), &default_config());
        assert!(svg.contains("<polyline"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("#3498db"));
    }

    #[test]
    fn bar_chart_has_rects() {
        let svg = render_bar_chart(&sample_series(), &default_config());
        assert!(svg.contains("<rect"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn scatter_has_circles() {
        let svg = render_scatter_chart(&sample_series(), &default_config());
        assert!(svg.contains("<circle"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn pie_chart_arcs() {
        let slices = vec![
            (30.0, "A".into(), "#e74c3c".into()),
            (70.0, "B".into(), "#2ecc71".into()),
        ];
        let svg = render_pie_chart(&slices, &default_config());
        assert!(svg.contains("<path"));
        assert!(svg.contains("30%"));
        assert!(svg.contains("70%"));
    }

    #[test]
    fn legend_rendered() {
        let config = ChartConfig {
            show_legend: true,
            ..default_config()
        };
        let svg = render_line_chart(&sample_series(), &config);
        assert!(svg.contains("legend"));
        assert!(svg.contains("Sales"));
    }

    #[test]
    fn empty_series_handles_gracefully() {
        let empty: Vec<Series> = Vec::new();
        let svg = render_line_chart(&empty, &default_config());
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));

        let svg2 = render_bar_chart(&empty, &default_config());
        assert!(svg2.contains("</svg>"));
    }

    #[test]
    fn sparkline_minimal() {
        let svg = render_sparkline(&[1.0, 3.0, 2.0, 5.0], 100.0, 30.0, "#333");
        assert!(svg.contains("<polyline"));
        assert!(svg.contains("#333"));
        assert!(!svg.contains("axis"));
        assert!(!svg.contains("grid"));
    }

    #[test]
    fn scale_from_data_auto_ranges() {
        let vals = vec![10.0, 20.0, 30.0];
        let scale = LinearScale::from_data(&vals, (0.0, 300.0));
        assert_eq!(scale.min, 10.0);
        assert_eq!(scale.max, 30.0);
        assert_eq!(scale.map(10.0), 0.0);
        assert_eq!(scale.map(30.0), 300.0);
    }

    #[test]
    fn donut_has_inner_cutout() {
        let slices = vec![
            (50.0, "X".into(), "#f00".into()),
            (50.0, "Y".into(), "#0f0".into()),
        ];
        let svg = render_donut_chart(&slices, &default_config());
        assert!(svg.contains("<path"));
        // Donut paths have the inner arc (A ... 0 pattern for inner radius)
        // Both slices should be present
        assert!(svg.contains("X 50%"));
        assert!(svg.contains("Y 50%"));
    }

    #[test]
    fn area_chart_has_polygon() {
        let svg = render_area_chart(&sample_series(), &default_config());
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("fill-opacity"));
    }
}
