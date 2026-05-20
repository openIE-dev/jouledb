//! Sparkline rendering: line sparkline, bar sparkline, win/loss (binary),
//! reference lines (min/max/avg/median), normal range band, SVG path
//! generation, responsive scaling, data point highlighting.  Pure Rust SVG.

use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// Type of sparkline to render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparklineType {
    Line,
    Bar,
    /// Win/loss: positive = win (up), negative/zero = loss (down).
    WinLoss,
}

/// Which reference lines to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceLine {
    Min,
    Max,
    Average,
    Median,
}

/// A data point that should be highlighted (e.g., first, last, min, max).
#[derive(Debug, Clone)]
pub struct Highlight {
    pub index: usize,
    pub color: String,
    pub radius: f64,
}

/// Normal range band — shaded background between two values.
#[derive(Debug, Clone)]
pub struct NormalBand {
    pub lo: f64,
    pub hi: f64,
    pub color: String,
    pub opacity: f64,
}

/// Configuration for a sparkline.
#[derive(Debug, Clone)]
pub struct SparklineConfig {
    pub width: f64,
    pub height: f64,
    pub spark_type: SparklineType,
    pub color: String,
    pub line_width: f64,
    pub reference_lines: Vec<ReferenceLine>,
    pub reference_color: String,
    pub highlights: Vec<Highlight>,
    pub normal_band: Option<NormalBand>,
    /// Padding inside the SVG.
    pub padding: f64,
}

impl Default for SparklineConfig {
    fn default() -> Self {
        Self {
            width: 200.0,
            height: 40.0,
            spark_type: SparklineType::Line,
            color: "#3498db".into(),
            line_width: 1.5,
            reference_lines: Vec::new(),
            reference_color: "#e74c3c".into(),
            highlights: Vec::new(),
            normal_band: None,
            padding: 2.0,
        }
    }
}

// ── Statistics ───────────────────────────────────────────────────

fn data_min(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::INFINITY, f64::min)
}

fn data_max(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn data_avg(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn data_median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[mid - 1] + s[mid]) / 2.0
    } else {
        s[mid]
    }
}

/// Get the reference value for a reference line type.
pub fn reference_value(data: &[f64], rl: ReferenceLine) -> f64 {
    match rl {
        ReferenceLine::Min => data_min(data),
        ReferenceLine::Max => data_max(data),
        ReferenceLine::Average => data_avg(data),
        ReferenceLine::Median => data_median(data),
    }
}

// ── Scaling ──────────────────────────────────────────────────────

/// Map data to plot coordinates.
struct Scale {
    x_scale: f64,
    y_scale: f64,
    y_min: f64,
    x_off: f64,
    y_off: f64,
    plot_h: f64,
}

impl Scale {
    fn new(data: &[f64], config: &SparklineConfig) -> Self {
        let n = data.len().max(1);
        let plot_w = config.width - 2.0 * config.padding;
        let plot_h = config.height - 2.0 * config.padding;
        let y_min = data_min(data);
        let y_max = data_max(data);
        let y_range = (y_max - y_min).max(f64::EPSILON);
        Self {
            x_scale: plot_w / (n - 1).max(1) as f64,
            y_scale: plot_h / y_range,
            y_min,
            x_off: config.padding,
            y_off: config.padding,
            plot_h,
        }
    }

    fn x(&self, i: usize) -> f64 {
        self.x_off + i as f64 * self.x_scale
    }

    fn y(&self, v: f64) -> f64 {
        self.y_off + self.plot_h - (v - self.y_min) * self.y_scale
    }
}

// ── SVG generation ───────────────────────────────────────────────

/// Generate SVG path data for a line sparkline.
pub fn line_path(data: &[f64], config: &SparklineConfig) -> String {
    if data.is_empty() {
        return String::new();
    }
    let scale = Scale::new(data, config);
    let mut d = String::new();
    for (i, v) in data.iter().enumerate() {
        let cmd = if i == 0 { "M" } else { "L" };
        let _ = write!(d, "{cmd}{:.2},{:.2} ", scale.x(i), scale.y(*v));
    }
    d
}

/// Render a sparkline to SVG.
pub fn render_svg(data: &[f64], config: &SparklineConfig) -> String {
    if data.is_empty() {
        return format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}"></svg>"#,
            config.width, config.height,
        );
    }

    let scale = Scale::new(data, config);
    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#,
        config.width, config.height,
    );

    // Normal band.
    if let Some(band) = &config.normal_band {
        let y_top = scale.y(band.hi);
        let y_bot = scale.y(band.lo);
        let _ = write!(
            svg,
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" opacity="{}"/>"#,
            config.padding,
            y_top,
            config.width - 2.0 * config.padding,
            (y_bot - y_top).abs(),
            band.color,
            band.opacity,
        );
    }

    // Reference lines.
    for rl in &config.reference_lines {
        let v = reference_value(data, *rl);
        let y = scale.y(v);
        let _ = write!(
            svg,
            r#"<line x1="{:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="{}" stroke-width="0.5" stroke-dasharray="3,2"/>"#,
            config.padding,
            config.width - config.padding,
            config.reference_color,
        );
    }

    match config.spark_type {
        SparklineType::Line => {
            let path = line_path(data, config);
            let _ = write!(
                svg,
                r#"<path d="{path}" fill="none" stroke="{}" stroke-width="{}"/>"#,
                config.color, config.line_width,
            );
        }
        SparklineType::Bar => {
            let n = data.len();
            let plot_w = config.width - 2.0 * config.padding;
            let bar_w = (plot_w / n as f64) * 0.8;
            let gap = (plot_w / n as f64) * 0.2;
            let zero_y = scale.y(0.0_f64.max(data_min(data)));
            for (i, v) in data.iter().enumerate() {
                let x = config.padding + i as f64 * (bar_w + gap);
                let vy = scale.y(*v);
                let (rect_y, rect_h) = if *v >= 0.0 {
                    (vy, (zero_y - vy).abs())
                } else {
                    (zero_y, (vy - zero_y).abs())
                };
                let _ = write!(
                    svg,
                    r#"<rect x="{x:.1}" y="{rect_y:.1}" width="{bar_w:.1}" height="{rect_h:.1}" fill="{}"/>"#,
                    config.color,
                );
            }
        }
        SparklineType::WinLoss => {
            let n = data.len();
            let plot_w = config.width - 2.0 * config.padding;
            let bar_w = (plot_w / n as f64) * 0.8;
            let gap = (plot_w / n as f64) * 0.2;
            let mid = config.height / 2.0;
            let unit_h = (config.height - 2.0 * config.padding) / 2.0;
            for (i, v) in data.iter().enumerate() {
                let x = config.padding + i as f64 * (bar_w + gap);
                let (rect_y, color) = if *v > 0.0 {
                    (mid - unit_h, "#2ecc71")
                } else {
                    (mid, "#e74c3c")
                };
                let _ = write!(
                    svg,
                    r#"<rect x="{x:.1}" y="{rect_y:.1}" width="{bar_w:.1}" height="{unit_h:.1}" fill="{color}"/>"#,
                );
            }
        }
    }

    // Highlights.
    for hl in &config.highlights {
        if hl.index < data.len() {
            let cx = scale.x(hl.index);
            let cy = scale.y(data[hl.index]);
            let _ = write!(
                svg,
                r#"<circle cx="{cx:.1}" cy="{cy:.1}" r="{}" fill="{}"/>"#,
                hl.radius, hl.color,
            );
        }
    }

    svg.push_str("</svg>");
    svg
}

/// Convenience: create highlights for first, last, min, max points.
pub fn auto_highlights(data: &[f64]) -> Vec<Highlight> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut highlights = Vec::new();
    let min_idx = data
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let max_idx = data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    highlights.push(Highlight {
        index: 0,
        color: "#95a5a6".into(),
        radius: 2.0,
    });
    highlights.push(Highlight {
        index: data.len() - 1,
        color: "#2c3e50".into(),
        radius: 2.0,
    });
    highlights.push(Highlight {
        index: min_idx,
        color: "#e74c3c".into(),
        radius: 2.5,
    });
    highlights.push(Highlight {
        index: max_idx,
        color: "#2ecc71".into(),
        radius: 2.5,
    });
    highlights
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<f64> {
        vec![5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 9.0, 1.0, 6.0]
    }

    #[test]
    fn line_sparkline_svg() {
        let svg = render_svg(&sample(), &SparklineConfig::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn bar_sparkline_svg() {
        let cfg = SparklineConfig {
            spark_type: SparklineType::Bar,
            ..Default::default()
        };
        let svg = render_svg(&sample(), &cfg);
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn win_loss_sparkline() {
        let data = vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
        let cfg = SparklineConfig {
            spark_type: SparklineType::WinLoss,
            ..Default::default()
        };
        let svg = render_svg(&data, &cfg);
        assert!(svg.contains("#2ecc71")); // win
        assert!(svg.contains("#e74c3c")); // loss
    }

    #[test]
    fn reference_lines() {
        let cfg = SparklineConfig {
            reference_lines: vec![ReferenceLine::Min, ReferenceLine::Max, ReferenceLine::Average],
            ..Default::default()
        };
        let svg = render_svg(&sample(), &cfg);
        // Should have dashed reference lines.
        assert!(svg.contains("stroke-dasharray"));
    }

    #[test]
    fn reference_value_computation() {
        let data = sample();
        assert!((reference_value(&data, ReferenceLine::Min) - 1.0).abs() < 1e-9);
        assert!((reference_value(&data, ReferenceLine::Max) - 9.0).abs() < 1e-9);
        assert!((reference_value(&data, ReferenceLine::Average) - 5.0).abs() < 1e-9);
        assert!((reference_value(&data, ReferenceLine::Median) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn normal_band() {
        let cfg = SparklineConfig {
            normal_band: Some(NormalBand {
                lo: 3.0,
                hi: 7.0,
                color: "#ecf0f1".into(),
                opacity: 0.3,
            }),
            ..Default::default()
        };
        let svg = render_svg(&sample(), &cfg);
        assert!(svg.contains("#ecf0f1"));
    }

    #[test]
    fn line_path_generation() {
        let data = vec![0.0, 5.0, 10.0];
        let cfg = SparklineConfig::default();
        let path = line_path(&data, &cfg);
        assert!(path.starts_with("M"));
        assert!(path.contains("L"));
    }

    #[test]
    fn responsive_scaling() {
        let data = sample();
        let small = SparklineConfig {
            width: 50.0,
            height: 15.0,
            ..Default::default()
        };
        let svg = render_svg(&data, &small);
        assert!(svg.contains("width=\"50\""));
        assert!(svg.contains("height=\"15\""));
    }

    #[test]
    fn highlights_in_svg() {
        let data = sample();
        let cfg = SparklineConfig {
            highlights: vec![
                Highlight { index: 0, color: "#ff0000".into(), radius: 3.0 },
                Highlight { index: 8, color: "#00ff00".into(), radius: 3.0 },
            ],
            ..Default::default()
        };
        let svg = render_svg(&data, &cfg);
        assert!(svg.contains("<circle"));
        assert!(svg.contains("#ff0000"));
    }

    #[test]
    fn auto_highlights_count() {
        let h = auto_highlights(&sample());
        assert_eq!(h.len(), 4); // first, last, min, max
    }

    #[test]
    fn empty_data_svg() {
        let svg = render_svg(&[], &SparklineConfig::default());
        assert!(svg.contains("svg"));
        assert!(!svg.contains("<path"));
    }

    #[test]
    fn single_point() {
        let svg = render_svg(&[42.0], &SparklineConfig::default());
        assert!(svg.contains("<path"));
    }

    #[test]
    fn median_reference_even_count() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let med = reference_value(&data, ReferenceLine::Median);
        assert!((med - 2.5).abs() < 1e-9);
    }
}
