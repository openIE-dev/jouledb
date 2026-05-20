//! Bullet chart: quantitative scale, featured measure bar, comparative measure
//! marker, qualitative ranges (poor/satisfactory/good), horizontal/vertical,
//! title/subtitle, target line, delta indicator.  Pure Rust SVG output.

use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// A qualitative range band (e.g., "poor", "satisfactory", "good").
#[derive(Debug, Clone)]
pub struct QualitativeRange {
    pub label: String,
    pub max_value: f64,
    pub color: String,
}

impl QualitativeRange {
    pub fn new(label: impl Into<String>, max_value: f64, color: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            max_value,
            color: color.into(),
        }
    }
}

/// A single bullet chart specification.
#[derive(Debug, Clone)]
pub struct BulletSpec {
    pub title: String,
    pub subtitle: String,
    /// The featured (actual) measure value.
    pub actual: f64,
    /// The comparative (target) measure value.
    pub target: f64,
    /// Qualitative ranges, ordered by ascending max_value.
    pub ranges: Vec<QualitativeRange>,
}

impl BulletSpec {
    pub fn new(title: impl Into<String>, actual: f64, target: f64) -> Self {
        Self {
            title: title.into(),
            subtitle: String::new(),
            actual,
            target,
            ranges: Vec::new(),
        }
    }

    pub fn with_subtitle(mut self, sub: impl Into<String>) -> Self {
        self.subtitle = sub.into();
        self
    }

    pub fn with_ranges(mut self, ranges: Vec<QualitativeRange>) -> Self {
        self.ranges = ranges;
        self
    }

    /// Add the standard three-range scheme (poor / satisfactory / good).
    pub fn with_standard_ranges(mut self, poor_max: f64, sat_max: f64, good_max: f64) -> Self {
        self.ranges = vec![
            QualitativeRange::new("Poor", poor_max, "#d4d4d4"),
            QualitativeRange::new("Satisfactory", sat_max, "#b0b0b0"),
            QualitativeRange::new("Good", good_max, "#8c8c8c"),
        ];
        self
    }

    /// Scale maximum — largest range max or max of actual/target.
    pub fn scale_max(&self) -> f64 {
        let range_max = self
            .ranges
            .iter()
            .map(|r| r.max_value)
            .fold(0.0_f64, f64::max);
        range_max.max(self.actual).max(self.target)
    }

    /// Delta between actual and target.
    pub fn delta(&self) -> f64 {
        self.actual - self.target
    }

    /// Whether actual meets or exceeds target.
    pub fn on_target(&self) -> bool {
        self.actual >= self.target
    }
}

/// Orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

/// Delta indicator style.
#[derive(Debug, Clone)]
pub struct DeltaIndicator {
    pub positive_color: String,
    pub negative_color: String,
    pub show: bool,
}

impl Default for DeltaIndicator {
    fn default() -> Self {
        Self {
            positive_color: "#2ecc71".into(),
            negative_color: "#e74c3c".into(),
            show: false,
        }
    }
}

/// Rendering configuration.
#[derive(Debug, Clone)]
pub struct BulletConfig {
    pub orientation: Orientation,
    /// Width of the entire chart.
    pub width: f64,
    /// Height per bullet (or width if vertical).
    pub bar_height: f64,
    /// Space between multiple bullets.
    pub spacing: f64,
    /// Left margin for title.
    pub label_width: f64,
    /// Thickness of the actual measure bar relative to bar_height.
    pub measure_thickness: f64,
    /// Target marker width.
    pub target_width: f64,
    pub delta: DeltaIndicator,
}

impl Default for BulletConfig {
    fn default() -> Self {
        Self {
            orientation: Orientation::Horizontal,
            width: 500.0,
            bar_height: 40.0,
            spacing: 20.0,
            label_width: 120.0,
            measure_thickness: 0.4,
            target_width: 3.0,
            delta: DeltaIndicator::default(),
        }
    }
}

// ── SVG rendering ────────────────────────────────────────────────

/// Render one or more bullet charts to SVG.
pub fn render_svg(specs: &[BulletSpec], config: &BulletConfig) -> String {
    let n = specs.len();
    let total_height = n as f64 * config.bar_height + (n.saturating_sub(1)) as f64 * config.spacing + 20.0;
    let mut svg = String::new();

    match config.orientation {
        Orientation::Horizontal => {
            let _ = write!(
                svg,
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{total_height}">"#,
                config.width,
            );
            for (i, spec) in specs.iter().enumerate() {
                let y_off = 10.0 + i as f64 * (config.bar_height + config.spacing);
                render_horizontal_bullet(&mut svg, spec, config, y_off);
            }
        }
        Orientation::Vertical => {
            let total_width = n as f64 * config.bar_height + (n.saturating_sub(1)) as f64 * config.spacing + 20.0;
            let _ = write!(
                svg,
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{}">"#,
                config.width,
            );
            for (i, spec) in specs.iter().enumerate() {
                let x_off = 10.0 + i as f64 * (config.bar_height + config.spacing);
                render_vertical_bullet(&mut svg, spec, config, x_off);
            }
        }
    }

    svg.push_str("</svg>");
    svg
}

fn render_horizontal_bullet(
    svg: &mut String,
    spec: &BulletSpec,
    config: &BulletConfig,
    y_off: f64,
) {
    let plot_w = config.width - config.label_width - 20.0;
    let scale_max = spec.scale_max().max(f64::EPSILON);

    // Title and subtitle.
    let _ = write!(
        svg,
        r#"<text x="{:.1}" y="{:.1}" text-anchor="end" font-size="12" font-weight="bold">{}</text>"#,
        config.label_width - 8.0,
        y_off + config.bar_height * 0.45,
        spec.title,
    );
    if !spec.subtitle.is_empty() {
        let _ = write!(
            svg,
            r#"<text x="{:.1}" y="{:.1}" text-anchor="end" font-size="9" fill="gray">{}</text>"#,
            config.label_width - 8.0,
            y_off + config.bar_height * 0.75,
            spec.subtitle,
        );
    }

    // Qualitative ranges (draw widest first).
    let mut sorted_ranges: Vec<&QualitativeRange> = spec.ranges.iter().collect();
    sorted_ranges.sort_by(|a, b| b.max_value.partial_cmp(&a.max_value).unwrap());
    for range in &sorted_ranges {
        let w = range.max_value / scale_max * plot_w;
        let _ = write!(
            svg,
            r#"<rect x="{:.1}" y="{y_off:.1}" width="{w:.1}" height="{:.1}" fill="{}"/>"#,
            config.label_width, config.bar_height, range.color,
        );
    }

    // Featured measure bar.
    let bar_h = config.bar_height * config.measure_thickness;
    let bar_y = y_off + (config.bar_height - bar_h) / 2.0;
    let bar_w = spec.actual / scale_max * plot_w;
    let _ = write!(
        svg,
        r#"<rect x="{:.1}" y="{bar_y:.1}" width="{bar_w:.1}" height="{bar_h:.1}" fill="darkgray"/>"#,
        config.label_width,
    );

    // Target marker.
    let target_x = config.label_width + spec.target / scale_max * plot_w;
    let _ = write!(
        svg,
        r#"<line x1="{target_x:.1}" y1="{:.1}" x2="{target_x:.1}" y2="{:.1}" stroke="black" stroke-width="{}"/>"#,
        y_off + config.bar_height * 0.15,
        y_off + config.bar_height * 0.85,
        config.target_width,
    );

    // Delta indicator.
    if config.delta.show {
        let delta = spec.delta();
        let color = if delta >= 0.0 {
            &config.delta.positive_color
        } else {
            &config.delta.negative_color
        };
        let _ = write!(
            svg,
            r#"<text x="{:.1}" y="{:.1}" font-size="10" fill="{color}">{:+.1}</text>"#,
            config.width - 15.0,
            y_off + config.bar_height * 0.6,
            delta,
        );
    }
}

fn render_vertical_bullet(
    svg: &mut String,
    spec: &BulletSpec,
    config: &BulletConfig,
    x_off: f64,
) {
    let plot_h = config.width - config.label_width - 20.0;
    let scale_max = spec.scale_max().max(f64::EPSILON);
    let bot = config.width - 20.0;

    // Qualitative ranges.
    let mut sorted_ranges: Vec<&QualitativeRange> = spec.ranges.iter().collect();
    sorted_ranges.sort_by(|a, b| b.max_value.partial_cmp(&a.max_value).unwrap());
    for range in &sorted_ranges {
        let h = range.max_value / scale_max * plot_h;
        let _ = write!(
            svg,
            r#"<rect x="{x_off:.1}" y="{:.1}" width="{:.1}" height="{h:.1}" fill="{}"/>"#,
            bot - h, config.bar_height, range.color,
        );
    }

    // Featured measure.
    let bar_w = config.bar_height * config.measure_thickness;
    let bar_x = x_off + (config.bar_height - bar_w) / 2.0;
    let bar_h = spec.actual / scale_max * plot_h;
    let _ = write!(
        svg,
        r#"<rect x="{bar_x:.1}" y="{:.1}" width="{bar_w:.1}" height="{bar_h:.1}" fill="darkgray"/>"#,
        bot - bar_h,
    );

    // Target.
    let target_y = bot - spec.target / scale_max * plot_h;
    let _ = write!(
        svg,
        r#"<line x1="{:.1}" y1="{target_y:.1}" x2="{:.1}" y2="{target_y:.1}" stroke="black" stroke-width="{}"/>"#,
        x_off + config.bar_height * 0.15,
        x_off + config.bar_height * 0.85,
        config.target_width,
    );

    // Title below.
    let _ = write!(
        svg,
        r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-size="11">{}</text>"#,
        x_off + config.bar_height / 2.0,
        config.width - 5.0,
        spec.title,
    );
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spec() -> BulletSpec {
        BulletSpec::new("Revenue", 275.0, 250.0)
            .with_subtitle("USD (thousands)")
            .with_standard_ranges(150.0, 225.0, 300.0)
    }

    #[test]
    fn scale_max_from_ranges() {
        let s = sample_spec();
        assert!((s.scale_max() - 300.0).abs() < 1e-9);
    }

    #[test]
    fn scale_max_exceeds_ranges() {
        let s = BulletSpec::new("X", 400.0, 350.0)
            .with_standard_ranges(100.0, 200.0, 300.0);
        assert!((s.scale_max() - 400.0).abs() < 1e-9);
    }

    #[test]
    fn delta_positive() {
        let s = sample_spec();
        assert!((s.delta() - 25.0).abs() < 1e-9);
        assert!(s.on_target());
    }

    #[test]
    fn delta_negative() {
        let s = BulletSpec::new("X", 200.0, 250.0);
        assert!((s.delta() - (-50.0)).abs() < 1e-9);
        assert!(!s.on_target());
    }

    #[test]
    fn standard_ranges_count() {
        let s = sample_spec();
        assert_eq!(s.ranges.len(), 3);
    }

    #[test]
    fn render_horizontal() {
        let specs = vec![sample_spec()];
        let svg = render_svg(&specs, &BulletConfig::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<line"));
        assert!(svg.contains("Revenue"));
    }

    #[test]
    fn render_vertical() {
        let specs = vec![sample_spec()];
        let cfg = BulletConfig {
            orientation: Orientation::Vertical,
            ..Default::default()
        };
        let svg = render_svg(&specs, &cfg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Revenue"));
    }

    #[test]
    fn multiple_bullets() {
        let specs = vec![
            sample_spec(),
            BulletSpec::new("Profit", 22.0, 26.0)
                .with_standard_ranges(15.0, 20.0, 30.0),
        ];
        let svg = render_svg(&specs, &BulletConfig::default());
        assert!(svg.contains("Revenue"));
        assert!(svg.contains("Profit"));
    }

    #[test]
    fn delta_indicator() {
        let specs = vec![sample_spec()];
        let cfg = BulletConfig {
            delta: DeltaIndicator {
                show: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let svg = render_svg(&specs, &cfg);
        assert!(svg.contains("+25.0"));
    }

    #[test]
    fn custom_ranges() {
        let s = BulletSpec::new("Test", 50.0, 40.0).with_ranges(vec![
            QualitativeRange::new("Low", 30.0, "#aaa"),
            QualitativeRange::new("Mid", 60.0, "#888"),
            QualitativeRange::new("High", 100.0, "#666"),
        ]);
        assert_eq!(s.ranges.len(), 3);
        assert!((s.scale_max() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn subtitle_in_svg() {
        let specs = vec![sample_spec()];
        let svg = render_svg(&specs, &BulletConfig::default());
        assert!(svg.contains("USD (thousands)"));
    }

    #[test]
    fn empty_specs() {
        let svg = render_svg(&[], &BulletConfig::default());
        assert!(svg.contains("svg"));
    }

    #[test]
    fn no_ranges_still_renders() {
        let specs = vec![BulletSpec::new("Bare", 50.0, 40.0)];
        let svg = render_svg(&specs, &BulletConfig::default());
        assert!(svg.contains("Bare"));
    }
}
