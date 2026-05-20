//! Axis rendering: orientation, tick generation, tick formatting,
//! axis layout, SVG path generation.

use std::fmt;

// ── Orientation ─────────────────────────────────────────────────

/// Axis orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Top,
    Bottom,
    Left,
    Right,
}

// ── AxisConfig ──────────────────────────────────────────────────

/// Configuration for axis rendering.
#[derive(Debug, Clone)]
pub struct AxisConfig {
    pub orientation: Orientation,
    /// Length of tick marks in pixels.
    pub tick_size: f64,
    /// Distance from tick to label in pixels.
    pub label_offset: f64,
    /// Label rotation in degrees (0 = horizontal).
    pub label_rotation: f64,
    /// Font size for labels.
    pub font_size: f64,
    /// Whether to draw gridlines.
    pub show_grid: bool,
    /// Length of gridlines (typically the chart dimension perpendicular to axis).
    pub grid_length: f64,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            orientation: Orientation::Bottom,
            tick_size: 6.0,
            label_offset: 9.0,
            label_rotation: 0.0,
            font_size: 10.0,
            show_grid: false,
            grid_length: 0.0,
        }
    }
}

impl AxisConfig {
    pub fn bottom() -> Self {
        Self { orientation: Orientation::Bottom, ..Self::default() }
    }

    pub fn top() -> Self {
        Self { orientation: Orientation::Top, ..Self::default() }
    }

    pub fn left() -> Self {
        Self { orientation: Orientation::Left, ..Self::default() }
    }

    pub fn right() -> Self {
        Self { orientation: Orientation::Right, ..Self::default() }
    }

    pub fn with_tick_size(mut self, size: f64) -> Self {
        self.tick_size = size;
        self
    }

    pub fn with_label_offset(mut self, offset: f64) -> Self {
        self.label_offset = offset;
        self
    }

    pub fn with_label_rotation(mut self, degrees: f64) -> Self {
        self.label_rotation = degrees;
        self
    }

    pub fn with_grid(mut self, length: f64) -> Self {
        self.show_grid = true;
        self.grid_length = length;
        self
    }
}

// ── Tick Formatting ─────────────────────────────────────────────

/// How to format tick labels.
#[derive(Debug, Clone)]
pub enum TickFormat {
    /// Fixed decimal precision: `format_number(v, precision)`.
    Number { precision: usize },
    /// SI prefix: k, M, G, T, etc.
    SiPrefix,
    /// Percentage (multiply by 100, add %).
    Percentage { precision: usize },
    /// Date format string (strftime-like, simplified).
    Date { pattern: String },
    /// Custom format function result (pre-computed labels).
    Custom { labels: Vec<String> },
}

impl TickFormat {
    /// Format a single numeric value.
    pub fn format_value(&self, value: f64, index: usize) -> String {
        match self {
            TickFormat::Number { precision } => format!("{:.prec$}", value, prec = precision),
            TickFormat::SiPrefix => format_si(value),
            TickFormat::Percentage { precision } => {
                format!("{:.prec$}%", value * 100.0, prec = precision)
            }
            TickFormat::Date { pattern } => {
                // Simplified: just return the pattern with value as Unix timestamp
                if let Some(dt) = chrono::DateTime::from_timestamp(value as i64, 0) {
                    dt.format(pattern).to_string()
                } else {
                    value.to_string()
                }
            }
            TickFormat::Custom { labels } => {
                labels.get(index).cloned().unwrap_or_else(|| value.to_string())
            }
        }
    }
}

/// Format with SI prefix (k, M, G, T, etc.).
fn format_si(value: f64) -> String {
    let abs = value.abs();
    if abs < 1e-15 {
        return "0.0".to_string();
    }
    if abs >= 1e12 {
        format!("{:.1}T", value / 1e12)
    } else if abs >= 1e9 {
        format!("{:.1}G", value / 1e9)
    } else if abs >= 1e6 {
        format!("{:.1}M", value / 1e6)
    } else if abs >= 1e3 {
        format!("{:.1}k", value / 1e3)
    } else if abs >= 1.0 {
        format!("{:.1}", value)
    } else if abs >= 1e-3 {
        format!("{:.1}m", value * 1e3)
    } else if abs >= 1e-6 {
        format!("{:.1}µ", value * 1e6)
    } else if abs >= 1e-9 {
        format!("{:.1}n", value * 1e9)
    } else {
        format!("{:.2e}", value)
    }
}

// ── Tick ────────────────────────────────────────────────────────

/// A positioned tick on the axis.
#[derive(Debug, Clone)]
pub struct Tick {
    /// Value in domain.
    pub value: f64,
    /// Position along the axis in pixels.
    pub position: f64,
    /// Formatted label.
    pub label: String,
}

// ── Axis Layout ─────────────────────────────────────────────────

/// Computed axis layout.
#[derive(Debug, Clone)]
pub struct AxisLayout {
    /// Ticks with positions and labels.
    pub ticks: Vec<Tick>,
    /// Label positions (x, y) for each tick.
    pub label_positions: Vec<(f64, f64)>,
    /// Gridline coordinates: (x1, y1, x2, y2) for each tick.
    pub gridlines: Vec<(f64, f64, f64, f64)>,
    /// Axis line path data.
    pub axis_path: String,
    /// Full SVG fragment.
    pub svg: String,
}

/// Compute axis layout from tick values and positions.
pub fn compute_axis_layout(
    tick_values: &[f64],
    tick_positions: &[f64],
    config: &AxisConfig,
    format: &TickFormat,
    axis_length: f64,
) -> AxisLayout {
    assert_eq!(tick_values.len(), tick_positions.len());

    let ticks: Vec<Tick> = tick_values
        .iter()
        .zip(tick_positions.iter())
        .enumerate()
        .map(|(i, (&val, &pos))| Tick {
            value: val,
            position: pos,
            label: format.format_value(val, i),
        })
        .collect();

    let is_horizontal = matches!(config.orientation, Orientation::Top | Orientation::Bottom);
    let sign = match config.orientation {
        Orientation::Bottom | Orientation::Right => 1.0,
        Orientation::Top | Orientation::Left => -1.0,
    };

    let label_positions: Vec<(f64, f64)> = ticks
        .iter()
        .map(|t| {
            if is_horizontal {
                (t.position, sign * (config.tick_size + config.label_offset))
            } else {
                (sign * (config.tick_size + config.label_offset), t.position)
            }
        })
        .collect();

    let gridlines: Vec<(f64, f64, f64, f64)> = if config.show_grid {
        ticks
            .iter()
            .map(|t| {
                if is_horizontal {
                    (t.position, 0.0, t.position, -sign * config.grid_length)
                } else {
                    (0.0, t.position, sign * config.grid_length, t.position)
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Build axis line path
    let axis_path = if is_horizontal {
        format!("M0,0L{},0", axis_length)
    } else {
        format!("M0,0L0,{}", axis_length)
    };

    // Build SVG fragment
    let mut svg = String::new();
    svg.push_str(&format!("<g class=\"axis axis-{}\">\n", match config.orientation {
        Orientation::Top => "top",
        Orientation::Bottom => "bottom",
        Orientation::Left => "left",
        Orientation::Right => "right",
    }));
    svg.push_str(&format!("  <path d=\"{}\" fill=\"none\" stroke=\"currentColor\"/>\n", axis_path));

    for (i, tick) in ticks.iter().enumerate() {
        let (tx1, ty1, tx2, ty2) = if is_horizontal {
            (tick.position, 0.0, tick.position, sign * config.tick_size)
        } else {
            (0.0, tick.position, sign * config.tick_size, tick.position)
        };
        svg.push_str(&format!(
            "  <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"currentColor\"/>\n",
            tx1, ty1, tx2, ty2
        ));

        let (lx, ly) = label_positions[i];
        let anchor = match config.orientation {
            Orientation::Left => "end",
            Orientation::Right => "start",
            _ => "middle",
        };
        let dy = match config.orientation {
            Orientation::Top => "-0.3em",
            Orientation::Bottom => "0.7em",
            _ => "0.32em",
        };

        if config.label_rotation.abs() > 0.01 {
            svg.push_str(&format!(
                "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"{}\" dy=\"{}\" \
                 font-size=\"{}\" transform=\"rotate({:.1},{:.1},{:.1})\">{}</text>\n",
                lx, ly, anchor, dy, config.font_size, config.label_rotation, lx, ly, tick.label
            ));
        } else {
            svg.push_str(&format!(
                "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"{}\" dy=\"{}\" \
                 font-size=\"{}\">{}</text>\n",
                lx, ly, anchor, dy, config.font_size, tick.label
            ));
        }
    }

    for (x1, y1, x2, y2) in &gridlines {
        svg.push_str(&format!(
            "  <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
             stroke=\"currentColor\" opacity=\"0.2\"/>\n",
            x1, y1, x2, y2
        ));
    }

    svg.push_str("</g>");

    AxisLayout {
        ticks,
        label_positions,
        gridlines,
        axis_path,
        svg,
    }
}

/// Convenience: generate tick values and positions for a linear range.
pub fn linear_axis(
    domain: (f64, f64),
    range: (f64, f64),
    tick_count: usize,
    config: &AxisConfig,
    format: &TickFormat,
) -> AxisLayout {
    let ticks = super::scale::nice_ticks(domain.0, domain.1, tick_count);
    let d = domain.1 - domain.0;
    let r = range.1 - range.0;
    let positions: Vec<f64> = ticks
        .iter()
        .map(|v| {
            if d.abs() < f64::EPSILON {
                range.0
            } else {
                range.0 + (v - domain.0) / d * r
            }
        })
        .collect();
    let axis_length = r.abs();
    compute_axis_layout(&ticks, &positions, config, format, axis_length)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        let f = TickFormat::Number { precision: 2 };
        assert_eq!(f.format_value(3.14159, 0), "3.14");
    }

    #[test]
    fn test_format_si_prefix() {
        let f = TickFormat::SiPrefix;
        assert_eq!(f.format_value(1500.0, 0), "1.5k");
        assert_eq!(f.format_value(2_500_000.0, 0), "2.5M");
        assert_eq!(f.format_value(0.005, 0), "5.0m");
    }

    #[test]
    fn test_format_percentage() {
        let f = TickFormat::Percentage { precision: 1 };
        assert_eq!(f.format_value(0.75, 0), "75.0%");
    }

    #[test]
    fn test_format_custom() {
        let f = TickFormat::Custom {
            labels: vec!["Jan".into(), "Feb".into(), "Mar".into()],
        };
        assert_eq!(f.format_value(0.0, 1), "Feb");
    }

    #[test]
    fn test_axis_config_builders() {
        let c = AxisConfig::bottom().with_tick_size(10.0).with_grid(400.0);
        assert_eq!(c.orientation, Orientation::Bottom);
        assert_eq!(c.tick_size, 10.0);
        assert!(c.show_grid);
        assert_eq!(c.grid_length, 400.0);
    }

    #[test]
    fn test_compute_axis_layout_basic() {
        let config = AxisConfig::bottom();
        let format = TickFormat::Number { precision: 0 };
        let layout = compute_axis_layout(
            &[0.0, 50.0, 100.0],
            &[0.0, 250.0, 500.0],
            &config,
            &format,
            500.0,
        );
        assert_eq!(layout.ticks.len(), 3);
        assert_eq!(layout.ticks[0].label, "0");
        assert_eq!(layout.ticks[1].label, "50");
        assert!(layout.svg.contains("<path"));
        assert!(layout.svg.contains("<text"));
    }

    #[test]
    fn test_axis_path_horizontal() {
        let config = AxisConfig::bottom();
        let format = TickFormat::Number { precision: 0 };
        let layout = compute_axis_layout(&[0.0], &[0.0], &config, &format, 500.0);
        assert_eq!(layout.axis_path, "M0,0L500,0");
    }

    #[test]
    fn test_axis_path_vertical() {
        let config = AxisConfig::left();
        let format = TickFormat::Number { precision: 0 };
        let layout = compute_axis_layout(&[0.0], &[0.0], &config, &format, 300.0);
        assert_eq!(layout.axis_path, "M0,0L0,300");
    }

    #[test]
    fn test_axis_with_gridlines() {
        let config = AxisConfig::bottom().with_grid(300.0);
        let format = TickFormat::Number { precision: 0 };
        let layout = compute_axis_layout(
            &[0.0, 100.0],
            &[0.0, 500.0],
            &config,
            &format,
            500.0,
        );
        assert_eq!(layout.gridlines.len(), 2);
        assert!(layout.svg.contains("opacity=\"0.2\""));
    }

    #[test]
    fn test_axis_with_rotation() {
        let config = AxisConfig::bottom().with_label_rotation(45.0);
        let format = TickFormat::Number { precision: 0 };
        let layout = compute_axis_layout(
            &[0.0, 100.0],
            &[0.0, 500.0],
            &config,
            &format,
            500.0,
        );
        assert!(layout.svg.contains("rotate(45.0"));
    }

    #[test]
    fn test_linear_axis() {
        let config = AxisConfig::bottom();
        let format = TickFormat::Number { precision: 0 };
        let layout = linear_axis((0.0, 100.0), (0.0, 500.0), 5, &config, &format);
        assert!(!layout.ticks.is_empty());
        // First tick position should be near 0
        assert!(layout.ticks[0].position >= 0.0);
    }

    #[test]
    fn test_label_positions_bottom() {
        let config = AxisConfig::bottom();
        let format = TickFormat::Number { precision: 0 };
        let layout = compute_axis_layout(&[50.0], &[250.0], &config, &format, 500.0);
        // Label should be below axis (positive y)
        assert!(layout.label_positions[0].1 > 0.0);
    }

    #[test]
    fn test_label_positions_left() {
        let config = AxisConfig::left();
        let format = TickFormat::Number { precision: 0 };
        let layout = compute_axis_layout(&[50.0], &[150.0], &config, &format, 300.0);
        // Label should be to the left (negative x)
        assert!(layout.label_positions[0].0 < 0.0);
    }

    #[test]
    fn test_si_format_edge_cases() {
        assert_eq!(format_si(0.0), "0.0");
        assert_eq!(format_si(1e12), "1.0T");
        assert_eq!(format_si(1e9), "1.0G");
    }
}
