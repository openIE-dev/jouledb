//! Waterfall chart: positive/negative/total bars, running total calculation,
//! connector lines between bars, color coding (increase=green, decrease=red,
//! total=blue), label formatting, horizontal mode.  Pure Rust SVG output.

use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// The kind of entry in a waterfall chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// An incremental change (positive or negative).
    Delta,
    /// An absolute total / subtotal bar.
    Total,
}

/// A single data point in the waterfall.
#[derive(Debug, Clone)]
pub struct WaterfallEntry {
    pub label: String,
    pub value: f64,
    pub kind: EntryKind,
}

impl WaterfallEntry {
    pub fn delta(label: impl Into<String>, value: f64) -> Self {
        Self {
            label: label.into(),
            value,
            kind: EntryKind::Delta,
        }
    }

    pub fn total(label: impl Into<String>, value: f64) -> Self {
        Self {
            label: label.into(),
            value,
            kind: EntryKind::Total,
        }
    }
}

/// Orientation of the chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Vertical,
    Horizontal,
}

/// Color scheme for the waterfall bars.
#[derive(Debug, Clone)]
pub struct WaterfallColors {
    pub increase: String,
    pub decrease: String,
    pub total: String,
    pub connector: String,
}

impl Default for WaterfallColors {
    fn default() -> Self {
        Self {
            increase: "#2ecc71".into(),
            decrease: "#e74c3c".into(),
            total: "#3498db".into(),
            connector: "#95a5a6".into(),
        }
    }
}

// ── Computed bar ─────────────────────────────────────────────────

/// A bar ready for rendering with computed positions.
#[derive(Debug, Clone)]
pub struct WaterfallBar {
    pub label: String,
    pub value: f64,
    pub kind: EntryKind,
    /// Bottom of the bar (lower value).
    pub start: f64,
    /// Top of the bar (higher value).
    pub end: f64,
    pub color: String,
    pub running_total: f64,
}

impl WaterfallBar {
    pub fn bar_height(&self) -> f64 {
        (self.end - self.start).abs()
    }

    pub fn is_increase(&self) -> bool {
        self.kind == EntryKind::Delta && self.value >= 0.0
    }

    pub fn is_decrease(&self) -> bool {
        self.kind == EntryKind::Delta && self.value < 0.0
    }
}

// ── Computation ──────────────────────────────────────────────────

/// Compute waterfall bars from entries.
pub fn compute_bars(entries: &[WaterfallEntry], colors: &WaterfallColors) -> Vec<WaterfallBar> {
    let mut bars = Vec::with_capacity(entries.len());
    let mut running = 0.0_f64;

    for entry in entries {
        match entry.kind {
            EntryKind::Delta => {
                let prev = running;
                running += entry.value;
                let (start, end) = if entry.value >= 0.0 {
                    (prev, running)
                } else {
                    (running, prev)
                };
                let color = if entry.value >= 0.0 {
                    colors.increase.clone()
                } else {
                    colors.decrease.clone()
                };
                bars.push(WaterfallBar {
                    label: entry.label.clone(),
                    value: entry.value,
                    kind: EntryKind::Delta,
                    start,
                    end,
                    color,
                    running_total: running,
                });
            }
            EntryKind::Total => {
                running = entry.value;
                bars.push(WaterfallBar {
                    label: entry.label.clone(),
                    value: entry.value,
                    kind: EntryKind::Total,
                    start: 0.0,
                    end: entry.value,
                    color: colors.total.clone(),
                    running_total: running,
                });
            }
        }
    }

    bars
}

/// Compute running totals as a convenience function.
pub fn running_totals(entries: &[WaterfallEntry]) -> Vec<f64> {
    let bars = compute_bars(entries, &WaterfallColors::default());
    bars.iter().map(|b| b.running_total).collect()
}

// ── Label formatting ─────────────────────────────────────────────

/// Format a value for display on bars.
pub fn format_value(value: f64, prefix: &str, suffix: &str, decimals: usize) -> String {
    let sign = if value >= 0.0 { "+" } else { "" };
    format!("{prefix}{sign}{value:.prec$}{suffix}", prec = decimals)
}

/// Format a value as compact (K/M/B).
pub fn format_compact(value: f64) -> String {
    let abs = value.abs();
    if abs >= 1_000_000_000.0 {
        format!("{:.1}B", value / 1_000_000_000.0)
    } else if abs >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{:.1}K", value / 1_000.0)
    } else {
        format!("{value:.0}")
    }
}

// ── SVG rendering ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WaterfallConfig {
    pub width: f64,
    pub height: f64,
    pub orientation: Orientation,
    pub colors: WaterfallColors,
    pub bar_gap: f64,
    pub margin: f64,
    pub show_connectors: bool,
    pub show_labels: bool,
}

impl Default for WaterfallConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 400.0,
            orientation: Orientation::Vertical,
            colors: WaterfallColors::default(),
            bar_gap: 8.0,
            margin: 60.0,
            show_connectors: true,
            show_labels: true,
        }
    }
}

pub fn render_svg(entries: &[WaterfallEntry], config: &WaterfallConfig) -> String {
    let bars = compute_bars(entries, &config.colors);
    if bars.is_empty() {
        return r#"<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"></svg>"#.into();
    }

    // Value range.
    let mut min_val = 0.0_f64;
    let mut max_val = 0.0_f64;
    for b in &bars {
        min_val = min_val.min(b.start).min(b.end);
        max_val = max_val.max(b.start).max(b.end);
    }
    let val_range = (max_val - min_val).max(f64::EPSILON);

    let plot_w = config.width - 2.0 * config.margin;
    let plot_h = config.height - 2.0 * config.margin;
    let n = bars.len();
    let total_gap = config.bar_gap * (n as f64 - 1.0).max(0.0);

    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#,
        config.width, config.height,
    );

    match config.orientation {
        Orientation::Vertical => {
            let bar_w = (plot_w - total_gap) / n.max(1) as f64;

            let to_y = |v: f64| -> f64 {
                config.margin + plot_h * (1.0 - (v - min_val) / val_range)
            };

            for (i, bar) in bars.iter().enumerate() {
                let x = config.margin + i as f64 * (bar_w + config.bar_gap);
                let y_top = to_y(bar.end);
                let y_bot = to_y(bar.start);
                let h = (y_bot - y_top).abs();
                let _ = write!(
                    svg,
                    r#"<rect x="{x:.1}" y="{:.1}" width="{bar_w:.1}" height="{h:.1}" fill="{}"/>"#,
                    y_top.min(y_bot),
                    bar.color,
                );

                // Connector to next bar.
                if config.show_connectors && i + 1 < n && bar.kind == EntryKind::Delta {
                    let next_x = config.margin + (i + 1) as f64 * (bar_w + config.bar_gap);
                    let cy = to_y(bar.running_total);
                    let _ = write!(
                        svg,
                        r#"<line x1="{:.1}" y1="{cy:.1}" x2="{next_x:.1}" y2="{cy:.1}" stroke="{}" stroke-dasharray="4,2"/>"#,
                        x + bar_w,
                        config.colors.connector,
                    );
                }

                // Label.
                if config.show_labels {
                    let label_y = y_top.min(y_bot) - 4.0;
                    let _ = write!(
                        svg,
                        r#"<text x="{:.1}" y="{label_y:.1}" text-anchor="middle" font-size="11">{}</text>"#,
                        x + bar_w / 2.0,
                        bar.label,
                    );
                }
            }
        }
        Orientation::Horizontal => {
            let bar_h = (plot_h - total_gap) / n.max(1) as f64;

            let to_x = |v: f64| -> f64 {
                config.margin + plot_w * (v - min_val) / val_range
            };

            for (i, bar) in bars.iter().enumerate() {
                let y = config.margin + i as f64 * (bar_h + config.bar_gap);
                let x_lo = to_x(bar.start);
                let x_hi = to_x(bar.end);
                let w = (x_hi - x_lo).abs();
                let _ = write!(
                    svg,
                    r#"<rect x="{:.1}" y="{y:.1}" width="{w:.1}" height="{bar_h:.1}" fill="{}"/>"#,
                    x_lo.min(x_hi),
                    bar.color,
                );

                if config.show_labels {
                    let _ = write!(
                        svg,
                        r#"<text x="{:.1}" y="{:.1}" text-anchor="end" dominant-baseline="central" font-size="11">{}</text>"#,
                        config.margin - 4.0,
                        y + bar_h / 2.0,
                        bar.label,
                    );
                }
            }
        }
    }

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> Vec<WaterfallEntry> {
        vec![
            WaterfallEntry::total("Start", 100.0),
            WaterfallEntry::delta("Sales", 50.0),
            WaterfallEntry::delta("Returns", -20.0),
            WaterfallEntry::delta("Costs", -40.0),
            WaterfallEntry::total("End", 90.0),
        ]
    }

    #[test]
    fn running_total_correct() {
        let entries = sample_entries();
        let totals = running_totals(&entries);
        assert_eq!(totals.len(), 5);
        assert!((totals[0] - 100.0).abs() < 1e-9); // Total bar
        assert!((totals[1] - 150.0).abs() < 1e-9); // +50
        assert!((totals[2] - 130.0).abs() < 1e-9); // -20
        assert!((totals[3] - 90.0).abs() < 1e-9);  // -40
        assert!((totals[4] - 90.0).abs() < 1e-9);  // Total
    }

    #[test]
    fn bar_colors() {
        let entries = sample_entries();
        let colors = WaterfallColors::default();
        let bars = compute_bars(&entries, &colors);
        assert_eq!(bars[0].color, colors.total);
        assert_eq!(bars[1].color, colors.increase);
        assert_eq!(bars[2].color, colors.decrease);
        assert_eq!(bars[4].color, colors.total);
    }

    #[test]
    fn delta_bar_direction() {
        let entries = sample_entries();
        let bars = compute_bars(&entries, &WaterfallColors::default());
        assert!(bars[1].is_increase());
        assert!(bars[2].is_decrease());
    }

    #[test]
    fn bar_height_positive() {
        let entries = sample_entries();
        let bars = compute_bars(&entries, &WaterfallColors::default());
        for b in &bars {
            assert!(b.bar_height() >= 0.0);
        }
    }

    #[test]
    fn format_value_positive() {
        let s = format_value(50.0, "$", "", 0);
        assert_eq!(s, "$+50");
    }

    #[test]
    fn format_value_negative() {
        let s = format_value(-20.0, "$", "", 0);
        assert_eq!(s, "$-20");
    }

    #[test]
    fn format_compact_billions() {
        assert_eq!(format_compact(2_500_000_000.0), "2.5B");
    }

    #[test]
    fn format_compact_millions() {
        assert_eq!(format_compact(1_200_000.0), "1.2M");
    }

    #[test]
    fn format_compact_thousands() {
        assert_eq!(format_compact(4_500.0), "4.5K");
    }

    #[test]
    fn render_vertical_svg() {
        let entries = sample_entries();
        let cfg = WaterfallConfig::default();
        let svg = render_svg(&entries, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<line")); // connectors
    }

    #[test]
    fn render_horizontal_svg() {
        let entries = sample_entries();
        let cfg = WaterfallConfig {
            orientation: Orientation::Horizontal,
            ..Default::default()
        };
        let svg = render_svg(&entries, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn empty_entries() {
        let svg = render_svg(&[], &WaterfallConfig::default());
        assert!(svg.contains("svg"));
    }

    #[test]
    fn all_positive_deltas() {
        let entries = vec![
            WaterfallEntry::total("Start", 0.0),
            WaterfallEntry::delta("A", 10.0),
            WaterfallEntry::delta("B", 20.0),
        ];
        let totals = running_totals(&entries);
        assert!((totals[2] - 30.0).abs() < 1e-9);
    }

    #[test]
    fn custom_colors() {
        let colors = WaterfallColors {
            increase: "#00ff00".into(),
            decrease: "#ff0000".into(),
            total: "#0000ff".into(),
            connector: "#aaa".into(),
        };
        let entries = vec![
            WaterfallEntry::delta("Up", 10.0),
            WaterfallEntry::delta("Down", -5.0),
        ];
        let bars = compute_bars(&entries, &colors);
        assert_eq!(bars[0].color, "#00ff00");
        assert_eq!(bars[1].color, "#ff0000");
    }
}
