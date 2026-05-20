//! Funnel chart: trapezoid segments with conversion rates, percentage labels,
//! horizontal/vertical orientation.  Pure Rust SVG output — no browser dependency.

use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// A single stage in the funnel.
#[derive(Debug, Clone)]
pub struct FunnelStage {
    pub label: String,
    pub value: f64,
    pub color: String,
}

impl FunnelStage {
    pub fn new(label: impl Into<String>, value: f64, color: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.max(0.0),
            color: color.into(),
        }
    }
}

/// Computed metrics for a laid-out stage.
#[derive(Debug, Clone)]
pub struct FunnelSegment {
    /// Index into original stages.
    pub index: usize,
    pub label: String,
    pub value: f64,
    pub color: String,
    /// Width at top of trapezoid (proportional to value).
    pub top_width: f64,
    /// Width at bottom of trapezoid.
    pub bottom_width: f64,
    /// Conversion rate from previous stage (1.0 for the first).
    pub conversion_rate: f64,
    /// Cumulative conversion from the first stage.
    pub cumulative_rate: f64,
    /// Layout: bounding box.
    pub x: f64,
    pub y: f64,
    pub segment_width: f64,
    pub segment_height: f64,
}

// ── Orientation ─────────────────────────────────────────────────

/// Funnel orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Vertical,
    Horizontal,
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for the funnel chart.
#[derive(Debug, Clone)]
pub struct FunnelChartConfig {
    pub width: f64,
    pub height: f64,
    pub orientation: Orientation,
    pub padding: f64,
    pub gap: f64,
    pub font_size: f64,
    /// Whether to show conversion rate annotations.
    pub show_conversion: bool,
    /// Whether to show percentage labels on each segment.
    pub show_percent: bool,
    /// Minimum width ratio for the smallest segment (0..1).
    pub min_width_ratio: f64,
}

impl Default for FunnelChartConfig {
    fn default() -> Self {
        Self {
            width: 500.0,
            height: 400.0,
            orientation: Orientation::Vertical,
            padding: 40.0,
            gap: 3.0,
            font_size: 12.0,
            show_conversion: true,
            show_percent: true,
            min_width_ratio: 0.1,
        }
    }
}

// ── Layout ──────────────────────────────────────────────────────

/// Compute funnel segments from stages.
pub fn compute_segments(
    stages: &[FunnelStage],
    cfg: &FunnelChartConfig,
) -> Vec<FunnelSegment> {
    if stages.is_empty() {
        return Vec::new();
    }

    let n = stages.len();
    let first_value = stages[0].value.max(f64::EPSILON);

    let usable_width = cfg.width - 2.0 * cfg.padding;
    let usable_height = cfg.height - 2.0 * cfg.padding;

    let (seg_main, seg_cross) = match cfg.orientation {
        Orientation::Vertical => {
            let total_gap = cfg.gap * (n.saturating_sub(1)) as f64;
            let seg_h = ((usable_height - total_gap) / n as f64).max(1.0);
            (seg_h, usable_width)
        }
        Orientation::Horizontal => {
            let total_gap = cfg.gap * (n.saturating_sub(1)) as f64;
            let seg_w = ((usable_width - total_gap) / n as f64).max(1.0);
            (seg_w, usable_height)
        }
    };

    let mut segments = Vec::with_capacity(n);
    for (i, stage) in stages.iter().enumerate() {
        let ratio = (stage.value / first_value).clamp(cfg.min_width_ratio, 1.0);
        let next_ratio = if i + 1 < n {
            (stages[i + 1].value / first_value).clamp(cfg.min_width_ratio, 1.0)
        } else {
            ratio * 0.7 // taper the last segment
        };

        let top_w = ratio * seg_cross;
        let bottom_w = next_ratio * seg_cross;

        let conversion_rate = if i == 0 {
            1.0
        } else {
            let prev = stages[i - 1].value;
            if prev > 0.0 {
                stage.value / prev
            } else {
                0.0
            }
        };
        let cumulative_rate = if first_value > 0.0 {
            stage.value / first_value
        } else {
            0.0
        };

        let (x, y, sw, sh) = match cfg.orientation {
            Orientation::Vertical => {
                let x = cfg.padding;
                let y = cfg.padding + i as f64 * (seg_main + cfg.gap);
                (x, y, usable_width, seg_main)
            }
            Orientation::Horizontal => {
                let x = cfg.padding + i as f64 * (seg_main + cfg.gap);
                let y = cfg.padding;
                (x, y, seg_main, usable_height)
            }
        };

        segments.push(FunnelSegment {
            index: i,
            label: stage.label.clone(),
            value: stage.value,
            color: stage.color.clone(),
            top_width: top_w,
            bottom_width: bottom_w,
            conversion_rate,
            cumulative_rate,
            x,
            y,
            segment_width: sw,
            segment_height: sh,
        });
    }

    segments
}

/// Format a rate as a percentage string.
pub fn format_percent(rate: f64) -> String {
    format!("{:.1}%", rate * 100.0)
}

// ── Rendering ───────────────────────────────────────────────────

/// Render a vertical trapezoid segment.
fn render_vertical_segment(seg: &FunnelSegment, cfg: &FunnelChartConfig) -> String {
    let mut svg = String::new();
    let center_x = seg.x + seg.segment_width / 2.0;

    let top_half = seg.top_width / 2.0;
    let bot_half = seg.bottom_width / 2.0;

    let x0 = center_x - top_half;
    let x1 = center_x + top_half;
    let x2 = center_x + bot_half;
    let x3 = center_x - bot_half;
    let y0 = seg.y;
    let y1 = seg.y + seg.segment_height;

    let _ = write!(
        svg,
        "<polygon points=\"{x0},{y0} {x1},{y0} {x2},{y1} {x3},{y1}\" \
         fill=\"{}\" stroke=\"white\" stroke-width=\"1\" />",
        seg.color
    );

    // Label
    let lx = center_x;
    let ly = seg.y + seg.segment_height / 2.0;
    let fs = cfg.font_size;
    let _ = write!(
        svg,
        "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" \
         text-anchor=\"middle\" dominant-baseline=\"middle\" fill=\"white\" \
         font-weight=\"bold\">{}</text>",
        seg.label
    );

    // Value
    let vy = ly + fs + 2.0;
    let vfs = cfg.font_size * 0.9;
    let _ = write!(
        svg,
        "<text x=\"{lx}\" y=\"{vy}\" font-size=\"{vfs}\" \
         text-anchor=\"middle\" dominant-baseline=\"middle\" fill=\"white\">{:.0}</text>",
        seg.value
    );

    // Percentage
    if cfg.show_percent && seg.index > 0 {
        let px = center_x + top_half + 8.0;
        let py = seg.y + seg.segment_height / 2.0;
        let pfs = cfg.font_size * 0.85;
        let _ = write!(
            svg,
            "<text x=\"{px}\" y=\"{py}\" font-size=\"{pfs}\" \
             dominant-baseline=\"middle\" fill=\"gray\">{}</text>",
            format_percent(seg.cumulative_rate)
        );
    }

    // Conversion rate arrow
    if cfg.show_conversion && seg.index > 0 {
        let ax = center_x - top_half - 30.0;
        let ay = seg.y - cfg.gap / 2.0;
        let afs = cfg.font_size * 0.8;
        let _ = write!(
            svg,
            "<text x=\"{ax}\" y=\"{ay}\" font-size=\"{afs}\" \
             text-anchor=\"end\" dominant-baseline=\"middle\" fill=\"gray\">{}</text>",
            format_percent(seg.conversion_rate)
        );
    }

    svg
}

/// Render a horizontal trapezoid segment.
fn render_horizontal_segment(seg: &FunnelSegment, cfg: &FunnelChartConfig) -> String {
    let mut svg = String::new();
    let center_y = seg.y + seg.segment_height / 2.0;

    let top_half = seg.top_width / 2.0;
    let bot_half = seg.bottom_width / 2.0;

    let y0 = center_y - top_half;
    let y1 = center_y + top_half;
    let y2 = center_y + bot_half;
    let y3 = center_y - bot_half;
    let x0 = seg.x;
    let x1 = seg.x + seg.segment_width;

    let _ = write!(
        svg,
        "<polygon points=\"{x0},{y0} {x1},{y3} {x1},{y2} {x0},{y1}\" \
         fill=\"{}\" stroke=\"white\" stroke-width=\"1\" />",
        seg.color
    );

    let lx = seg.x + seg.segment_width / 2.0;
    let ly = center_y;
    let fs = cfg.font_size;
    let _ = write!(
        svg,
        "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" \
         text-anchor=\"middle\" dominant-baseline=\"middle\" fill=\"white\" \
         font-weight=\"bold\">{}</text>",
        seg.label
    );

    svg
}

/// Render the complete funnel chart as SVG.
pub fn render_funnel_chart(
    stages: &[FunnelStage],
    cfg: &FunnelChartConfig,
) -> String {
    let segments = compute_segments(stages, cfg);
    let mut svg = String::with_capacity(4096);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">",
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    for seg in &segments {
        match cfg.orientation {
            Orientation::Vertical => svg.push_str(&render_vertical_segment(seg, cfg)),
            Orientation::Horizontal => svg.push_str(&render_horizontal_segment(seg, cfg)),
        }
    }

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stages() -> Vec<FunnelStage> {
        vec![
            FunnelStage::new("Visitors", 1000.0, "steelblue"),
            FunnelStage::new("Signups", 600.0, "dodgerblue"),
            FunnelStage::new("Active", 300.0, "royalblue"),
            FunnelStage::new("Paid", 100.0, "navy"),
        ]
    }

    #[test]
    fn stage_new() {
        let s = FunnelStage::new("Test", 42.0, "red");
        assert_eq!(s.label, "Test");
        assert!((s.value - 42.0).abs() < 1e-9);
        assert_eq!(s.color, "red");
    }

    #[test]
    fn stage_clamps_negative() {
        let s = FunnelStage::new("X", -5.0, "blue");
        assert_eq!(s.value, 0.0);
    }

    #[test]
    fn compute_segments_count() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let segs = compute_segments(&stages, &cfg);
        assert_eq!(segs.len(), 4);
    }

    #[test]
    fn compute_segments_empty() {
        let cfg = FunnelChartConfig::default();
        let segs = compute_segments(&[], &cfg);
        assert!(segs.is_empty());
    }

    #[test]
    fn conversion_rates() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let segs = compute_segments(&stages, &cfg);
        assert!((segs[0].conversion_rate - 1.0).abs() < 1e-9);
        assert!((segs[1].conversion_rate - 0.6).abs() < 1e-9);
        assert!((segs[2].conversion_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn cumulative_rates() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let segs = compute_segments(&stages, &cfg);
        assert!((segs[0].cumulative_rate - 1.0).abs() < 1e-9);
        assert!((segs[1].cumulative_rate - 0.6).abs() < 1e-9);
        assert!((segs[2].cumulative_rate - 0.3).abs() < 1e-9);
        assert!((segs[3].cumulative_rate - 0.1).abs() < 1e-9);
    }

    #[test]
    fn top_width_decreases() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let segs = compute_segments(&stages, &cfg);
        for i in 1..segs.len() {
            assert!(segs[i].top_width <= segs[i - 1].top_width + 1e-9);
        }
    }

    #[test]
    fn format_percent_works() {
        assert_eq!(format_percent(0.5), "50.0%");
        assert_eq!(format_percent(1.0), "100.0%");
        assert_eq!(format_percent(0.0), "0.0%");
    }

    #[test]
    fn config_default_sane() {
        let cfg = FunnelChartConfig::default();
        assert!(cfg.width > 0.0);
        assert!(cfg.height > 0.0);
        assert!(cfg.min_width_ratio > 0.0);
    }

    #[test]
    fn render_vertical_produces_svg() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let svg = render_funnel_chart(&stages, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn render_contains_labels() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let svg = render_funnel_chart(&stages, &cfg);
        assert!(svg.contains("Visitors"));
        assert!(svg.contains("Signups"));
        assert!(svg.contains("Active"));
        assert!(svg.contains("Paid"));
    }

    #[test]
    fn render_contains_polygons() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let svg = render_funnel_chart(&stages, &cfg);
        assert_eq!(svg.matches("<polygon").count(), 4);
    }

    #[test]
    fn render_horizontal() {
        let stages = sample_stages();
        let mut cfg = FunnelChartConfig::default();
        cfg.orientation = Orientation::Horizontal;
        let svg = render_funnel_chart(&stages, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<polygon"));
    }

    #[test]
    fn render_single_stage() {
        let stages = vec![FunnelStage::new("Only", 100.0, "teal")];
        let cfg = FunnelChartConfig::default();
        let svg = render_funnel_chart(&stages, &cfg);
        assert!(svg.contains("Only"));
    }

    #[test]
    fn segment_indices_sequential() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let segs = compute_segments(&stages, &cfg);
        for (i, seg) in segs.iter().enumerate() {
            assert_eq!(seg.index, i);
        }
    }

    #[test]
    fn segments_have_positive_dimensions() {
        let stages = sample_stages();
        let cfg = FunnelChartConfig::default();
        let segs = compute_segments(&stages, &cfg);
        for seg in &segs {
            assert!(seg.segment_width > 0.0);
            assert!(seg.segment_height > 0.0);
            assert!(seg.top_width > 0.0);
            assert!(seg.bottom_width > 0.0);
        }
    }
}
