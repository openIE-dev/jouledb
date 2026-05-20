//! Funnel chart: horizontal / vertical trapezoid funnels, inverted pyramid,
//! conversion-rate annotations.  Pure Rust SVG output.

// ── Stage ────────────────────────────────────────────────────────

/// A single stage of the funnel.
#[derive(Debug, Clone)]
pub struct FunnelStage {
    pub label: String,
    pub value: f64,
    pub color: String,
}

// ── Orientation ──────────────────────────────────────────────────

/// Whether the funnel is drawn top-to-bottom or left-to-right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Vertical,
    Horizontal,
}

// ── Config ───────────────────────────────────────────────────────

/// Configuration for a funnel chart.
#[derive(Debug, Clone)]
pub struct FunnelConfig {
    pub width: f64,
    pub height: f64,
    pub orientation: Orientation,
    pub padding: f64,
    pub font_size: f64,
    pub gap: f64,
    /// If true, renders widest at bottom (pyramid).
    pub inverted: bool,
}

impl Default for FunnelConfig {
    fn default() -> Self {
        Self {
            width: 400.0,
            height: 300.0,
            orientation: Orientation::Vertical,
            padding: 30.0,
            font_size: 12.0,
            gap: 2.0,
            inverted: false,
        }
    }
}

// ── Geometry helpers ─────────────────────────────────────────────

/// Compute the width of each stage, proportional to its value relative to
/// the maximum value in the funnel.
pub fn stage_widths(stages: &[FunnelStage], max_width: f64) -> Vec<f64> {
    if stages.is_empty() {
        return Vec::new();
    }
    let max_val = stages
        .iter()
        .map(|s| s.value)
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);
    stages
        .iter()
        .map(|s| (s.value / max_val) * max_width)
        .collect()
}

/// Compute conversion rates between consecutive stages.
/// Returns `stages.len() - 1` rates as percentages.
pub fn conversion_rates(stages: &[FunnelStage]) -> Vec<f64> {
    stages
        .windows(2)
        .map(|w| {
            if w[0].value.abs() < f64::EPSILON {
                0.0
            } else {
                w[1].value / w[0].value * 100.0
            }
        })
        .collect()
}

/// Drop-off values between consecutive stages.
pub fn drop_offs(stages: &[FunnelStage]) -> Vec<f64> {
    stages.windows(2).map(|w| w[0].value - w[1].value).collect()
}

/// A trapezoid defined by four corner points.
#[derive(Debug, Clone)]
pub struct Trapezoid {
    pub top_left: (f64, f64),
    pub top_right: (f64, f64),
    pub bottom_right: (f64, f64),
    pub bottom_left: (f64, f64),
}

impl Trapezoid {
    /// SVG path `d` attribute for this trapezoid.
    pub fn svg_path(&self) -> String {
        let (tlx, tly) = self.top_left;
        let (trx, try_) = self.top_right;
        let (brx, bry) = self.bottom_right;
        let (blx, bly) = self.bottom_left;
        format!("M {tlx} {tly} L {trx} {try_} L {brx} {bry} L {blx} {bly} Z")
    }
}

/// Compute trapezoid geometries for each stage (vertical orientation).
pub fn trapezoids_vertical(stages: &[FunnelStage], cfg: &FunnelConfig) -> Vec<Trapezoid> {
    if stages.is_empty() {
        return Vec::new();
    }
    let usable_w = cfg.width - 2.0 * cfg.padding;
    let n = stages.len();
    let total_gap = cfg.gap * (n as f64 - 1.0).max(0.0);
    let usable_h = cfg.height - 2.0 * cfg.padding - total_gap;
    let stage_h = usable_h / n as f64;
    let widths = stage_widths(stages, usable_w);
    let cx = cfg.width / 2.0;

    let ordered_widths: Vec<f64> = if cfg.inverted {
        widths.into_iter().rev().collect()
    } else {
        widths
    };

    let mut traps = Vec::with_capacity(n);
    for i in 0..n {
        let y_top = cfg.padding + i as f64 * (stage_h + cfg.gap);
        let y_bot = y_top + stage_h;
        let w_top = ordered_widths[i];
        let w_bot = if i + 1 < n {
            ordered_widths[i + 1]
        } else {
            ordered_widths[i] * 0.6 // taper the last stage
        };
        traps.push(Trapezoid {
            top_left: (cx - w_top / 2.0, y_top),
            top_right: (cx + w_top / 2.0, y_top),
            bottom_right: (cx + w_bot / 2.0, y_bot),
            bottom_left: (cx - w_bot / 2.0, y_bot),
        });
    }
    traps
}

/// Compute trapezoid geometries for each stage (horizontal orientation).
pub fn trapezoids_horizontal(stages: &[FunnelStage], cfg: &FunnelConfig) -> Vec<Trapezoid> {
    if stages.is_empty() {
        return Vec::new();
    }
    let usable_h = cfg.height - 2.0 * cfg.padding;
    let n = stages.len();
    let total_gap = cfg.gap * (n as f64 - 1.0).max(0.0);
    let usable_w = cfg.width - 2.0 * cfg.padding - total_gap;
    let stage_w = usable_w / n as f64;
    let widths = stage_widths(stages, usable_h);
    let cy = cfg.height / 2.0;

    let ordered_widths: Vec<f64> = if cfg.inverted {
        widths.into_iter().rev().collect()
    } else {
        widths
    };

    let mut traps = Vec::with_capacity(n);
    for i in 0..n {
        let x_left = cfg.padding + i as f64 * (stage_w + cfg.gap);
        let x_right = x_left + stage_w;
        let h_left = ordered_widths[i];
        let h_right = if i + 1 < n {
            ordered_widths[i + 1]
        } else {
            ordered_widths[i] * 0.6
        };
        traps.push(Trapezoid {
            top_left: (x_left, cy - h_left / 2.0),
            top_right: (x_right, cy - h_right / 2.0),
            bottom_right: (x_right, cy + h_right / 2.0),
            bottom_left: (x_left, cy + h_left / 2.0),
        });
    }
    traps
}

// ── Rendering ────────────────────────────────────────────────────

/// Render a funnel chart as SVG.
pub fn render_funnel(stages: &[FunnelStage], cfg: &FunnelConfig) -> String {
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    if stages.is_empty() {
        svg.push_str("</svg>");
        return svg;
    }

    let traps = match cfg.orientation {
        Orientation::Vertical => trapezoids_vertical(stages, cfg),
        Orientation::Horizontal => trapezoids_horizontal(stages, cfg),
    };

    let rates = conversion_rates(stages);
    let display_stages: Vec<&FunnelStage> = if cfg.inverted {
        stages.iter().rev().collect()
    } else {
        stages.iter().collect()
    };

    for (i, (trap, stage)) in traps.iter().zip(display_stages.iter()).enumerate() {
        let d = trap.svg_path();
        svg.push_str(&format!(
            "<path d=\"{d}\" fill=\"{}\" stroke=\"#fff\" />",
            stage.color
        ));

        // Label + percentage
        let lx = (trap.top_left.0 + trap.top_right.0 + trap.bottom_left.0 + trap.bottom_right.0) / 4.0;
        let ly = (trap.top_left.1 + trap.top_right.1 + trap.bottom_left.1 + trap.bottom_right.1) / 4.0;
        let fs = cfg.font_size;
        let pct = if !stages.is_empty() {
            let max_val = stages.iter().map(|s| s.value).fold(0.0_f64, f64::max).max(f64::EPSILON);
            stage.value / max_val * 100.0
        } else {
            0.0
        };
        svg.push_str(&format!(
            "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" text-anchor=\"middle\" dominant-baseline=\"middle\" fill=\"#fff\">{} ({pct:.0}%)</text>",
            stage.label
        ));

        // Conversion rate annotation
        if i < rates.len() {
            let conv_x = lx + (cfg.width - 2.0 * cfg.padding) * 0.4;
            let conv_y = ly + (trap.bottom_left.1 - trap.top_left.1) * 0.5 + cfg.gap;
            let conv = rates[i];
            svg.push_str(&format!(
                "<text x=\"{conv_x}\" y=\"{conv_y}\" font-size=\"{}\" fill=\"#666\">{conv:.1}%</text>",
                fs - 2.0
            ));
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
            FunnelStage { label: "Visitors".into(), value: 1000.0, color: "#3498db".into() },
            FunnelStage { label: "Leads".into(), value: 600.0, color: "#2ecc71".into() },
            FunnelStage { label: "Prospects".into(), value: 300.0, color: "#f1c40f".into() },
            FunnelStage { label: "Customers".into(), value: 100.0, color: "#e74c3c".into() },
        ]
    }

    #[test]
    fn stage_widths_proportional() {
        let stages = sample_stages();
        let widths = stage_widths(&stages, 200.0);
        assert_eq!(widths.len(), 4);
        assert!((widths[0] - 200.0).abs() < 1e-9); // max value = full width
        assert!((widths[1] - 120.0).abs() < 1e-9); // 600/1000 * 200
        assert!((widths[2] - 60.0).abs() < 1e-9);  // 300/1000 * 200
        assert!((widths[3] - 20.0).abs() < 1e-9);  // 100/1000 * 200
    }

    #[test]
    fn stage_widths_empty() {
        assert!(stage_widths(&[], 200.0).is_empty());
    }

    #[test]
    fn conversion_rates_correct() {
        let stages = sample_stages();
        let rates = conversion_rates(&stages);
        assert_eq!(rates.len(), 3);
        assert!((rates[0] - 60.0).abs() < 1e-9);  // 600/1000
        assert!((rates[1] - 50.0).abs() < 1e-9);  // 300/600
        assert!((rates[2] - 100.0 / 3.0).abs() < 0.1); // 100/300
    }

    #[test]
    fn conversion_rate_zero_denominator() {
        let stages = vec![
            FunnelStage { label: "A".into(), value: 0.0, color: "#f00".into() },
            FunnelStage { label: "B".into(), value: 10.0, color: "#0f0".into() },
        ];
        let rates = conversion_rates(&stages);
        assert!((rates[0]).abs() < 1e-9);
    }

    #[test]
    fn drop_offs_correct() {
        let stages = sample_stages();
        let drops = drop_offs(&stages);
        assert_eq!(drops.len(), 3);
        assert!((drops[0] - 400.0).abs() < 1e-9);
        assert!((drops[1] - 300.0).abs() < 1e-9);
        assert!((drops[2] - 200.0).abs() < 1e-9);
    }

    #[test]
    fn trapezoid_svg_path_valid() {
        let t = Trapezoid {
            top_left: (10.0, 0.0),
            top_right: (90.0, 0.0),
            bottom_right: (80.0, 50.0),
            bottom_left: (20.0, 50.0),
        };
        let d = t.svg_path();
        assert!(d.starts_with('M'));
        assert!(d.contains('L'));
        assert!(d.ends_with('Z'));
    }

    #[test]
    fn trapezoids_vertical_count() {
        let stages = sample_stages();
        let cfg = FunnelConfig::default();
        let traps = trapezoids_vertical(&stages, &cfg);
        assert_eq!(traps.len(), 4);
    }

    #[test]
    fn trapezoids_vertical_widths_decrease() {
        let stages = sample_stages();
        let cfg = FunnelConfig::default();
        let traps = trapezoids_vertical(&stages, &cfg);
        for i in 0..traps.len() - 1 {
            let w_cur = traps[i].top_right.0 - traps[i].top_left.0;
            let w_next = traps[i + 1].top_right.0 - traps[i + 1].top_left.0;
            assert!(w_cur >= w_next, "Stage {i} should be wider than stage {}", i + 1);
        }
    }

    #[test]
    fn trapezoids_horizontal_count() {
        let stages = sample_stages();
        let cfg = FunnelConfig {
            orientation: Orientation::Horizontal,
            ..FunnelConfig::default()
        };
        let traps = trapezoids_horizontal(&stages, &cfg);
        assert_eq!(traps.len(), 4);
    }

    #[test]
    fn render_funnel_svg() {
        let stages = sample_stages();
        let cfg = FunnelConfig::default();
        let svg = render_funnel(&stages, &cfg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<path"));
        assert!(svg.contains("Visitors"));
        assert!(svg.contains("Customers"));
    }

    #[test]
    fn render_funnel_empty() {
        let svg = render_funnel(&[], &FunnelConfig::default());
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(!svg.contains("<path"));
    }

    #[test]
    fn inverted_funnel_reverses_widths() {
        let stages = sample_stages();
        let cfg_normal = FunnelConfig::default();
        let cfg_inverted = FunnelConfig {
            inverted: true,
            ..FunnelConfig::default()
        };
        let normal = trapezoids_vertical(&stages, &cfg_normal);
        let inverted = trapezoids_vertical(&stages, &cfg_inverted);
        // First stage of inverted should be narrower than first of normal
        let w_normal = normal[0].top_right.0 - normal[0].top_left.0;
        let w_inv = inverted[0].top_right.0 - inverted[0].top_left.0;
        assert!(w_inv < w_normal);
    }

    #[test]
    fn render_funnel_has_percentage_labels() {
        let stages = sample_stages();
        let cfg = FunnelConfig::default();
        let svg = render_funnel(&stages, &cfg);
        assert!(svg.contains("100%")); // visitors = max
        assert!(svg.contains("60%"));  // leads
    }
}
