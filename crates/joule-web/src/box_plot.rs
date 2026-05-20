//! Box plot statistics: quartile computation (Q1/Q2/Q3), IQR, whisker
//! calculation (1.5*IQR), outlier detection, notched box plots, violin plot
//! shape (KDE), grouping, horizontal/vertical orientation.  Pure Rust SVG.

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;

// ── Statistics ───────────────────────────────────────────────────

fn sorted_copy(data: &[f64]) -> Vec<f64> {
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ── BoxStats ─────────────────────────────────────────────────────

/// Computed box plot statistics for a single dataset.
#[derive(Debug, Clone)]
pub struct BoxStats {
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub mean: f64,
    pub iqr: f64,
    pub whisker_lo: f64,
    pub whisker_hi: f64,
    pub outliers: Vec<f64>,
    pub n: usize,
    /// Notch bounds (95 % CI of median).
    pub notch_lo: f64,
    pub notch_hi: f64,
}

impl BoxStats {
    /// Compute box plot statistics from raw data.
    pub fn compute(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self {
                min: 0.0,
                q1: 0.0,
                median: 0.0,
                q3: 0.0,
                max: 0.0,
                mean: 0.0,
                iqr: 0.0,
                whisker_lo: 0.0,
                whisker_hi: 0.0,
                outliers: Vec::new(),
                n: 0,
                notch_lo: 0.0,
                notch_hi: 0.0,
            };
        }

        let sorted = sorted_copy(data);
        let n = sorted.len();
        let q1 = percentile(&sorted, 0.25);
        let med = percentile(&sorted, 0.5);
        let q3 = percentile(&sorted, 0.75);
        let iqr_val = q3 - q1;
        let fence_lo = q1 - 1.5 * iqr_val;
        let fence_hi = q3 + 1.5 * iqr_val;

        // Whiskers are the most extreme points within the fences.
        let whisker_lo = sorted
            .iter()
            .copied()
            .find(|v| *v >= fence_lo)
            .unwrap_or(sorted[0]);
        let whisker_hi = sorted
            .iter()
            .rev()
            .copied()
            .find(|v| *v <= fence_hi)
            .unwrap_or(sorted[n - 1]);

        let outliers: Vec<f64> = sorted
            .iter()
            .copied()
            .filter(|v| *v < fence_lo || *v > fence_hi)
            .collect();

        let mean_val = data.iter().sum::<f64>() / n as f64;

        // Notch: median +/- 1.57 * IQR / sqrt(n)
        let notch_half = 1.57 * iqr_val / (n as f64).sqrt();

        Self {
            min: sorted[0],
            q1,
            median: med,
            q3,
            max: sorted[n - 1],
            mean: mean_val,
            iqr: iqr_val,
            whisker_lo,
            whisker_hi,
            outliers,
            n,
            notch_lo: med - notch_half,
            notch_hi: med + notch_half,
        }
    }

    /// Check whether a value is an outlier.
    pub fn is_outlier(&self, v: f64) -> bool {
        v < self.q1 - 1.5 * self.iqr || v > self.q3 + 1.5 * self.iqr
    }
}

// ── Violin (KDE shape) ──────────────────────────────────────────

/// Kernel density estimate for a violin plot silhouette.
#[derive(Debug, Clone)]
pub struct ViolinShape {
    pub points: Vec<(f64, f64)>, // (value, density)
    pub max_density: f64,
}

impl ViolinShape {
    /// Compute Gaussian KDE for a dataset.
    pub fn compute(data: &[f64], n_points: usize) -> Self {
        if data.is_empty() {
            return Self {
                points: Vec::new(),
                max_density: 0.0,
            };
        }
        let sorted = sorted_copy(data);
        let n = sorted.len();
        let std_d = std_dev(data);
        let iqr_val = percentile(&sorted, 0.75) - percentile(&sorted, 0.25);
        let spread = std_d.min(iqr_val / 1.34);
        let bw = (0.9 * spread / (n as f64).powf(0.2)).max(f64::EPSILON);

        let lo = sorted[0] - 3.0 * bw;
        let hi = sorted[n - 1] + 3.0 * bw;
        let step = (hi - lo) / n_points.max(1) as f64;

        let mut max_d = 0.0_f64;
        let points: Vec<(f64, f64)> = (0..n_points)
            .map(|i| {
                let x = lo + i as f64 * step;
                let d: f64 = data
                    .iter()
                    .map(|xi| {
                        let u = (x - xi) / bw;
                        (-0.5 * u * u).exp() / (bw * (2.0 * PI).sqrt())
                    })
                    .sum::<f64>()
                    / n as f64;
                max_d = max_d.max(d);
                (x, d)
            })
            .collect();

        Self {
            points,
            max_density: max_d,
        }
    }
}

fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}

// ── Grouped box plots ───────────────────────────────────────────

/// A named group for grouped box plots.
#[derive(Debug, Clone)]
pub struct BoxGroup {
    pub name: String,
    pub data: Vec<f64>,
    pub color: String,
}

/// Orientation for rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Vertical,
    Horizontal,
}

/// Rendering configuration.
#[derive(Debug, Clone)]
pub struct BoxPlotConfig {
    pub width: f64,
    pub height: f64,
    pub orientation: Orientation,
    pub show_notch: bool,
    pub show_outliers: bool,
    pub box_width: f64,
    pub margin: f64,
}

impl Default for BoxPlotConfig {
    fn default() -> Self {
        Self {
            width: 600.0,
            height: 400.0,
            orientation: Orientation::Vertical,
            show_notch: false,
            show_outliers: true,
            box_width: 40.0,
            margin: 50.0,
        }
    }
}

// ── SVG rendering ────────────────────────────────────────────────

pub fn render_svg(groups: &[BoxGroup], config: &BoxPlotConfig) -> String {
    let stats: Vec<BoxStats> = groups.iter().map(|g| BoxStats::compute(&g.data)).collect();

    // Find global min/max.
    let global_min = stats
        .iter()
        .map(|s| {
            s.outliers
                .iter()
                .copied()
                .fold(s.whisker_lo, f64::min)
        })
        .fold(f64::INFINITY, f64::min);
    let global_max = stats
        .iter()
        .map(|s| {
            s.outliers
                .iter()
                .copied()
                .fold(s.whisker_hi, f64::max)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let data_range = (global_max - global_min).max(f64::EPSILON);

    let plot_w = config.width - 2.0 * config.margin;
    let plot_h = config.height - 2.0 * config.margin;
    let n = groups.len().max(1);
    let spacing = if config.orientation == Orientation::Vertical {
        plot_w / n as f64
    } else {
        plot_h / n as f64
    };

    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#,
        config.width, config.height,
    );

    for (i, (grp, st)) in groups.iter().zip(stats.iter()).enumerate() {
        let center = config.margin + spacing * (i as f64 + 0.5);
        let half_box = config.box_width / 2.0;

        // Map value to pixel position.
        let to_px = |v: f64| -> f64 {
            let t = (v - global_min) / data_range;
            match config.orientation {
                Orientation::Vertical => config.margin + plot_h * (1.0 - t),
                Orientation::Horizontal => config.margin + plot_w * t,
            }
        };

        let q1_px = to_px(st.q1);
        let q3_px = to_px(st.q3);
        let med_px = to_px(st.median);
        let wlo_px = to_px(st.whisker_lo);
        let whi_px = to_px(st.whisker_hi);

        match config.orientation {
            Orientation::Vertical => {
                // Whisker line.
                let _ = write!(
                    svg,
                    r#"<line x1="{center:.1}" y1="{whi_px:.1}" x2="{center:.1}" y2="{wlo_px:.1}" stroke="gray" stroke-width="1"/>"#,
                );
                // Box.
                let box_y = q3_px.min(q1_px);
                let box_h = (q1_px - q3_px).abs();
                let _ = write!(
                    svg,
                    r#"<rect x="{:.1}" y="{box_y:.1}" width="{:.1}" height="{box_h:.1}" fill="{}" stroke="gray" opacity="0.7"/>"#,
                    center - half_box,
                    config.box_width,
                    grp.color,
                );
                // Median line.
                let _ = write!(
                    svg,
                    r#"<line x1="{:.1}" y1="{med_px:.1}" x2="{:.1}" y2="{med_px:.1}" stroke="black" stroke-width="2"/>"#,
                    center - half_box, center + half_box,
                );
            }
            Orientation::Horizontal => {
                let _ = write!(
                    svg,
                    r#"<line x1="{wlo_px:.1}" y1="{center:.1}" x2="{whi_px:.1}" y2="{center:.1}" stroke="gray" stroke-width="1"/>"#,
                );
                let box_x = q1_px.min(q3_px);
                let box_w = (q3_px - q1_px).abs();
                let _ = write!(
                    svg,
                    r#"<rect x="{box_x:.1}" y="{:.1}" width="{box_w:.1}" height="{:.1}" fill="{}" stroke="gray" opacity="0.7"/>"#,
                    center - half_box,
                    config.box_width,
                    grp.color,
                );
                let _ = write!(
                    svg,
                    r#"<line x1="{med_px:.1}" y1="{:.1}" x2="{med_px:.1}" y2="{:.1}" stroke="black" stroke-width="2"/>"#,
                    center - half_box, center + half_box,
                );
            }
        }

        // Outliers.
        if config.show_outliers {
            for &o in &st.outliers {
                let opx = to_px(o);
                match config.orientation {
                    Orientation::Vertical => {
                        let _ = write!(
                            svg,
                            r#"<circle cx="{center:.1}" cy="{opx:.1}" r="3" fill="none" stroke="{}" stroke-width="1"/>"#,
                            grp.color,
                        );
                    }
                    Orientation::Horizontal => {
                        let _ = write!(
                            svg,
                            r#"<circle cx="{opx:.1}" cy="{center:.1}" r="3" fill="none" stroke="{}" stroke-width="1"/>"#,
                            grp.color,
                        );
                    }
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

    fn sample() -> Vec<f64> {
        vec![
            2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 18.0, 22.0,
            50.0,
        ]
    }

    #[test]
    fn quartiles() {
        let stats = BoxStats::compute(&sample());
        assert!((stats.q1 - 7.0).abs() < 1.0);
        assert!((stats.median - 11.0).abs() < 1.0);
        assert!((stats.q3 - 14.0).abs() < 1.0);
    }

    #[test]
    fn iqr_value() {
        let stats = BoxStats::compute(&sample());
        assert!((stats.iqr - (stats.q3 - stats.q1)).abs() < 1e-9);
    }

    #[test]
    fn whiskers_within_fences() {
        let stats = BoxStats::compute(&sample());
        let fence_lo = stats.q1 - 1.5 * stats.iqr;
        let fence_hi = stats.q3 + 1.5 * stats.iqr;
        assert!(stats.whisker_lo >= fence_lo);
        assert!(stats.whisker_hi <= fence_hi);
    }

    #[test]
    fn outlier_detection() {
        let stats = BoxStats::compute(&sample());
        // 50 is likely an outlier.
        assert!(stats.outliers.contains(&50.0));
        assert!(stats.is_outlier(50.0));
        assert!(!stats.is_outlier(10.0));
    }

    #[test]
    fn notch_bounds() {
        let stats = BoxStats::compute(&sample());
        assert!(stats.notch_lo < stats.median);
        assert!(stats.notch_hi > stats.median);
    }

    #[test]
    fn empty_data() {
        let stats = BoxStats::compute(&[]);
        assert_eq!(stats.n, 0);
        assert!(stats.outliers.is_empty());
    }

    #[test]
    fn single_value() {
        let stats = BoxStats::compute(&[5.0]);
        assert!((stats.median - 5.0).abs() < 1e-9);
        assert_eq!(stats.iqr, 0.0);
    }

    #[test]
    fn violin_shape() {
        let v = ViolinShape::compute(&sample(), 50);
        assert_eq!(v.points.len(), 50);
        assert!(v.max_density > 0.0);
    }

    #[test]
    fn violin_empty() {
        let v = ViolinShape::compute(&[], 50);
        assert!(v.points.is_empty());
    }

    #[test]
    fn render_vertical() {
        let groups = vec![
            BoxGroup {
                name: "A".into(),
                data: sample(),
                color: "#3498db".into(),
            },
        ];
        let svg = render_svg(&groups, &BoxPlotConfig::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn render_horizontal() {
        let groups = vec![
            BoxGroup {
                name: "A".into(),
                data: sample(),
                color: "#e74c3c".into(),
            },
        ];
        let cfg = BoxPlotConfig {
            orientation: Orientation::Horizontal,
            ..Default::default()
        };
        let svg = render_svg(&groups, &cfg);
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn grouped_box_plots() {
        let groups = vec![
            BoxGroup {
                name: "A".into(),
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                color: "#3498db".into(),
            },
            BoxGroup {
                name: "B".into(),
                data: vec![3.0, 4.0, 5.0, 6.0, 7.0],
                color: "#e74c3c".into(),
            },
        ];
        let svg = render_svg(&groups, &BoxPlotConfig::default());
        // Should have two boxes.
        assert!(svg.matches("<rect").count() >= 2);
    }

    #[test]
    fn mean_close_to_median_for_symmetric() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = BoxStats::compute(&data);
        assert!((stats.mean - 3.0).abs() < 1e-9);
        assert!((stats.median - 3.0).abs() < 1e-9);
    }
}
