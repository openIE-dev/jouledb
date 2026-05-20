//! Histogram computation: auto-binning (Sturges/Scott/Freedman-Diaconis),
//! custom bin edges, frequency/density/cumulative modes, bin statistics,
//! kernel density estimation, and multi-series overlay.  Pure Rust SVG output.

use std::fmt::Write as FmtWrite;

// ── Binning rules ────────────────────────────────────────────────

/// Strategy for choosing the number of bins.
#[derive(Debug, Clone)]
pub enum BinRule {
    /// Sturges' rule: ceil(1 + log2(n)).
    Sturges,
    /// Scott's rule: bin width = 3.49 * std / n^(1/3).
    Scott,
    /// Freedman-Diaconis: bin width = 2 * IQR / n^(1/3).
    FreedmanDiaconis,
    /// Fixed number of equal-width bins.
    Fixed(usize),
    /// Custom bin edges (must be sorted, at least 2 values).
    Custom(Vec<f64>),
}

/// How y-axis values are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramMode {
    Frequency,
    Density,
    Cumulative,
}

// ── Statistics helpers ───────────────────────────────────────────

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    let m = mean(data);
    let var = data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / data.len().max(1) as f64;
    var.sqrt()
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

fn iqr(sorted: &[f64]) -> f64 {
    percentile(sorted, 0.75) - percentile(sorted, 0.25)
}

fn median(sorted: &[f64]) -> f64 {
    percentile(sorted, 0.5)
}

// ── Bin edge computation ─────────────────────────────────────────

/// Compute bin edges from data and rule.
pub fn compute_edges(data: &[f64], rule: &BinRule) -> Vec<f64> {
    if data.is_empty() {
        return vec![0.0, 1.0];
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = sorted[0];
    let hi = sorted[sorted.len() - 1];

    match rule {
        BinRule::Custom(edges) => edges.clone(),
        BinRule::Fixed(n) => equal_edges(lo, hi, *n),
        BinRule::Sturges => {
            let k = (1.0 + (data.len() as f64).log2()).ceil() as usize;
            equal_edges(lo, hi, k.max(1))
        }
        BinRule::Scott => {
            let s = std_dev(data);
            let w = 3.49 * s / (data.len() as f64).powf(1.0 / 3.0);
            let k = ((hi - lo) / w.max(f64::EPSILON)).ceil() as usize;
            equal_edges(lo, hi, k.max(1))
        }
        BinRule::FreedmanDiaconis => {
            let q = iqr(&sorted);
            let w = 2.0 * q / (data.len() as f64).powf(1.0 / 3.0);
            let k = ((hi - lo) / w.max(f64::EPSILON)).ceil() as usize;
            equal_edges(lo, hi, k.max(1))
        }
    }
}

fn equal_edges(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let range = hi - lo;
    let step = if range.abs() < f64::EPSILON {
        1.0
    } else {
        range / n as f64
    };
    (0..=n).map(|i| lo + i as f64 * step).collect()
}

// ── Bin ──────────────────────────────────────────────────────────

/// A single histogram bin.
#[derive(Debug, Clone)]
pub struct Bin {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
    pub values: Vec<f64>,
}

impl Bin {
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn mean(&self) -> f64 {
        mean(&self.values)
    }

    pub fn median(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut s = self.values.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median(&s)
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }
}

// ── Histogram ────────────────────────────────────────────────────

/// A computed histogram.
#[derive(Debug, Clone)]
pub struct Histogram {
    pub bins: Vec<Bin>,
    pub total_count: usize,
    pub mode: HistogramMode,
}

impl Histogram {
    /// Build a histogram from data.
    pub fn build(data: &[f64], rule: &BinRule, mode: HistogramMode) -> Self {
        let edges = compute_edges(data, rule);
        let n_bins = if edges.len() < 2 { 1 } else { edges.len() - 1 };
        let mut bins: Vec<Bin> = (0..n_bins)
            .map(|i| Bin {
                lo: edges[i],
                hi: edges.get(i + 1).copied().unwrap_or(edges[i] + 1.0),
                count: 0,
                values: Vec::new(),
            })
            .collect();

        for &v in data {
            // Binary search for the right bin.
            let idx = match bins.binary_search_by(|b| {
                if v < b.lo {
                    std::cmp::Ordering::Greater
                } else if v >= b.hi && !(b.hi == edges[edges.len() - 1] && v == b.hi) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            }) {
                Ok(i) => i,
                Err(_) => {
                    // Put in last bin if beyond range.
                    bins.len() - 1
                }
            };
            if idx < bins.len() {
                bins[idx].count += 1;
                bins[idx].values.push(v);
            }
        }

        Self {
            bins,
            total_count: data.len(),
            mode,
        }
    }

    /// Y-value for a bin based on mode.
    pub fn y_value(&self, bin_index: usize) -> f64 {
        let bin = &self.bins[bin_index];
        match self.mode {
            HistogramMode::Frequency => bin.count as f64,
            HistogramMode::Density => {
                let w = bin.width().max(f64::EPSILON);
                bin.count as f64 / (self.total_count.max(1) as f64 * w)
            }
            HistogramMode::Cumulative => {
                let cum: usize = self.bins[..=bin_index].iter().map(|b| b.count).sum();
                cum as f64
            }
        }
    }

    /// All y-values.
    pub fn y_values(&self) -> Vec<f64> {
        (0..self.bins.len()).map(|i| self.y_value(i)).collect()
    }

    pub fn max_y(&self) -> f64 {
        self.y_values()
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
    }
}

// ── Kernel Density Estimation ────────────────────────────────────

/// Gaussian KDE evaluated at evenly spaced points.
pub fn kde(data: &[f64], n_points: usize, bandwidth: Option<f64>) -> Vec<(f64, f64)> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = sorted[0];
    let hi = sorted[sorted.len() - 1];
    let pad = (hi - lo) * 0.1;
    let bw = bandwidth.unwrap_or_else(|| {
        // Silverman's rule of thumb.
        let s = std_dev(data);
        let q = iqr(&sorted);
        let spread = s.min(q / 1.34);
        0.9 * spread / (data.len() as f64).powf(0.2)
    });
    let bw = bw.max(f64::EPSILON);
    let step = ((hi - lo) + 2.0 * pad) / n_points.max(1) as f64;
    let start = lo - pad;

    (0..n_points)
        .map(|i| {
            let x = start + i as f64 * step;
            let density: f64 = data
                .iter()
                .map(|xi| {
                    let u = (x - xi) / bw;
                    (-0.5 * u * u).exp() / (bw * (2.0 * std::f64::consts::PI).sqrt())
                })
                .sum::<f64>()
                / data.len() as f64;
            (x, density)
        })
        .collect()
}

// ── Multi-series ─────────────────────────────────────────────────

/// A named data series for overlay histograms.
#[derive(Debug, Clone)]
pub struct HistogramSeries {
    pub name: String,
    pub data: Vec<f64>,
    pub color: String,
}

/// Build overlaid histograms sharing the same bin edges.
pub fn overlay(series: &[HistogramSeries], rule: &BinRule, mode: HistogramMode) -> Vec<(String, Histogram)> {
    // Gather all data to compute shared edges.
    let all: Vec<f64> = series.iter().flat_map(|s| s.data.iter().copied()).collect();
    let edges = compute_edges(&all, rule);
    let shared_rule = BinRule::Custom(edges);
    series
        .iter()
        .map(|s| {
            let h = Histogram::build(&s.data, &shared_rule, mode);
            (s.name.clone(), h)
        })
        .collect()
}

// ── SVG rendering ────────────────────────────────────────────────

pub fn render_svg(hist: &Histogram, width: f64, height: f64, color: &str) -> String {
    let margin = 40.0;
    let plot_w = width - 2.0 * margin;
    let plot_h = height - 2.0 * margin;
    let max_y = hist.max_y().max(f64::EPSILON);
    let lo = hist.bins.first().map_or(0.0, |b| b.lo);
    let hi = hist.bins.last().map_or(1.0, |b| b.hi);
    let x_range = (hi - lo).max(f64::EPSILON);

    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">"#,
    );

    for (i, bin) in hist.bins.iter().enumerate() {
        let x = margin + (bin.lo - lo) / x_range * plot_w;
        let w = bin.width() / x_range * plot_w;
        let y_val = hist.y_value(i);
        let bar_h = y_val / max_y * plot_h;
        let y = margin + plot_h - bar_h;
        let _ = write!(
            svg,
            r#"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{bar_h:.1}" fill="{color}" opacity="0.75"/>"#,
        );
    }

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        vec![
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 2.0,
        ]
    }

    #[test]
    fn sturges_rule() {
        let data = sample_data();
        let h = Histogram::build(&data, &BinRule::Sturges, HistogramMode::Frequency);
        assert!(!h.bins.is_empty());
        let total: usize = h.bins.iter().map(|b| b.count).sum();
        assert_eq!(total, data.len());
    }

    #[test]
    fn scott_rule() {
        let data = sample_data();
        let h = Histogram::build(&data, &BinRule::Scott, HistogramMode::Frequency);
        let total: usize = h.bins.iter().map(|b| b.count).sum();
        assert_eq!(total, data.len());
    }

    #[test]
    fn freedman_diaconis_rule() {
        let data = sample_data();
        let h = Histogram::build(&data, &BinRule::FreedmanDiaconis, HistogramMode::Frequency);
        let total: usize = h.bins.iter().map(|b| b.count).sum();
        assert_eq!(total, data.len());
    }

    #[test]
    fn fixed_bins() {
        let data = sample_data();
        let h = Histogram::build(&data, &BinRule::Fixed(5), HistogramMode::Frequency);
        assert_eq!(h.bins.len(), 5);
    }

    #[test]
    fn custom_edges() {
        let data = sample_data();
        let edges = vec![0.0, 3.0, 6.0, 10.0];
        let h = Histogram::build(&data, &BinRule::Custom(edges), HistogramMode::Frequency);
        assert_eq!(h.bins.len(), 3);
        let total: usize = h.bins.iter().map(|b| b.count).sum();
        assert_eq!(total, data.len());
    }

    #[test]
    fn density_mode() {
        let data = sample_data();
        let h = Histogram::build(&data, &BinRule::Fixed(4), HistogramMode::Density);
        // Density values should be non-negative.
        for i in 0..h.bins.len() {
            assert!(h.y_value(i) >= 0.0);
        }
    }

    #[test]
    fn cumulative_mode() {
        let data = sample_data();
        let h = Histogram::build(&data, &BinRule::Fixed(4), HistogramMode::Cumulative);
        let ys = h.y_values();
        // Should be monotonically non-decreasing.
        for w in ys.windows(2) {
            assert!(w[1] >= w[0]);
        }
        // Last cumulative equals total.
        assert_eq!(*ys.last().unwrap() as usize, data.len());
    }

    #[test]
    fn bin_statistics() {
        let data = vec![1.0, 2.0, 3.0];
        let h = Histogram::build(&data, &BinRule::Fixed(1), HistogramMode::Frequency);
        let bin = &h.bins[0];
        assert!((bin.mean() - 2.0).abs() < 1e-9);
        assert!((bin.median() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn kde_output() {
        let data = sample_data();
        let points = kde(&data, 50, None);
        assert_eq!(points.len(), 50);
        // All densities non-negative.
        for (_x, y) in &points {
            assert!(*y >= 0.0);
        }
    }

    #[test]
    fn multi_series_overlay() {
        let series = vec![
            HistogramSeries {
                name: "A".into(),
                data: vec![1.0, 2.0, 3.0, 4.0],
                color: "#e74c3c".into(),
            },
            HistogramSeries {
                name: "B".into(),
                data: vec![2.0, 3.0, 4.0, 5.0],
                color: "#3498db".into(),
            },
        ];
        let result = overlay(&series, &BinRule::Fixed(3), HistogramMode::Frequency);
        assert_eq!(result.len(), 2);
        // Same number of bins since edges are shared.
        assert_eq!(result[0].1.bins.len(), result[1].1.bins.len());
    }

    #[test]
    fn svg_render() {
        let data = sample_data();
        let h = Histogram::build(&data, &BinRule::Sturges, HistogramMode::Frequency);
        let svg = render_svg(&h, 400.0, 300.0, "#3498db");
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn empty_data() {
        let h = Histogram::build(&[], &BinRule::Sturges, HistogramMode::Frequency);
        assert!(!h.bins.is_empty()); // still has the default bin
        assert_eq!(h.total_count, 0);
    }

    #[test]
    fn single_value_data() {
        let h = Histogram::build(&[42.0], &BinRule::Sturges, HistogramMode::Frequency);
        let total: usize = h.bins.iter().map(|b| b.count).sum();
        assert_eq!(total, 1);
    }
}
