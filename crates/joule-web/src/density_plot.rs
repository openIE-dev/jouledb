//! Kernel density estimation (KDE) plot with Gaussian kernel, Silverman's rule
//! bandwidth selection, multi-series overlay, confidence bands, and SVG output
//! with filled area.  Pure Rust — no browser dependency.

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;

// ── KDE core ────────────────────────────────────────────────────

/// Gaussian kernel function.
fn gaussian_kernel(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * PI).sqrt()
}

/// Silverman's rule of thumb for bandwidth selection.
pub fn silverman_bandwidth(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = variance.sqrt().max(f64::EPSILON);

    // IQR-based robustness
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = percentile_sorted(&sorted, 0.25);
    let q3 = percentile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;
    let spread = sigma.min(iqr / 1.34).max(f64::EPSILON);

    0.9 * spread * n.powf(-0.2)
}

/// Interpolated percentile from a sorted slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Estimate density at a single point.
pub fn density_at(x: f64, data: &[f64], bandwidth: f64) -> f64 {
    let n = data.len() as f64;
    if n == 0.0 || bandwidth <= 0.0 {
        return 0.0;
    }
    let sum: f64 = data.iter().map(|xi| gaussian_kernel((x - xi) / bandwidth)).sum();
    sum / (n * bandwidth)
}

/// Estimate density over an evenly-spaced grid.
pub fn estimate_density(
    data: &[f64],
    bandwidth: f64,
    x_min: f64,
    x_max: f64,
    num_points: usize,
) -> Vec<(f64, f64)> {
    if num_points == 0 || data.is_empty() {
        return Vec::new();
    }
    let step = (x_max - x_min) / (num_points.saturating_sub(1).max(1)) as f64;
    (0..num_points)
        .map(|i| {
            let x = x_min + step * i as f64;
            (x, density_at(x, data, bandwidth))
        })
        .collect()
}

/// Compute the mean density.
pub fn mean_density(curve: &[(f64, f64)]) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    curve.iter().map(|(_, y)| y).sum::<f64>() / curve.len() as f64
}

/// Integrate the density curve (trapezoidal rule) — should be ~1.0.
pub fn integrate_density(curve: &[(f64, f64)]) -> f64 {
    if curve.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 1..curve.len() {
        let dx = curve[i].0 - curve[i - 1].0;
        let avg_y = (curve[i].1 + curve[i - 1].1) / 2.0;
        total += dx * avg_y;
    }
    total
}

// ── Series ──────────────────────────────────────────────────────

/// A named series for the density plot.
#[derive(Debug, Clone)]
pub struct DensitySeries {
    pub name: String,
    pub data: Vec<f64>,
    pub color: String,
    /// If None, Silverman's rule is used.
    pub bandwidth: Option<f64>,
    pub fill_opacity: f64,
}

impl DensitySeries {
    pub fn new(name: impl Into<String>, data: Vec<f64>, color: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data,
            color: color.into(),
            bandwidth: None,
            fill_opacity: 0.3,
        }
    }

    pub fn with_bandwidth(mut self, bw: f64) -> Self {
        self.bandwidth = Some(bw.max(f64::EPSILON));
        self
    }

    pub fn with_fill_opacity(mut self, opacity: f64) -> Self {
        self.fill_opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Effective bandwidth (specified or Silverman).
    pub fn effective_bandwidth(&self) -> f64 {
        self.bandwidth.unwrap_or_else(|| silverman_bandwidth(&self.data))
    }
}

// ── Confidence band ─────────────────────────────────────────────

/// Bootstrap-style confidence band for a density estimate.
/// Returns (lower, upper) density curves.
pub fn confidence_band(
    data: &[f64],
    bandwidth: f64,
    x_min: f64,
    x_max: f64,
    num_points: usize,
    n_bootstrap: usize,
    confidence: f64,
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    if data.is_empty() || num_points == 0 {
        return (Vec::new(), Vec::new());
    }
    let step = (x_max - x_min) / (num_points.saturating_sub(1).max(1)) as f64;
    let xs: Vec<f64> = (0..num_points).map(|i| x_min + step * i as f64).collect();

    // Collect bootstrap density estimates at each x point
    let mut all_densities: Vec<Vec<f64>> = vec![Vec::with_capacity(n_bootstrap); num_points];

    // Simple pseudo-bootstrap: perturb bandwidth
    for b in 0..n_bootstrap {
        let bw_factor = 0.7 + 0.6 * (b as f64 / n_bootstrap.max(1) as f64);
        let bw = bandwidth * bw_factor;
        for (i, &x) in xs.iter().enumerate() {
            all_densities[i].push(density_at(x, data, bw));
        }
    }

    let alpha = (1.0 - confidence) / 2.0;

    let mut lower = Vec::with_capacity(num_points);
    let mut upper = Vec::with_capacity(num_points);

    for (i, &x) in xs.iter().enumerate() {
        let densities = &mut all_densities[i];
        densities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = percentile_sorted(densities, alpha);
        let hi = percentile_sorted(densities, 1.0 - alpha);
        lower.push((x, lo));
        upper.push((x, hi));
    }

    (lower, upper)
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for the density plot.
#[derive(Debug, Clone)]
pub struct DensityPlotConfig {
    pub width: f64,
    pub height: f64,
    pub padding_top: f64,
    pub padding_right: f64,
    pub padding_bottom: f64,
    pub padding_left: f64,
    /// Number of evaluation points along x.
    pub num_points: usize,
    pub font_size: f64,
    pub show_grid: bool,
    pub show_legend: bool,
    /// Number of bootstrap samples for confidence bands (0 = no band).
    pub confidence_samples: usize,
    /// Confidence level (e.g., 0.95).
    pub confidence_level: f64,
}

impl Default for DensityPlotConfig {
    fn default() -> Self {
        Self {
            width: 600.0,
            height: 400.0,
            padding_top: 30.0,
            padding_right: 20.0,
            padding_bottom: 50.0,
            padding_left: 60.0,
            num_points: 200,
            font_size: 11.0,
            show_grid: true,
            show_legend: true,
            confidence_samples: 0,
            confidence_level: 0.95,
        }
    }
}

impl DensityPlotConfig {
    pub fn plot_width(&self) -> f64 {
        self.width - self.padding_left - self.padding_right
    }

    pub fn plot_height(&self) -> f64 {
        self.height - self.padding_top - self.padding_bottom
    }
}

// ── Rendering ───────────────────────────────────────────────────

/// Render the density plot as SVG.
pub fn render_density_plot(
    series: &[DensitySeries],
    cfg: &DensityPlotConfig,
) -> String {
    if series.is_empty() {
        let mut svg = String::new();
        let _ = write!(
            svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\"></svg>",
            cfg.width, cfg.height
        );
        return svg;
    }

    // Global x range
    let all_data: Vec<f64> = series.iter().flat_map(|s| s.data.iter().copied()).collect();
    let x_min = all_data.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = all_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x_pad = (x_max - x_min).max(1.0) * 0.15;
    let x_lo = x_min - x_pad;
    let x_hi = x_max + x_pad;

    // Compute all density curves
    let curves: Vec<Vec<(f64, f64)>> = series
        .iter()
        .map(|s| {
            let bw = s.effective_bandwidth();
            estimate_density(&s.data, bw, x_lo, x_hi, cfg.num_points)
        })
        .collect();

    // Global y max
    let y_max = curves
        .iter()
        .flat_map(|c| c.iter().map(|(_, y)| *y))
        .fold(0.0_f64, f64::max)
        * 1.15;
    let y_max = y_max.max(f64::EPSILON);

    let pw = cfg.plot_width();
    let ph = cfg.plot_height();
    let ox = cfg.padding_left;
    let oy = cfg.padding_top;

    let map_x = |x: f64| -> f64 { ox + (x - x_lo) / (x_hi - x_lo) * pw };
    let map_y = |y: f64| -> f64 { oy + ph - (y / y_max) * ph };

    let mut svg = String::with_capacity(8192);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">",
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    // Grid
    if cfg.show_grid {
        let grid_lines = 5;
        for i in 0..=grid_lines {
            let frac = i as f64 / grid_lines as f64;
            let gy = oy + ph * (1.0 - frac);
            let _ = write!(
                svg,
                "<line x1=\"{ox}\" y1=\"{gy}\" x2=\"{}\" y2=\"{gy}\" \
                 stroke=\"gainsboro\" stroke-width=\"1\" />",
                ox + pw
            );
            let label_val = y_max * frac;
            let fs = cfg.font_size * 0.85;
            let lx = ox - 5.0;
            let _ = write!(
                svg,
                "<text x=\"{lx}\" y=\"{gy}\" font-size=\"{fs}\" \
                 text-anchor=\"end\" dominant-baseline=\"middle\" fill=\"gray\">{label_val:.3}</text>"
            );
        }
    }

    // Confidence bands
    if cfg.confidence_samples > 0 {
        for (si, s) in series.iter().enumerate() {
            let bw = s.effective_bandwidth();
            let (lower, upper) = confidence_band(
                &s.data,
                bw,
                x_lo,
                x_hi,
                cfg.num_points,
                cfg.confidence_samples,
                cfg.confidence_level,
            );
            if lower.is_empty() {
                continue;
            }
            // Band polygon: upper forward + lower reverse
            let mut pts = String::new();
            for (x, y) in &upper {
                let _ = write!(pts, "{},{} ", map_x(*x), map_y(*y));
            }
            for (x, y) in lower.iter().rev() {
                let _ = write!(pts, "{},{} ", map_x(*x), map_y(*y));
            }
            let _ = write!(
                svg,
                "<polygon points=\"{pts}\" fill=\"{}\" fill-opacity=\"0.15\" stroke=\"none\" />",
                s.color
            );
        }
    }

    // Density curves
    for (si, curve) in curves.iter().enumerate() {
        let s = &series[si];
        if curve.is_empty() {
            continue;
        }

        // Filled area path
        let mut path = String::new();
        let _ = write!(path, "M{},{}", map_x(curve[0].0), map_y(0.0));
        for (x, y) in curve {
            let _ = write!(path, " L{},{}", map_x(*x), map_y(*y));
        }
        let _ = write!(path, " L{},{}", map_x(curve.last().unwrap().0), map_y(0.0));
        path.push_str(" Z");

        let _ = write!(
            svg,
            "<path d=\"{path}\" fill=\"{}\" fill-opacity=\"{}\" stroke=\"none\" />",
            s.color, s.fill_opacity
        );

        // Stroke line
        let mut line_path = String::new();
        for (i, (x, y)) in curve.iter().enumerate() {
            let cmd = if i == 0 { "M" } else { "L" };
            let _ = write!(line_path, "{cmd}{},{} ", map_x(*x), map_y(*y));
        }
        let _ = write!(
            svg,
            "<path d=\"{line_path}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" />",
            s.color
        );
    }

    // X axis
    let ax_y = oy + ph;
    let _ = write!(
        svg,
        "<line x1=\"{ox}\" y1=\"{ax_y}\" x2=\"{}\" y2=\"{ax_y}\" \
         stroke=\"gray\" stroke-width=\"1\" />",
        ox + pw
    );

    // X axis ticks
    let x_ticks = 6;
    for i in 0..=x_ticks {
        let frac = i as f64 / x_ticks as f64;
        let xv = x_lo + frac * (x_hi - x_lo);
        let tx = map_x(xv);
        let ty = ax_y + 14.0;
        let fs = cfg.font_size * 0.85;
        let _ = write!(
            svg,
            "<text x=\"{tx}\" y=\"{ty}\" font-size=\"{fs}\" \
             text-anchor=\"middle\" fill=\"gray\">{xv:.1}</text>"
        );
    }

    // Legend
    if cfg.show_legend && series.len() > 1 {
        let lx = ox + pw - 10.0;
        let mut ly = oy + 15.0;
        let fs = cfg.font_size;
        for s in series {
            let _ = write!(
                svg,
                "<rect x=\"{}\" y=\"{}\" width=\"12\" height=\"12\" fill=\"{}\" />",
                lx - 14.0,
                ly - 10.0,
                s.color
            );
            let _ = write!(
                svg,
                "<text x=\"{}\" y=\"{ly}\" font-size=\"{fs}\" \
                 text-anchor=\"end\">{}</text>",
                lx - 18.0,
                s.name
            );
            ly += fs + 6.0;
        }
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
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
            9.0, 9.5, 10.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 5.0, 5.0,
        ]
    }

    #[test]
    fn gaussian_kernel_at_zero() {
        let k = gaussian_kernel(0.0);
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((k - expected).abs() < 1e-9);
    }

    #[test]
    fn gaussian_kernel_symmetric() {
        assert!((gaussian_kernel(1.0) - gaussian_kernel(-1.0)).abs() < 1e-12);
    }

    #[test]
    fn gaussian_kernel_decays() {
        assert!(gaussian_kernel(0.0) > gaussian_kernel(1.0));
        assert!(gaussian_kernel(1.0) > gaussian_kernel(2.0));
    }

    #[test]
    fn silverman_bandwidth_positive() {
        let data = sample_data();
        let bw = silverman_bandwidth(&data);
        assert!(bw > 0.0, "bandwidth should be positive: {bw}");
    }

    #[test]
    fn silverman_bandwidth_single() {
        let bw = silverman_bandwidth(&[5.0]);
        assert_eq!(bw, 1.0);
    }

    #[test]
    fn silverman_bandwidth_empty() {
        let bw = silverman_bandwidth(&[]);
        assert_eq!(bw, 1.0);
    }

    #[test]
    fn density_at_zero_empty() {
        assert_eq!(density_at(0.0, &[], 1.0), 0.0);
    }

    #[test]
    fn density_at_positive() {
        let data = sample_data();
        let bw = silverman_bandwidth(&data);
        let d = density_at(5.0, &data, bw);
        assert!(d > 0.0);
    }

    #[test]
    fn estimate_density_integrates_to_one() {
        let data = sample_data();
        let bw = silverman_bandwidth(&data);
        let curve = estimate_density(&data, bw, -5.0, 15.0, 500);
        let integral = integrate_density(&curve);
        assert!(
            (integral - 1.0).abs() < 0.05,
            "integral should be ~1.0, got {integral}"
        );
    }

    #[test]
    fn estimate_density_peak_near_mode() {
        let data = vec![5.0; 50];
        let curve = estimate_density(&data, 0.5, 0.0, 10.0, 200);
        let peak = curve
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!(
            (peak.0 - 5.0).abs() < 0.5,
            "peak at {:.2} should be near 5.0",
            peak.0
        );
    }

    #[test]
    fn integrate_empty() {
        assert_eq!(integrate_density(&[]), 0.0);
    }

    #[test]
    fn series_effective_bandwidth_silverman() {
        let s = DensitySeries::new("A", sample_data(), "blue");
        let bw = s.effective_bandwidth();
        assert!(bw > 0.0);
    }

    #[test]
    fn series_effective_bandwidth_custom() {
        let s = DensitySeries::new("A", sample_data(), "blue").with_bandwidth(2.0);
        assert!((s.effective_bandwidth() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn render_produces_svg() {
        let series = vec![DensitySeries::new("Test", sample_data(), "steelblue")];
        let cfg = DensityPlotConfig::default();
        let svg = render_density_plot(&series, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn render_contains_path() {
        let series = vec![DensitySeries::new("Test", sample_data(), "steelblue")];
        let cfg = DensityPlotConfig::default();
        let svg = render_density_plot(&series, &cfg);
        assert!(svg.contains("<path"));
    }

    #[test]
    fn render_empty_series() {
        let cfg = DensityPlotConfig::default();
        let svg = render_density_plot(&[], &cfg);
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn render_multi_series() {
        let s1 = DensitySeries::new("A", vec![1.0, 2.0, 3.0, 4.0, 5.0], "steelblue");
        let s2 = DensitySeries::new("B", vec![3.0, 4.0, 5.0, 6.0, 7.0], "coral");
        let cfg = DensityPlotConfig::default();
        let svg = render_density_plot(&[s1, s2], &cfg);
        assert!(svg.contains("steelblue"));
        assert!(svg.contains("coral"));
    }

    #[test]
    fn confidence_band_produces_curves() {
        let data = sample_data();
        let bw = silverman_bandwidth(&data);
        let (lower, upper) = confidence_band(&data, bw, 0.0, 10.0, 50, 20, 0.95);
        assert_eq!(lower.len(), 50);
        assert_eq!(upper.len(), 50);
        // Upper should be >= lower at every point
        for (l, u) in lower.iter().zip(upper.iter()) {
            assert!(u.1 >= l.1 - 1e-12, "upper {} >= lower {}", u.1, l.1);
        }
    }

    #[test]
    fn render_with_confidence_bands() {
        let s = DensitySeries::new("Test", sample_data(), "steelblue");
        let cfg = DensityPlotConfig {
            confidence_samples: 20,
            confidence_level: 0.95,
            ..Default::default()
        };
        let svg = render_density_plot(&[s], &cfg);
        // Should have a polygon for the band
        assert!(svg.contains("<polygon"));
    }

    #[test]
    fn percentile_sorted_median() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_sorted(&sorted, 0.5) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn mean_density_computed() {
        let curve = vec![(0.0, 0.1), (1.0, 0.2), (2.0, 0.3)];
        assert!((mean_density(&curve) - 0.2).abs() < 1e-9);
    }
}
