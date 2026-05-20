//! Statistical chart types: histogram, boxplot, violin, KDE, QQ.
//!
//! Replaces R/ggplot2 geom_histogram, geom_boxplot, geom_violin,
//! Python seaborn.histplot, seaborn.boxplot, seaborn.violinplot,
//! MATLAB histogram(), boxplot().
//!
//! All output is SVG.

// ── Histogram ──────────────────────────────────────────────────────

/// Histogram configuration.
#[derive(Debug, Clone)]
pub struct HistConfig {
    pub width: f64,
    pub height: f64,
    pub bins: BinMethod,
    pub color: String,
    pub stroke: String,
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub density: bool,       // normalize to probability density
    pub cumulative: bool,
    pub orientation: Orientation,
}

impl Default for HistConfig {
    fn default() -> Self {
        Self {
            width: 600.0,
            height: 400.0,
            bins: BinMethod::Auto,
            color: "#4C78A8".to_string(),
            stroke: "#2a5783".to_string(),
            title: None,
            x_label: None,
            y_label: Some("Count".to_string()),
            density: false,
            cumulative: false,
            orientation: Orientation::Vertical,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BinMethod {
    Auto,
    Count(usize),
    Width(f64),
    Sturges,
    Scott,
    FreedmanDiaconis,
}

#[derive(Debug, Clone, Copy)]
pub enum Orientation {
    Vertical,
    Horizontal,
}

/// Compute bin edges and counts.
pub fn histogram_bins(data: &[f64], method: &BinMethod) -> (Vec<f64>, Vec<usize>) {
    let n = data.len();
    if n == 0 { return (vec![], vec![]); }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let max = sorted[n - 1];
    let range = (max - min).max(1e-10);

    let n_bins = match method {
        BinMethod::Count(k) => *k,
        BinMethod::Width(w) => (range / w).ceil() as usize,
        BinMethod::Sturges => (1.0 + (n as f64).log2()).ceil() as usize,
        BinMethod::Scott => {
            let std = std_dev(data);
            (range / (3.5 * std / (n as f64).cbrt())).ceil().max(1.0) as usize
        }
        BinMethod::FreedmanDiaconis => {
            let iqr = percentile(&sorted, 0.75) - percentile(&sorted, 0.25);
            let bw = 2.0 * iqr / (n as f64).cbrt();
            if bw > 0.0 { (range / bw).ceil() as usize } else { (n as f64).sqrt() as usize }
        }
        BinMethod::Auto => {
            // Use Freedman-Diaconis if IQR > 0, else Sturges
            let iqr = percentile(&sorted, 0.75) - percentile(&sorted, 0.25);
            if iqr > 0.0 {
                let bw = 2.0 * iqr / (n as f64).cbrt();
                (range / bw).ceil() as usize
            } else {
                (1.0 + (n as f64).log2()).ceil() as usize
            }
        }
    }.max(1);

    let bin_width = range / n_bins as f64;
    let edges: Vec<f64> = (0..=n_bins).map(|i| min + i as f64 * bin_width).collect();
    let mut counts = vec![0usize; n_bins];

    for &v in data {
        let idx = ((v - min) / bin_width).floor() as usize;
        let idx = idx.min(n_bins - 1);
        counts[idx] += 1;
    }

    (edges, counts)
}

/// Render a histogram to SVG.
pub fn histogram_svg(data: &[f64], config: &HistConfig) -> String {
    let (edges, counts) = histogram_bins(data, &config.bins);
    if edges.is_empty() { return "<svg></svg>".to_string(); }

    let n_bins = counts.len();
    let max_count = *counts.iter().max().unwrap_or(&1) as f64;
    let padding = 50.0;
    let plot_w = config.width - 2.0 * padding;
    let plot_h = config.height - 2.0 * padding;

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">",
        config.width, config.height
    );
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    if let Some(title) = &config.title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{title}</text>",
            config.width / 2.0
        ));
    }

    // Bars
    let bar_w = plot_w / n_bins as f64;
    for (i, &count) in counts.iter().enumerate() {
        let h = (count as f64 / max_count) * plot_h;
        let x = padding + i as f64 * bar_w;
        let y = padding + plot_h - h;
        svg.push_str(&format!(
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{h:.1}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"0.5\"/>",
            bar_w - 1.0, config.color, config.stroke
        ));
    }

    // X axis labels
    let n_labels = n_bins.min(10);
    let step = (n_bins / n_labels).max(1);
    for i in (0..=n_bins).step_by(step) {
        if i < edges.len() {
            let x = padding + i as f64 * bar_w;
            svg.push_str(&format!(
                "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"10\">{:.2}</text>",
                padding + plot_h + 15.0, edges[i]
            ));
        }
    }

    // Y axis labels
    for i in 0..5 {
        let t = i as f64 / 4.0;
        let val = t * max_count;
        let y = padding + plot_h - t * plot_h;
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{y:.1}\" text-anchor=\"end\" font-size=\"10\" dominant-baseline=\"middle\">{:.0}</text>",
            padding - 5.0, val
        ));
    }

    // Axis labels
    if let Some(xl) = &config.x_label {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"12\">{xl}</text>",
            config.width / 2.0, config.height - 5.0
        ));
    }

    svg.push_str("</svg>");
    svg
}

// ── Boxplot ────────────────────────────────────────────────────────

/// Five-number summary for boxplot.
#[derive(Debug, Clone)]
pub struct BoxStats {
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
    pub mean: Option<f64>,
}

/// Compute five-number summary with outlier detection.
pub fn box_stats(data: &[f64]) -> BoxStats {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return BoxStats { min: 0.0, q1: 0.0, median: 0.0, q3: 0.0, max: 0.0, outliers: vec![], mean: None };
    }

    let q1 = percentile(&sorted, 0.25);
    let median = percentile(&sorted, 0.50);
    let q3 = percentile(&sorted, 0.75);
    let iqr = q3 - q1;
    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    let whisker_lo = sorted.iter().copied().find(|&v| v >= lower_fence).unwrap_or(sorted[0]);
    let whisker_hi = sorted.iter().rev().copied().find(|&v| v <= upper_fence).unwrap_or(sorted[n - 1]);
    let outliers: Vec<f64> = sorted.iter().copied().filter(|&v| v < lower_fence || v > upper_fence).collect();
    let mean = Some(sorted.iter().sum::<f64>() / n as f64);

    BoxStats { min: whisker_lo, q1, median, q3, max: whisker_hi, outliers, mean }
}

/// Render boxplots to SVG.
pub fn boxplot_svg(groups: &[(&str, &[f64])], width: f64, height: f64, title: Option<&str>) -> String {
    let padding = 60.0;
    let plot_w = width - 2.0 * padding;
    let plot_h = height - 2.0 * padding;
    let n = groups.len();
    let box_w = (plot_w / n as f64) * 0.6;
    let gap = plot_w / n as f64;

    // Compute global range
    let stats: Vec<BoxStats> = groups.iter().map(|(_, d)| box_stats(d)).collect();
    let global_min = stats.iter().map(|s| s.min.min(s.outliers.iter().copied().fold(f64::INFINITY, f64::min))).fold(f64::INFINITY, f64::min);
    let global_max = stats.iter().map(|s| s.max.max(s.outliers.iter().copied().fold(f64::NEG_INFINITY, f64::max))).fold(f64::NEG_INFINITY, f64::max);
    let data_range = (global_max - global_min).max(1e-10);
    let y_min = global_min - data_range * 0.05;
    let y_max = global_max + data_range * 0.05;
    let y_range = y_max - y_min;

    let to_y = |v: f64| -> f64 { padding + plot_h - (v - y_min) / y_range * plot_h };

    let mut svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">");
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    if let Some(t) = title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{t}</text>",
            width / 2.0
        ));
    }

    for (i, ((name, _), s)) in groups.iter().zip(stats.iter()).enumerate() {
        let cx = padding + gap * (i as f64 + 0.5);
        let x0 = cx - box_w / 2.0;

        // Whiskers
        svg.push_str(&format!(
            "<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#333\" stroke-width=\"1\"/>",
            to_y(s.max), to_y(s.q3)
        ));
        svg.push_str(&format!(
            "<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#333\" stroke-width=\"1\"/>",
            to_y(s.q1), to_y(s.min)
        ));
        // Whisker caps
        let cap_w = box_w * 0.3;
        svg.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#333\" stroke-width=\"1\"/>",
            cx - cap_w, to_y(s.max), cx + cap_w, to_y(s.max)
        ));
        svg.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#333\" stroke-width=\"1\"/>",
            cx - cap_w, to_y(s.min), cx + cap_w, to_y(s.min)
        ));

        // Box (IQR)
        let box_top = to_y(s.q3);
        let box_bot = to_y(s.q1);
        svg.push_str(&format!(
            "<rect x=\"{x0:.1}\" y=\"{box_top:.1}\" width=\"{box_w:.1}\" height=\"{:.1}\" fill=\"#4C78A8\" fill-opacity=\"0.7\" stroke=\"#333\" stroke-width=\"1\"/>",
            box_bot - box_top
        ));

        // Median line
        svg.push_str(&format!(
            "<line x1=\"{x0:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"white\" stroke-width=\"2\"/>",
            to_y(s.median), x0 + box_w, to_y(s.median)
        ));

        // Mean diamond
        if let Some(mean) = s.mean {
            let my = to_y(mean);
            svg.push_str(&format!(
                "<polygon points=\"{cx:.1},{:.1} {:.1},{my:.1} {cx:.1},{:.1} {:.1},{my:.1}\" fill=\"white\" stroke=\"#333\" stroke-width=\"0.5\"/>",
                my - 4.0, cx - 4.0, my + 4.0, cx + 4.0
            ));
        }

        // Outliers
        for &o in &s.outliers {
            svg.push_str(&format!(
                "<circle cx=\"{cx:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"none\" stroke=\"#333\" stroke-width=\"1\"/>",
                to_y(o)
            ));
        }

        // Label
        svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"10\">{name}</text>",
            padding + plot_h + 15.0
        ));
    }

    svg.push_str("</svg>");
    svg
}

// ── Violin plot ────────────────────────────────────────────────────

/// Render violin plots to SVG (mirrored KDE + boxplot).
pub fn violin_svg(groups: &[(&str, &[f64])], width: f64, height: f64, title: Option<&str>) -> String {
    let padding = 60.0;
    let plot_w = width - 2.0 * padding;
    let plot_h = height - 2.0 * padding;
    let n = groups.len();
    let gap = plot_w / n as f64;
    let violin_w = gap * 0.4;

    // Global range
    let all_vals: Vec<f64> = groups.iter().flat_map(|(_, d)| d.iter().copied()).collect();
    let global_min = all_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let global_max = all_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let data_range = (global_max - global_min).max(1e-10);
    let y_min = global_min - data_range * 0.05;
    let y_max = global_max + data_range * 0.05;
    let y_range = y_max - y_min;

    let to_y = |v: f64| -> f64 { padding + plot_h - (v - y_min) / y_range * plot_h };

    let mut svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">");
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    if let Some(t) = title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{t}</text>",
            width / 2.0
        ));
    }

    for (i, (name, data)) in groups.iter().enumerate() {
        let cx = padding + gap * (i as f64 + 0.5);

        // KDE
        let n_eval = 50;
        let eval_points: Vec<f64> = (0..n_eval).map(|j| y_min + (j as f64 / (n_eval - 1) as f64) * y_range).collect();
        let bw = kde_bandwidth(data);
        let densities: Vec<f64> = eval_points.iter().map(|&y| kde_at(data, y, bw)).collect();
        let max_density = densities.iter().copied().fold(0.0f64, f64::max).max(1e-10);

        // Build mirrored path
        let mut left_path = String::new();
        let mut right_path = String::new();
        for (j, (&y_val, &d)) in eval_points.iter().zip(densities.iter()).enumerate() {
            let sy = to_y(y_val);
            let half_w = (d / max_density) * violin_w;
            if j == 0 {
                left_path.push_str(&format!("M{:.1},{sy:.1}", cx - half_w));
                right_path.push_str(&format!("L{:.1},{sy:.1}", cx + half_w));
            } else {
                left_path.push_str(&format!(" L{:.1},{sy:.1}", cx - half_w));
                right_path = format!("L{:.1},{sy:.1} ", cx + half_w) + &right_path;
            }
        }

        svg.push_str(&format!(
            "<path d=\"{left_path} {right_path} Z\" fill=\"#4C78A8\" fill-opacity=\"0.5\" stroke=\"#4C78A8\" stroke-width=\"1\"/>"
        ));

        // Inner boxplot (thin)
        let s = box_stats(data);
        let inner_w = 4.0;
        svg.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{inner_w}\" height=\"{:.1}\" fill=\"#333\" fill-opacity=\"0.7\"/>",
            cx - inner_w / 2.0, to_y(s.q3), to_y(s.q1) - to_y(s.q3)
        ));
        // Median dot
        svg.push_str(&format!(
            "<circle cx=\"{cx:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"white\" stroke=\"#333\"/>",
            to_y(s.median)
        ));

        // Label
        svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"10\">{name}</text>",
            padding + plot_h + 15.0
        ));
    }

    svg.push_str("</svg>");
    svg
}

// ── KDE (Kernel Density Estimation) ────────────────────────────────

/// Silverman's rule of thumb for bandwidth.
pub fn kde_bandwidth(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 { return 1.0; }
    let std = std_dev(data);
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let iqr = percentile(&sorted, 0.75) - percentile(&sorted, 0.25);
    let spread = (std.min(iqr / 1.34)).max(1e-10);
    0.9 * spread * n.powf(-0.2)
}

/// Evaluate KDE at a single point using Gaussian kernel.
pub fn kde_at(data: &[f64], x: f64, bandwidth: f64) -> f64 {
    let n = data.len() as f64;
    let inv_bw = 1.0 / bandwidth;
    let norm = 1.0 / (n * bandwidth * (2.0 * std::f64::consts::PI).sqrt());
    data.iter().map(|&xi| {
        let u = (x - xi) * inv_bw;
        norm * (-0.5 * u * u).exp()
    }).sum()
}

/// Evaluate KDE over a grid of points.
pub fn kde_curve(data: &[f64], n_points: usize) -> (Vec<f64>, Vec<f64>) {
    if data.is_empty() { return (vec![], vec![]); }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let bw = kde_bandwidth(data);
    let pad = 3.0 * bw;

    let xs: Vec<f64> = (0..n_points).map(|i| (min - pad) + (max - min + 2.0 * pad) * i as f64 / (n_points - 1) as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| kde_at(data, x, bw)).collect();
    (xs, ys)
}

// ── QQ Plot ────────────────────────────────────────────────────────

/// Render a QQ (quantile-quantile) plot against normal distribution.
pub fn qq_svg(data: &[f64], width: f64, height: f64, title: Option<&str>) -> String {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n < 2 { return "<svg></svg>".to_string(); }

    let mean = sorted.iter().sum::<f64>() / n as f64;
    let std = std_dev(data);
    let padding = 50.0;
    let plot_w = width - 2.0 * padding;
    let plot_h = height - 2.0 * padding;

    // Theoretical quantiles (normal distribution via approximation)
    let theoretical: Vec<f64> = (0..n).map(|i| {
        let p = (i as f64 + 0.5) / n as f64;
        normal_ppf(p) // standard normal quantile
    }).collect();

    let t_min = theoretical[0];
    let t_max = theoretical[n - 1];
    let s_min = sorted[0];
    let s_max = sorted[n - 1];

    let to_x = |t: f64| -> f64 { padding + (t - t_min) / (t_max - t_min) * plot_w };
    let to_y = |s: f64| -> f64 { padding + plot_h - (s - s_min) / (s_max - s_min).max(1e-10) * plot_h };

    let mut svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">");
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    if let Some(t) = title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{t}</text>",
            width / 2.0
        ));
    }

    // Reference line (45-degree if data were perfectly normal)
    let ref_y0 = mean + std * t_min;
    let ref_y1 = mean + std * t_max;
    svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#ccc\" stroke-width=\"1\" stroke-dasharray=\"4\"/>",
        to_x(t_min), to_y(ref_y0), to_x(t_max), to_y(ref_y1)
    ));

    // Points
    for (i, (&t, &s)) in theoretical.iter().zip(sorted.iter()).enumerate() {
        svg.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"#4C78A8\" fill-opacity=\"0.7\"/>",
            to_x(t), to_y(s)
        ));
    }

    // Axis labels
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"12\">Theoretical Quantiles</text>",
        width / 2.0, height - 5.0
    ));

    svg.push_str("</svg>");
    svg
}

// ── Helpers ────────────────────────────────────────────────────────

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi.min(sorted.len() - 1)] * frac
}

fn std_dev(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 { return 0.0; }
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// Approximate inverse normal CDF (Abramowitz & Stegun rational approximation).
fn normal_ppf(p: f64) -> f64 {
    if p <= 0.0 { return -6.0; }
    if p >= 1.0 { return 6.0; }
    if (p - 0.5).abs() < 1e-15 { return 0.0; }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Rational approximation coefficients
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let result = t - num / den;

    if p < 0.5 { -result } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        vec![2.3, 4.1, 3.7, 5.2, 1.8, 4.5, 3.2, 6.1, 2.9, 4.8,
             3.5, 5.7, 2.1, 4.3, 3.9, 5.5, 1.5, 4.0, 3.3, 6.5]
    }

    #[test]
    fn histogram_bins_auto() {
        let (edges, counts) = histogram_bins(&sample_data(), &BinMethod::Auto);
        assert!(!edges.is_empty());
        assert!(!counts.is_empty());
        assert_eq!(edges.len(), counts.len() + 1);
        assert_eq!(counts.iter().sum::<usize>(), 20);
    }

    #[test]
    fn histogram_bins_fixed_count() {
        let (edges, counts) = histogram_bins(&sample_data(), &BinMethod::Count(5));
        assert_eq!(counts.len(), 5);
        assert_eq!(counts.iter().sum::<usize>(), 20);
    }

    #[test]
    fn histogram_svg_renders() {
        let svg = histogram_svg(&sample_data(), &HistConfig::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("rect"));
    }

    #[test]
    fn box_stats_correct() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let s = box_stats(&data);
        assert!((s.median - 5.5).abs() < 0.01);
        assert!(s.q1 < s.median);
        assert!(s.q3 > s.median);
    }

    #[test]
    fn box_stats_outliers() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        data.push(100.0); // extreme outlier
        let s = box_stats(&data);
        assert!(!s.outliers.is_empty());
        assert!(s.outliers.contains(&100.0));
    }

    #[test]
    fn boxplot_svg_renders() {
        let d1 = sample_data();
        let d2: Vec<f64> = d1.iter().map(|x| x * 1.5).collect();
        let svg = boxplot_svg(&[("A", &d1), ("B", &d2)], 400.0, 300.0, Some("Test"));
        assert!(svg.contains("rect"));
        assert!(svg.contains("line"));
    }

    #[test]
    fn violin_svg_renders() {
        let d1 = sample_data();
        let d2: Vec<f64> = d1.iter().map(|x| x * 1.5).collect();
        let svg = violin_svg(&[("A", &d1), ("B", &d2)], 400.0, 300.0, Some("Violin"));
        assert!(svg.contains("path"));
        assert!(svg.contains("circle"));
    }

    #[test]
    fn kde_bandwidth_positive() {
        let bw = kde_bandwidth(&sample_data());
        assert!(bw > 0.0);
    }

    #[test]
    fn kde_curve_sums_positive() {
        let (xs, ys) = kde_curve(&sample_data(), 100);
        assert_eq!(xs.len(), 100);
        assert_eq!(ys.len(), 100);
        assert!(ys.iter().all(|&y| y >= 0.0));
    }

    #[test]
    fn qq_plot_renders() {
        let svg = qq_svg(&sample_data(), 400.0, 300.0, Some("QQ"));
        assert!(svg.contains("circle"));
        assert!(svg.contains("line"));
    }

    #[test]
    fn normal_ppf_symmetry() {
        assert!((normal_ppf(0.5)).abs() < 0.01);
        assert!((normal_ppf(0.025) + 1.96).abs() < 0.05);
        assert!((normal_ppf(0.975) - 1.96).abs() < 0.05);
    }

    #[test]
    fn percentile_correct() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&data, 0.5) - 3.0).abs() < 0.01);
        assert!((percentile(&data, 1.0) - 5.0).abs() < 0.01);
    }
}
