//! Heatmap renderer: 2D grid of values, color scales (linear/quantize/quantile),
//! axis labels, cell tooltips, min/max clamping, missing value handling,
//! row/column clustering, and color palettes.  Pure Rust SVG output.

use std::fmt::Write as FmtWrite;

// ── Color palette ────────────────────────────────────────────────

/// Built-in palette types.
#[derive(Debug, Clone)]
pub enum Palette {
    /// Sequential: light to dark single hue.
    Sequential(Vec<[u8; 3]>),
    /// Diverging: two hues meeting at a neutral midpoint.
    Diverging {
        low: [u8; 3],
        mid: [u8; 3],
        high: [u8; 3],
    },
}

impl Palette {
    /// Viridis-like 5-stop sequential.
    pub fn viridis() -> Self {
        Palette::Sequential(vec![
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ])
    }

    /// Red-white-blue diverging.
    pub fn red_blue() -> Self {
        Palette::Diverging {
            low: [44, 123, 182],
            mid: [255, 255, 255],
            high: [215, 48, 39],
        }
    }

    /// Interpolate a color for `t` in [0, 1].
    pub fn sample(&self, t: f64) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        match self {
            Palette::Sequential(stops) => {
                if stops.is_empty() {
                    return [0, 0, 0];
                }
                if stops.len() == 1 {
                    return stops[0];
                }
                let n = stops.len() - 1;
                let idx = (t * n as f64).min(n as f64 - f64::EPSILON);
                let lo = idx.floor() as usize;
                let hi = (lo + 1).min(n);
                let frac = idx - lo as f64;
                lerp_rgb(stops[lo], stops[hi], frac)
            }
            Palette::Diverging { low, mid, high } => {
                if t < 0.5 {
                    lerp_rgb(*low, *mid, t * 2.0)
                } else {
                    lerp_rgb(*mid, *high, (t - 0.5) * 2.0)
                }
            }
        }
    }

    pub fn to_hex(&self, t: f64) -> String {
        let [r, g, b] = self.sample(t);
        format!("#{r:02x}{g:02x}{b:02x}")
    }
}

fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f64) -> [u8; 3] {
    [
        (a[0] as f64 + (b[0] as f64 - a[0] as f64) * t).round() as u8,
        (a[1] as f64 + (b[1] as f64 - a[1] as f64) * t).round() as u8,
        (a[2] as f64 + (b[2] as f64 - a[2] as f64) * t).round() as u8,
    ]
}

// ── Scale ────────────────────────────────────────────────────────

/// How values map to [0,1] for palette lookup.
#[derive(Debug, Clone)]
pub enum ColorScale {
    /// Linearly between min and max.
    Linear,
    /// Divide range into N equal-width bins.
    Quantize(usize),
    /// Divide data into N equal-count bins.
    Quantile(usize),
}

// ── Heatmap data ─────────────────────────────────────────────────

/// A 2D grid cell — `None` represents missing data.
pub type GridValue = Option<f64>;

/// Row-major 2D grid with axis labels.
#[derive(Debug, Clone)]
pub struct HeatmapData {
    pub values: Vec<Vec<GridValue>>,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
}

impl HeatmapData {
    pub fn new(
        values: Vec<Vec<GridValue>>,
        row_labels: Vec<String>,
        col_labels: Vec<String>,
    ) -> Self {
        Self {
            values,
            row_labels,
            col_labels,
        }
    }

    pub fn rows(&self) -> usize {
        self.values.len()
    }

    pub fn cols(&self) -> usize {
        self.values.first().map_or(0, |r| r.len())
    }

    /// Collect all non-missing values.
    pub fn flat_values(&self) -> Vec<f64> {
        self.values
            .iter()
            .flat_map(|row| row.iter().filter_map(|v| *v))
            .collect()
    }

    pub fn min_value(&self) -> Option<f64> {
        self.flat_values()
            .iter()
            .copied()
            .reduce(f64::min)
    }

    pub fn max_value(&self) -> Option<f64> {
        self.flat_values()
            .iter()
            .copied()
            .reduce(f64::max)
    }

    /// Clamp all values to [lo, hi].
    pub fn clamp(&mut self, lo: f64, hi: f64) {
        for row in &mut self.values {
            for cell in row.iter_mut() {
                if let Some(v) = cell {
                    *v = v.clamp(lo, hi);
                }
            }
        }
    }

    /// Reorder rows by single-linkage clustering (greedy nearest-neighbour).
    pub fn cluster_rows(&mut self) {
        let order = greedy_cluster_order(&self.values);
        self.apply_row_order(&order);
    }

    /// Reorder columns by single-linkage clustering.
    pub fn cluster_cols(&mut self) {
        let ncols = self.cols();
        if ncols == 0 {
            return;
        }
        // Transpose, cluster, transpose back.
        let mut transposed = transpose(&self.values, self.rows(), ncols);
        let order = greedy_cluster_order(&transposed);
        reorder_rows(&mut transposed, &order);
        let mut new_col_labels: Vec<String> = order.iter().map(|i| self.col_labels[*i].clone()).collect();
        let back = transpose(&transposed, ncols, self.rows());
        self.values = back;
        std::mem::swap(&mut self.col_labels, &mut new_col_labels);
    }

    fn apply_row_order(&mut self, order: &[usize]) {
        let new_values: Vec<Vec<GridValue>> = order.iter().map(|i| self.values[*i].clone()).collect();
        let new_labels: Vec<String> = order.iter().map(|i| self.row_labels[*i].clone()).collect();
        self.values = new_values;
        self.row_labels = new_labels;
    }
}

fn transpose(grid: &[Vec<GridValue>], nrows: usize, ncols: usize) -> Vec<Vec<GridValue>> {
    let mut out = vec![vec![None; nrows]; ncols];
    for r in 0..nrows {
        for c in 0..ncols {
            if c < grid[r].len() {
                out[c][r] = grid[r][c];
            }
        }
    }
    out
}

fn reorder_rows(grid: &mut Vec<Vec<GridValue>>, order: &[usize]) {
    let copy: Vec<Vec<GridValue>> = order.iter().map(|i| grid[*i].clone()).collect();
    *grid = copy;
}

fn row_distance(a: &[GridValue], b: &[GridValue]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for (va, vb) in a.iter().zip(b.iter()) {
        if let (Some(x), Some(y)) = (va, vb) {
            sum += (x - y) * (x - y);
            count += 1;
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        (sum / count as f64).sqrt()
    }
}

fn greedy_cluster_order(rows: &[Vec<GridValue>]) -> Vec<usize> {
    let n = rows.len();
    if n == 0 {
        return Vec::new();
    }
    let mut used = vec![false; n];
    let mut order = Vec::with_capacity(n);
    // Start with row 0.
    order.push(0);
    used[0] = true;
    for _ in 1..n {
        let last = *order.last().unwrap();
        let mut best = usize::MAX;
        let mut best_dist = f64::INFINITY;
        for j in 0..n {
            if used[j] {
                continue;
            }
            let d = row_distance(&rows[last], &rows[j]);
            if d < best_dist {
                best_dist = d;
                best = j;
            }
        }
        used[best] = true;
        order.push(best);
    }
    order
}

// ── Config & rendering ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HeatmapConfig {
    pub cell_width: f64,
    pub cell_height: f64,
    pub palette: Palette,
    pub scale: ColorScale,
    pub missing_color: String,
    /// Margin for axis labels.
    pub label_margin: f64,
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self {
            cell_width: 40.0,
            cell_height: 40.0,
            palette: Palette::viridis(),
            scale: ColorScale::Linear,
            missing_color: "#cccccc".into(),
            label_margin: 60.0,
        }
    }
}

/// A rendered cell with position and computed color.
#[derive(Debug, Clone)]
pub struct HeatmapCell {
    pub row: usize,
    pub col: usize,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub value: GridValue,
    pub color: String,
}

pub fn compute_cells(data: &HeatmapData, config: &HeatmapConfig) -> Vec<HeatmapCell> {
    let flat = data.flat_values();
    let min_v = flat.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let max_v = flat.iter().copied().reduce(f64::max).unwrap_or(1.0);
    let range = (max_v - min_v).max(f64::EPSILON);

    let quantiles = match &config.scale {
        ColorScale::Quantile(n) => {
            let mut sorted = flat.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Some(compute_quantile_breaks(&sorted, *n))
        }
        _ => None,
    };

    let mut cells = Vec::new();
    for (r, row) in data.values.iter().enumerate() {
        for (c, val) in row.iter().enumerate() {
            let x = config.label_margin + c as f64 * config.cell_width;
            let y = config.label_margin + r as f64 * config.cell_height;
            let color = match val {
                None => config.missing_color.clone(),
                Some(v) => {
                    let t = match &config.scale {
                        ColorScale::Linear => (v - min_v) / range,
                        ColorScale::Quantize(n) => {
                            let bin = (((v - min_v) / range) * *n as f64).floor() as usize;
                            let bin = bin.min(n - 1);
                            bin as f64 / (*n as f64 - 1.0).max(1.0)
                        }
                        ColorScale::Quantile(_) => {
                            if let Some(breaks) = &quantiles {
                                quantile_bin(*v, breaks)
                            } else {
                                (v - min_v) / range
                            }
                        }
                    };
                    config.palette.to_hex(t)
                }
            };
            cells.push(HeatmapCell {
                row: r,
                col: c,
                x,
                y,
                width: config.cell_width,
                height: config.cell_height,
                value: *val,
                color,
            });
        }
    }
    cells
}

fn compute_quantile_breaks(sorted: &[f64], n: usize) -> Vec<f64> {
    if n <= 1 || sorted.is_empty() {
        return Vec::new();
    }
    (1..n)
        .map(|i| {
            let idx = (i as f64 / n as f64 * sorted.len() as f64) as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

fn quantile_bin(v: f64, breaks: &[f64]) -> f64 {
    let n = breaks.len() + 1;
    for (i, brk) in breaks.iter().enumerate() {
        if v < *brk {
            return i as f64 / (n as f64 - 1.0).max(1.0);
        }
    }
    1.0
}

/// Generate tooltip text for a cell.
pub fn cell_tooltip(data: &HeatmapData, row: usize, col: usize) -> String {
    let rl = data.row_labels.get(row).map(|s| s.as_str()).unwrap_or("?");
    let cl = data.col_labels.get(col).map(|s| s.as_str()).unwrap_or("?");
    let val = data
        .values
        .get(row)
        .and_then(|r| r.get(col))
        .copied()
        .flatten();
    match val {
        Some(v) => format!("{rl} / {cl}: {v:.2}"),
        None => format!("{rl} / {cl}: N/A"),
    }
}

/// Render to SVG.
pub fn render_svg(data: &HeatmapData, config: &HeatmapConfig) -> String {
    let cells = compute_cells(data, config);
    let total_w = config.label_margin + data.cols() as f64 * config.cell_width;
    let total_h = config.label_margin + data.rows() as f64 * config.cell_height;
    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}">"#,
    );
    // Row labels.
    for (i, label) in data.row_labels.iter().enumerate() {
        let y = config.label_margin + i as f64 * config.cell_height + config.cell_height / 2.0;
        let _ = write!(
            svg,
            r#"<text x="{:.1}" y="{y:.1}" text-anchor="end" dominant-baseline="central" font-size="11">{label}</text>"#,
            config.label_margin - 4.0,
        );
    }
    // Col labels.
    for (j, label) in data.col_labels.iter().enumerate() {
        let x = config.label_margin + j as f64 * config.cell_width + config.cell_width / 2.0;
        let _ = write!(
            svg,
            r#"<text x="{x:.1}" y="{:.1}" text-anchor="middle" font-size="11">{label}</text>"#,
            config.label_margin - 4.0,
        );
    }
    // Cells.
    for cell in &cells {
        let _ = write!(
            svg,
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}"/>"#,
            cell.x, cell.y, cell.width, cell.height, cell.color,
        );
    }
    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> HeatmapData {
        HeatmapData::new(
            vec![
                vec![Some(1.0), Some(2.0), Some(3.0)],
                vec![Some(4.0), None, Some(6.0)],
                vec![Some(7.0), Some(8.0), Some(9.0)],
            ],
            vec!["R1".into(), "R2".into(), "R3".into()],
            vec!["C1".into(), "C2".into(), "C3".into()],
        )
    }

    #[test]
    fn flat_values_excludes_none() {
        let d = sample_data();
        let flat = d.flat_values();
        assert_eq!(flat.len(), 8);
    }

    #[test]
    fn min_max() {
        let d = sample_data();
        assert_eq!(d.min_value(), Some(1.0));
        assert_eq!(d.max_value(), Some(9.0));
    }

    #[test]
    fn clamp_values() {
        let mut d = sample_data();
        d.clamp(3.0, 7.0);
        assert_eq!(d.values[0][0], Some(3.0));
        assert_eq!(d.values[2][2], Some(7.0));
        assert_eq!(d.values[1][1], None); // missing preserved
    }

    #[test]
    fn compute_cells_count() {
        let d = sample_data();
        let cfg = HeatmapConfig::default();
        let cells = compute_cells(&d, &cfg);
        assert_eq!(cells.len(), 9);
    }

    #[test]
    fn missing_cell_gets_missing_color() {
        let d = sample_data();
        let cfg = HeatmapConfig::default();
        let cells = compute_cells(&d, &cfg);
        let missing = cells.iter().find(|c| c.row == 1 && c.col == 1).unwrap();
        assert_eq!(missing.color, cfg.missing_color);
    }

    #[test]
    fn palette_viridis_endpoints() {
        let p = Palette::viridis();
        let low = p.sample(0.0);
        let high = p.sample(1.0);
        assert_eq!(low, [68, 1, 84]);
        assert_eq!(high, [253, 231, 37]);
    }

    #[test]
    fn palette_diverging_midpoint() {
        let p = Palette::red_blue();
        let mid = p.sample(0.5);
        assert_eq!(mid, [255, 255, 255]);
    }

    #[test]
    fn quantize_scale() {
        let d = sample_data();
        let cfg = HeatmapConfig {
            scale: ColorScale::Quantize(3),
            ..Default::default()
        };
        let cells = compute_cells(&d, &cfg);
        // All cells should have valid hex colors.
        for c in &cells {
            assert!(c.color.starts_with('#'));
        }
    }

    #[test]
    fn quantile_scale() {
        let d = sample_data();
        let cfg = HeatmapConfig {
            scale: ColorScale::Quantile(4),
            ..Default::default()
        };
        let cells = compute_cells(&d, &cfg);
        assert_eq!(cells.len(), 9);
    }

    #[test]
    fn cluster_rows_preserves_data() {
        let mut d = sample_data();
        d.cluster_rows();
        assert_eq!(d.rows(), 3);
        assert_eq!(d.flat_values().len(), 8); // still 8 non-missing
    }

    #[test]
    fn cluster_cols_preserves_data() {
        let mut d = sample_data();
        d.cluster_cols();
        assert_eq!(d.cols(), 3);
    }

    #[test]
    fn tooltip_text() {
        let d = sample_data();
        assert_eq!(cell_tooltip(&d, 0, 0), "R1 / C1: 1.00");
        assert_eq!(cell_tooltip(&d, 1, 1), "R2 / C2: N/A");
    }

    #[test]
    fn svg_output() {
        let d = sample_data();
        let cfg = HeatmapConfig::default();
        let svg = render_svg(&d, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn empty_grid() {
        let d = HeatmapData::new(Vec::new(), Vec::new(), Vec::new());
        let cfg = HeatmapConfig::default();
        let cells = compute_cells(&d, &cfg);
        assert!(cells.is_empty());
    }
}
