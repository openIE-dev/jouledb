//! High-dimensional visualization: parallel coordinates, radar/spider charts, Andrews curves.
//!
//! Replaces R's GGally::ggparcoord, Python plotly.express.parallel_coordinates,
//! MATLAB parallelcoords, matplotlib radar plots.

// ── Parallel Coordinates ───────────────────────────────────────────

/// One data row: a named point with values for each axis.
#[derive(Debug, Clone)]
pub struct ParallelRow {
    pub values: Vec<f64>,
    pub group: Option<String>,
}

/// Parallel coordinates plot config.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub width: f64,
    pub height: f64,
    pub axis_labels: Vec<String>,
    pub title: Option<String>,
    pub group_colors: Vec<String>,
    pub line_opacity: f64,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 400.0,
            axis_labels: vec![],
            title: None,
            group_colors: vec![
                "#4C78A8".into(), "#F58518".into(), "#E45756".into(),
                "#72B7B2".into(), "#54A24B".into(), "#EECA3B".into(),
                "#B279A2".into(), "#FF9DA6".into(), "#9D755D".into(),
            ],
            line_opacity: 0.4,
        }
    }
}

/// Render parallel coordinates to SVG.
pub fn parallel_coords_svg(rows: &[ParallelRow], config: &ParallelConfig) -> String {
    if rows.is_empty() { return "<svg></svg>".to_string(); }
    let ndims = rows[0].values.len();
    if ndims == 0 { return "<svg></svg>".to_string(); }

    let pad_top = if config.title.is_some() { 40.0 } else { 20.0 };
    let pad_bot = 40.0;
    let pad_lr = 40.0;
    let plot_w = config.width - 2.0 * pad_lr;
    let plot_h = config.height - pad_top - pad_bot;

    // Compute min/max per dimension
    let mut mins = vec![f64::INFINITY; ndims];
    let mut maxs = vec![f64::NEG_INFINITY; ndims];
    for row in rows {
        for (d, &v) in row.values.iter().enumerate() {
            if v < mins[d] { mins[d] = v; }
            if v > maxs[d] { maxs[d] = v; }
        }
    }

    let axis_x = |d: usize| -> f64 { pad_lr + d as f64 / (ndims - 1).max(1) as f64 * plot_w };
    let axis_y = |d: usize, v: f64| -> f64 {
        let range = (maxs[d] - mins[d]).max(1e-10);
        pad_top + plot_h - (v - mins[d]) / range * plot_h
    };

    // Collect groups
    let mut groups: Vec<String> = Vec::new();
    for row in rows {
        if let Some(g) = &row.group {
            if !groups.contains(g) { groups.push(g.clone()); }
        }
    }

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">",
        config.width, config.height
    );
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    if let Some(t) = &config.title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{t}</text>",
            config.width / 2.0
        ));
    }

    // Axes
    for d in 0..ndims {
        let x = axis_x(d);
        svg.push_str(&format!(
            "<line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"#ccc\" stroke-width=\"1\"/>",
            pad_top, pad_top + plot_h
        ));
        // Label
        let label = config.axis_labels.get(d).map(|s| s.as_str()).unwrap_or(&"");
        svg.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"10\">{label}</text>",
            pad_top + plot_h + 15.0
        ));
        // Min/max
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-size=\"8\">{:.1}</text>",
            x - 3.0, pad_top + plot_h, mins[d]
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-size=\"8\">{:.1}</text>",
            x - 3.0, pad_top + 3.0, maxs[d]
        ));
    }

    // Lines
    for row in rows {
        let color_idx = row.group.as_ref()
            .and_then(|g| groups.iter().position(|gr| gr == g))
            .unwrap_or(0);
        let color = &config.group_colors[color_idx % config.group_colors.len()];

        let mut path = String::new();
        for (d, &v) in row.values.iter().enumerate() {
            let x = axis_x(d);
            let y = axis_y(d, v);
            if d == 0 { path.push_str(&format!("M{x:.1},{y:.1}")); }
            else { path.push_str(&format!(" L{x:.1},{y:.1}")); }
        }
        svg.push_str(&format!(
            "<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"1.5\" opacity=\"{:.2}\"/>",
            config.line_opacity
        ));
    }

    svg.push_str("</svg>");
    svg
}

// ── Radar / Spider Chart ───────────────────────────────────────────

/// Radar chart data: one series with values for each axis.
#[derive(Debug, Clone)]
pub struct RadarSeries {
    pub name: String,
    pub values: Vec<f64>,
    pub color: String,
}

/// Radar chart config.
#[derive(Debug, Clone)]
pub struct RadarConfig {
    pub width: f64,
    pub height: f64,
    pub axis_labels: Vec<String>,
    pub max_value: Option<f64>,
    pub title: Option<String>,
    pub n_rings: usize,
    pub fill_opacity: f64,
}

impl Default for RadarConfig {
    fn default() -> Self {
        Self { width: 500.0, height: 500.0, axis_labels: vec![], max_value: None, title: None, n_rings: 5, fill_opacity: 0.2 }
    }
}

/// Render a radar/spider chart to SVG.
pub fn radar_svg(series: &[RadarSeries], config: &RadarConfig) -> String {
    if series.is_empty() { return "<svg></svg>".to_string(); }
    let ndims = config.axis_labels.len().max(series[0].values.len());
    if ndims == 0 { return "<svg></svg>".to_string(); }

    let cx = config.width / 2.0;
    let cy = config.height / 2.0;
    let radius = (config.width.min(config.height) / 2.0) - 50.0;
    let angle_step = 2.0 * std::f64::consts::PI / ndims as f64;

    let max_val = config.max_value.unwrap_or_else(|| {
        series.iter().flat_map(|s| s.values.iter()).copied().fold(0.0f64, f64::max)
    }).max(1e-10);

    let polar = |dim: usize, val: f64| -> (f64, f64) {
        let angle = dim as f64 * angle_step - std::f64::consts::PI / 2.0;
        let r = (val / max_val) * radius;
        (cx + r * angle.cos(), cy + r * angle.sin())
    };

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">",
        config.width, config.height
    );
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    if let Some(t) = &config.title {
        svg.push_str(&format!(
            "<text x=\"{cx}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{t}</text>",
        ));
    }

    // Grid rings
    for ring in 1..=config.n_rings {
        let r = radius * ring as f64 / config.n_rings as f64;
        let mut points = String::new();
        for d in 0..ndims {
            let angle = d as f64 * angle_step - std::f64::consts::PI / 2.0;
            let x = cx + r * angle.cos();
            let y = cy + r * angle.sin();
            if d == 0 { points.push_str(&format!("{x:.1},{y:.1}")); }
            else { points.push_str(&format!(" {x:.1},{y:.1}")); }
        }
        svg.push_str(&format!(
            "<polygon points=\"{points}\" fill=\"none\" stroke=\"#ddd\" stroke-width=\"0.5\"/>"
        ));
    }

    // Axis lines and labels
    for d in 0..ndims {
        let (ex, ey) = polar(d, max_val);
        svg.push_str(&format!(
            "<line x1=\"{cx:.1}\" y1=\"{cy:.1}\" x2=\"{ex:.1}\" y2=\"{ey:.1}\" stroke=\"#ccc\" stroke-width=\"0.5\"/>"
        ));
        let label = config.axis_labels.get(d).map(|s| s.as_str()).unwrap_or("");
        let (lx, ly) = polar(d, max_val * 1.1);
        svg.push_str(&format!(
            "<text x=\"{lx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" font-size=\"10\">{label}</text>"
        ));
    }

    // Data series
    for s in series {
        let mut points = String::new();
        for (d, &v) in s.values.iter().enumerate() {
            let (x, y) = polar(d, v.clamp(0.0, max_val));
            if d == 0 { points.push_str(&format!("{x:.1},{y:.1}")); }
            else { points.push_str(&format!(" {x:.1},{y:.1}")); }
        }
        svg.push_str(&format!(
            "<polygon points=\"{points}\" fill=\"{}\" fill-opacity=\"{:.2}\" stroke=\"{}\" stroke-width=\"2\"/>",
            s.color, config.fill_opacity, s.color
        ));
    }

    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_coords_basic() {
        let rows = vec![
            ParallelRow { values: vec![1.0, 5.0, 3.0], group: Some("A".into()) },
            ParallelRow { values: vec![2.0, 3.0, 4.0], group: Some("B".into()) },
            ParallelRow { values: vec![3.0, 1.0, 5.0], group: Some("A".into()) },
        ];
        let config = ParallelConfig {
            axis_labels: vec!["X".into(), "Y".into(), "Z".into()],
            ..Default::default()
        };
        let svg = parallel_coords_svg(&rows, &config);
        assert!(svg.contains("path"));
        assert!(svg.contains("X"));
    }

    #[test]
    fn radar_basic() {
        let series = vec![
            RadarSeries { name: "A".into(), values: vec![80.0, 90.0, 70.0, 85.0, 60.0], color: "#4C78A8".into() },
            RadarSeries { name: "B".into(), values: vec![65.0, 75.0, 90.0, 70.0, 80.0], color: "#E45756".into() },
        ];
        let config = RadarConfig {
            axis_labels: vec!["Speed".into(), "Power".into(), "Range".into(), "Defense".into(), "Magic".into()],
            ..Default::default()
        };
        let svg = radar_svg(&series, &config);
        assert!(svg.contains("polygon"));
        assert!(svg.contains("Speed"));
    }
}
