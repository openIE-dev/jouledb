//! Subplot and facet composition — replaces ggplot2 facet_wrap/facet_grid,
//! matplotlib subplots, plotly make_subplots.
//!
//! Composes multiple SVG charts into a grid layout with shared axes,
//! titles, and spacing.

/// A single panel in a facet grid.
#[derive(Debug, Clone)]
pub struct Panel {
    pub svg: String,
    pub title: Option<String>,
}

/// Facet grid configuration.
#[derive(Debug, Clone)]
pub struct FacetConfig {
    pub total_width: f64,
    pub total_height: f64,
    pub ncols: usize,
    pub gap_x: f64,
    pub gap_y: f64,
    pub title: Option<String>,
    pub shared_x_label: Option<String>,
    pub shared_y_label: Option<String>,
}

impl Default for FacetConfig {
    fn default() -> Self {
        Self {
            total_width: 900.0,
            total_height: 600.0,
            ncols: 2,
            gap_x: 20.0,
            gap_y: 30.0,
            title: None,
            shared_x_label: None,
            shared_y_label: None,
        }
    }
}

/// Compose multiple SVG panels into a facet grid.
pub fn facet_grid(panels: &[Panel], config: &FacetConfig) -> String {
    let n = panels.len();
    if n == 0 { return "<svg></svg>".to_string(); }

    let ncols = config.ncols.max(1);
    let nrows = (n + ncols - 1) / ncols;
    let title_h = if config.title.is_some() { 30.0 } else { 0.0 };
    let label_h = if config.shared_x_label.is_some() { 25.0 } else { 0.0 };
    let label_w = if config.shared_y_label.is_some() { 25.0 } else { 0.0 };

    let panel_w = (config.total_width - label_w - (ncols - 1) as f64 * config.gap_x) / ncols as f64;
    let panel_h = (config.total_height - title_h - label_h - (nrows - 1) as f64 * config.gap_y) / nrows as f64;

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">",
        config.total_width, config.total_height
    );
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    // Main title
    if let Some(t) = &config.title {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"16\" font-weight=\"bold\">{t}</text>",
            config.total_width / 2.0
        ));
    }

    // Panels
    for (i, panel) in panels.iter().enumerate() {
        let col = i % ncols;
        let row = i / ncols;
        let x = label_w + col as f64 * (panel_w + config.gap_x);
        let y = title_h + row as f64 * (panel_h + config.gap_y);

        // Panel title
        if let Some(pt) = &panel.title {
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"11\" font-weight=\"bold\">{pt}</text>",
                x + panel_w / 2.0, y - 3.0
            ));
        }

        // Embed panel SVG as a nested group with viewBox
        svg.push_str(&format!(
            "<g transform=\"translate({x:.1},{y:.1})\"><svg width=\"{panel_w:.1}\" height=\"{panel_h:.1}\" viewBox=\"0 0 600 400\" preserveAspectRatio=\"xMidYMid meet\">"
        ));
        // Strip outer <svg> tags from panel content
        let inner = strip_svg_wrapper(&panel.svg);
        svg.push_str(&inner);
        svg.push_str("</svg></g>");
    }

    // Shared X label
    if let Some(xl) = &config.shared_x_label {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"12\">{xl}</text>",
            config.total_width / 2.0, config.total_height - 5.0
        ));
    }

    // Shared Y label (rotated)
    if let Some(yl) = &config.shared_y_label {
        svg.push_str(&format!(
            "<text x=\"12\" y=\"{}\" text-anchor=\"middle\" font-size=\"12\" transform=\"rotate(-90,12,{})\">{yl}</text>",
            config.total_height / 2.0, config.total_height / 2.0
        ));
    }

    svg.push_str("</svg>");
    svg
}

/// Strip <svg ...> and </svg> wrapper from an SVG string.
fn strip_svg_wrapper(svg: &str) -> &str {
    let start = svg.find('>').map(|i| i + 1).unwrap_or(0);
    let end = svg.rfind("</svg>").unwrap_or(svg.len());
    &svg[start..end]
}

/// Convenience: wrap a list of (label, data_fn) into panels.
/// `render_fn` takes a label and returns an SVG string for that panel.
pub fn facet_wrap<F>(labels: &[&str], ncols: usize, render_fn: F, config: &FacetConfig) -> String
where
    F: Fn(&str) -> String,
{
    let panels: Vec<Panel> = labels.iter().map(|&label| Panel {
        svg: render_fn(label),
        title: Some(label.to_string()),
    }).collect();
    facet_grid(&panels, &FacetConfig { ncols, ..config.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_svg() -> String {
        "<svg><rect x=\"0\" y=\"0\" width=\"100\" height=\"100\" fill=\"blue\"/></svg>".to_string()
    }

    #[test]
    fn facet_basic() {
        let panels = vec![
            Panel { svg: dummy_svg(), title: Some("A".into()) },
            Panel { svg: dummy_svg(), title: Some("B".into()) },
            Panel { svg: dummy_svg(), title: Some("C".into()) },
            Panel { svg: dummy_svg(), title: Some("D".into()) },
        ];
        let svg = facet_grid(&panels, &FacetConfig::default());
        assert!(svg.contains("<svg"));
        assert!(svg.contains("A"));
        assert!(svg.contains("D"));
    }

    #[test]
    fn facet_with_labels() {
        let panels = vec![Panel { svg: dummy_svg(), title: None }; 6];
        let config = FacetConfig {
            ncols: 3,
            title: Some("My Grid".into()),
            shared_x_label: Some("X".into()),
            shared_y_label: Some("Y".into()),
            ..Default::default()
        };
        let svg = facet_grid(&panels, &config);
        assert!(svg.contains("My Grid"));
        assert!(svg.contains("rotate(-90"));
    }

    #[test]
    fn facet_wrap_fn() {
        let svg = facet_wrap(
            &["Group A", "Group B", "Group C"],
            2,
            |label| format!("<svg><text>{label}</text></svg>"),
            &FacetConfig::default(),
        );
        assert!(svg.contains("Group A"));
        assert!(svg.contains("Group C"));
    }

    #[test]
    fn strip_wrapper() {
        let inner = strip_svg_wrapper("<svg width=\"100\"><rect/></svg>");
        assert_eq!(inner, "<rect/>");
    }
}
