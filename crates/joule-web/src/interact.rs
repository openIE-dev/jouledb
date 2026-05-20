//! Interactive chart layer: zoom, pan, tooltip, brush selection, crosshair.
//!
//! Replaces plotly interactivity, d3-zoom, d3-brush, Vega-Lite selections.
//!
//! Architecture: pure data model + SVG event attributes. The Lux/WASM runtime
//! handles the actual event dispatch. This module generates the SVG elements
//! with data-* attributes and inline event handlers that the runtime interprets.
//!
//! For server-side rendering, the interactive elements degrade gracefully —
//! the chart is still readable, just not interactive.

use std::fmt::Write;

// ── Tooltip ────────────────────────────────────────────────────────

/// Tooltip configuration.
#[derive(Debug, Clone)]
pub struct TooltipConfig {
    pub bg_color: String,
    pub text_color: String,
    pub border_color: String,
    pub font_size: f64,
    pub padding: f64,
    pub border_radius: f64,
    pub offset_x: f64,
    pub offset_y: f64,
}

impl Default for TooltipConfig {
    fn default() -> Self {
        Self {
            bg_color: "rgba(0,0,0,0.8)".into(),
            text_color: "white".into(),
            border_color: "rgba(255,255,255,0.2)".into(),
            font_size: 11.0,
            padding: 8.0,
            border_radius: 4.0,
            offset_x: 12.0,
            offset_y: -12.0,
        }
    }
}

/// Generate SVG defs for tooltip display (hidden by default).
pub fn tooltip_defs(id: &str, config: &TooltipConfig) -> String {
    format!(
        "<g id=\"{id}-tooltip\" visibility=\"hidden\" pointer-events=\"none\">\
         <rect rx=\"{:.0}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"0.5\"/>\
         <text fill=\"{}\" font-size=\"{:.0}\" dominant-baseline=\"middle\"/>\
         </g>",
        config.border_radius, config.bg_color, config.border_color,
        config.text_color, config.font_size,
    )
}

/// Wrap a chart element with tooltip hover behavior (data attributes).
pub fn with_tooltip(element_svg: &str, label: &str, x: f64, y: f64) -> String {
    // Add data attributes for the runtime to pick up
    if element_svg.contains("/>") {
        element_svg.replace(
            "/>",
            &format!(
                " data-tooltip=\"{}\" data-tx=\"{:.1}\" data-ty=\"{:.1}\" \
                 class=\"tooltip-target\"/>",
                html_escape(label), x, y
            ),
        )
    } else {
        element_svg.to_string()
    }
}

// ── Crosshair ──────────────────────────────────────────────────────

/// Crosshair overlay config.
#[derive(Debug, Clone)]
pub struct CrosshairConfig {
    pub color: String,
    pub width: f64,
    pub dash: String,
    pub show_x_label: bool,
    pub show_y_label: bool,
}

impl Default for CrosshairConfig {
    fn default() -> Self {
        Self {
            color: "#999".into(),
            width: 0.5,
            dash: "4,3".into(),
            show_x_label: true,
            show_y_label: true,
        }
    }
}

/// Generate crosshair overlay elements (hidden, positioned by runtime).
pub fn crosshair_overlay(
    id: &str,
    plot_x: f64, plot_y: f64, plot_w: f64, plot_h: f64,
    config: &CrosshairConfig,
) -> String {
    let mut s = String::new();
    let _ = write!(s,
        "<g id=\"{id}-crosshair\" visibility=\"hidden\" pointer-events=\"none\">\
         <line id=\"{id}-xhair-v\" x1=\"0\" y1=\"{plot_y:.1}\" x2=\"0\" y2=\"{:.1}\" \
               stroke=\"{}\" stroke-width=\"{}\" stroke-dasharray=\"{}\"/>\
         <line id=\"{id}-xhair-h\" x1=\"{plot_x:.1}\" y1=\"0\" x2=\"{:.1}\" y2=\"0\" \
               stroke=\"{}\" stroke-width=\"{}\" stroke-dasharray=\"{}\"/>",
        plot_y + plot_h, config.color, config.width, config.dash,
        plot_x + plot_w, config.color, config.width, config.dash,
    );

    if config.show_x_label {
        let _ = write!(s,
            "<rect id=\"{id}-xlabel-bg\" x=\"0\" y=\"{:.1}\" width=\"60\" height=\"16\" \
                   rx=\"2\" fill=\"{}\" opacity=\"0.9\"/>\
             <text id=\"{id}-xlabel\" x=\"0\" y=\"{:.1}\" text-anchor=\"middle\" \
                   font-size=\"9\" fill=\"white\"/>",
            plot_y + plot_h + 1.0, config.color, plot_y + plot_h + 12.0,
        );
    }

    if config.show_y_label {
        let _ = write!(s,
            "<rect id=\"{id}-ylabel-bg\" x=\"{:.1}\" y=\"0\" width=\"40\" height=\"14\" \
                   rx=\"2\" fill=\"{}\" opacity=\"0.9\"/>\
             <text id=\"{id}-ylabel\" x=\"{:.1}\" y=\"0\" text-anchor=\"end\" \
                   font-size=\"9\" fill=\"white\" dominant-baseline=\"middle\"/>",
            plot_x - 42.0, config.color, plot_x - 4.0,
        );
    }

    s.push_str("</g>");
    s
}

// ── Zoom & Pan ─────────────────────────────────────────────────────

/// Zoom/pan state.
#[derive(Debug, Clone)]
pub struct ViewTransform {
    pub scale_x: f64,
    pub scale_y: f64,
    pub translate_x: f64,
    pub translate_y: f64,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self { scale_x: 1.0, scale_y: 1.0, translate_x: 0.0, translate_y: 0.0 }
    }
}

impl ViewTransform {
    /// Apply zoom centered at (cx, cy).
    pub fn zoom(&mut self, factor: f64, cx: f64, cy: f64) {
        let new_sx = (self.scale_x * factor).clamp(0.1, 100.0);
        let new_sy = (self.scale_y * factor).clamp(0.1, 100.0);
        // Adjust translation to keep (cx, cy) fixed
        self.translate_x = cx - (cx - self.translate_x) * new_sx / self.scale_x;
        self.translate_y = cy - (cy - self.translate_y) * new_sy / self.scale_y;
        self.scale_x = new_sx;
        self.scale_y = new_sy;
    }

    /// Apply pan delta.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.translate_x += dx;
        self.translate_y += dy;
    }

    /// Reset to identity.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Generate SVG transform attribute.
    pub fn to_svg_transform(&self) -> String {
        format!(
            "translate({:.2},{:.2}) scale({:.4},{:.4})",
            self.translate_x, self.translate_y, self.scale_x, self.scale_y
        )
    }

    /// Transform a data coordinate to screen coordinate.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (x * self.scale_x + self.translate_x, y * self.scale_y + self.translate_y)
    }

    /// Inverse: screen coordinate to data coordinate.
    pub fn invert(&self, sx: f64, sy: f64) -> (f64, f64) {
        ((sx - self.translate_x) / self.scale_x, (sy - self.translate_y) / self.scale_y)
    }
}

/// Generate an interactive rect overlay for zoom/pan events.
pub fn zoom_pan_overlay(id: &str, x: f64, y: f64, w: f64, h: f64) -> String {
    format!(
        "<rect id=\"{id}-zoom-rect\" x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" \
         fill=\"transparent\" class=\"zoom-pan-area\" \
         data-min-zoom=\"0.1\" data-max-zoom=\"100\"/>",
    )
}

// ── Brush selection ────────────────────────────────────────────────

/// Brush selection rectangle.
#[derive(Debug, Clone)]
pub struct BrushSelection {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl BrushSelection {
    pub fn new() -> Self {
        Self { x0: 0.0, y0: 0.0, x1: 0.0, y1: 0.0 }
    }

    pub fn width(&self) -> f64 { (self.x1 - self.x0).abs() }
    pub fn height(&self) -> f64 { (self.y1 - self.y0).abs() }
    pub fn left(&self) -> f64 { self.x0.min(self.x1) }
    pub fn top(&self) -> f64 { self.y0.min(self.y1) }

    /// Check if a point is inside the selection.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.left() && x <= self.left() + self.width()
            && y >= self.top() && y <= self.top() + self.height()
    }
}

/// Brush selection mode.
#[derive(Debug, Clone, Copy)]
pub enum BrushMode {
    XY,    // 2D rectangle
    X,     // horizontal only
    Y,     // vertical only
}

/// Generate brush overlay elements.
pub fn brush_overlay(id: &str, x: f64, y: f64, w: f64, h: f64, mode: BrushMode) -> String {
    let mode_attr = match mode {
        BrushMode::XY => "xy",
        BrushMode::X => "x",
        BrushMode::Y => "y",
    };
    format!(
        "<g id=\"{id}-brush\">\
         <rect id=\"{id}-brush-bg\" x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" \
               fill=\"transparent\" class=\"brush-area\" data-brush-mode=\"{mode_attr}\"/>\
         <rect id=\"{id}-brush-sel\" x=\"0\" y=\"0\" width=\"0\" height=\"0\" \
               fill=\"#4C78A8\" fill-opacity=\"0.15\" stroke=\"#4C78A8\" stroke-width=\"1\" \
               visibility=\"hidden\" pointer-events=\"none\"/>\
         </g>",
    )
}

// ── Linked views ───────────────────────────────────────────────────

/// Linked selection: when data is selected in one chart, highlight in others.
#[derive(Debug, Clone)]
pub struct LinkedSelection {
    pub group: String,
    pub selected_indices: Vec<usize>,
}

impl LinkedSelection {
    pub fn new(group: &str) -> Self {
        Self { group: group.into(), selected_indices: vec![] }
    }

    /// Generate data attributes for linked selection.
    pub fn link_attr(&self, index: usize) -> String {
        format!("data-link-group=\"{}\" data-link-idx=\"{}\"", self.group, index)
    }
}

// ── Interactive chart wrapper ──────────────────────────────────────

/// Wrap a static SVG chart with interactive capabilities.
pub fn make_interactive(
    chart_svg: &str,
    id: &str,
    plot_x: f64,
    plot_y: f64,
    plot_w: f64,
    plot_h: f64,
    features: &InteractiveFeatures,
) -> String {
    let mut svg = chart_svg.to_string();

    // Find insertion point before </svg>
    let insert_pos = svg.rfind("</svg>").unwrap_or(svg.len());
    let mut extras = String::new();

    if features.tooltip {
        extras.push_str(&tooltip_defs(id, &TooltipConfig::default()));
    }

    if features.crosshair {
        extras.push_str(&crosshair_overlay(id, plot_x, plot_y, plot_w, plot_h, &CrosshairConfig::default()));
    }

    if features.zoom_pan {
        extras.push_str(&zoom_pan_overlay(id, plot_x, plot_y, plot_w, plot_h));
    }

    if features.brush {
        extras.push_str(&brush_overlay(id, plot_x, plot_y, plot_w, plot_h, features.brush_mode));
    }

    svg.insert_str(insert_pos, &extras);
    svg
}

/// Which interactive features to enable.
#[derive(Debug, Clone)]
pub struct InteractiveFeatures {
    pub tooltip: bool,
    pub crosshair: bool,
    pub zoom_pan: bool,
    pub brush: bool,
    pub brush_mode: BrushMode,
}

impl Default for InteractiveFeatures {
    fn default() -> Self {
        Self {
            tooltip: true,
            crosshair: true,
            zoom_pan: true,
            brush: false,
            brush_mode: BrushMode::XY,
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tooltip_defs_generated() {
        let svg = tooltip_defs("chart1", &TooltipConfig::default());
        assert!(svg.contains("chart1-tooltip"));
        assert!(svg.contains("visibility=\"hidden\""));
    }

    #[test]
    fn with_tooltip_adds_data_attrs() {
        let elem = "<circle cx=\"10\" cy=\"20\" r=\"5\"/>";
        let result = with_tooltip(elem, "Point A", 10.0, 20.0);
        assert!(result.contains("data-tooltip=\"Point A\""));
        assert!(result.contains("tooltip-target"));
    }

    #[test]
    fn crosshair_generated() {
        let svg = crosshair_overlay("c1", 50.0, 30.0, 500.0, 300.0, &CrosshairConfig::default());
        assert!(svg.contains("c1-xhair-v"));
        assert!(svg.contains("c1-xhair-h"));
    }

    #[test]
    fn view_transform_zoom() {
        let mut vt = ViewTransform::default();
        vt.zoom(2.0, 100.0, 100.0);
        assert!((vt.scale_x - 2.0).abs() < 0.01);
        let (sx, sy) = vt.apply(100.0, 100.0);
        // The zoom point should stay approximately fixed
        assert!((sx - 100.0).abs() < 1.0);
    }

    #[test]
    fn view_transform_pan() {
        let mut vt = ViewTransform::default();
        vt.pan(10.0, -5.0);
        assert!((vt.translate_x - 10.0).abs() < 0.01);
        assert!((vt.translate_y + 5.0).abs() < 0.01);
    }

    #[test]
    fn view_transform_roundtrip() {
        let mut vt = ViewTransform::default();
        vt.zoom(1.5, 200.0, 150.0);
        vt.pan(30.0, -20.0);
        let (sx, sy) = vt.apply(50.0, 75.0);
        let (dx, dy) = vt.invert(sx, sy);
        assert!((dx - 50.0).abs() < 0.01);
        assert!((dy - 75.0).abs() < 0.01);
    }

    #[test]
    fn brush_contains() {
        let mut b = BrushSelection::new();
        b.x0 = 10.0; b.y0 = 10.0; b.x1 = 50.0; b.y1 = 50.0;
        assert!(b.contains(30.0, 30.0));
        assert!(!b.contains(5.0, 30.0));
        assert!(!b.contains(30.0, 55.0));
    }

    #[test]
    fn brush_overlay_generated() {
        let svg = brush_overlay("b1", 50.0, 30.0, 500.0, 300.0, BrushMode::X);
        assert!(svg.contains("brush-area"));
        assert!(svg.contains("data-brush-mode=\"x\""));
    }

    #[test]
    fn make_interactive_adds_layers() {
        let chart = "<svg width=\"600\" height=\"400\"><rect/></svg>";
        let result = make_interactive(chart, "test", 50.0, 30.0, 500.0, 340.0, &InteractiveFeatures::default());
        assert!(result.contains("tooltip"));
        assert!(result.contains("crosshair"));
        assert!(result.contains("zoom-pan-area"));
    }

    #[test]
    fn linked_selection_attr() {
        let ls = LinkedSelection::new("group1");
        let attr = ls.link_attr(42);
        assert!(attr.contains("data-link-group=\"group1\""));
        assert!(attr.contains("data-link-idx=\"42\""));
    }
}
