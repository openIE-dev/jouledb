//! Map styling: fill/stroke/opacity rules, color ramps (sequential/diverging/qualitative),
//! data-driven styling (property-based rules), zoom-level styling, label placement rules,
//! and `StyleConfig` builder.

use core::fmt;

// ── Color ──────────────────────────────────────────────────────

/// RGBA color used throughout the styling system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: f64,
}

impl Color {
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub fn rgba(r: u8, g: u8, b: u8, a: f64) -> Self {
        Self { r, g, b, a: a.clamp(0.0, 1.0) }
    }

    pub fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    pub fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    /// Linearly interpolate between `self` and `other` at position `t` (0..1).
    pub fn lerp(&self, other: &Color, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color {
            r: (self.r as f64 + (other.r as f64 - self.r as f64) * t).round() as u8,
            g: (self.g as f64 + (other.g as f64 - self.g as f64) * t).round() as u8,
            b: (self.b as f64 + (other.b as f64 - self.b as f64) * t).round() as u8,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Return CSS `rgba(...)` string.
    pub fn to_css(&self) -> String {
        if (self.a - 1.0).abs() < 1e-6 {
            format!("rgb({},{},{})", self.r, self.g, self.b)
        } else {
            format!("rgba({},{},{},{:.2})", self.r, self.g, self.b, self.a)
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

// ── Color Ramp ────────────────────────────────────────────────

/// Kind of color ramp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RampKind {
    Sequential,
    Diverging,
    Qualitative,
}

impl fmt::Display for RampKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RampKind::Sequential => write!(f, "sequential"),
            RampKind::Diverging => write!(f, "diverging"),
            RampKind::Qualitative => write!(f, "qualitative"),
        }
    }
}

/// A multi-stop color ramp for data mapping.
#[derive(Debug, Clone)]
pub struct ColorRamp {
    pub kind: RampKind,
    pub stops: Vec<Color>,
}

impl ColorRamp {
    pub fn new(kind: RampKind, stops: Vec<Color>) -> Self {
        assert!(!stops.is_empty(), "color ramp needs at least one stop");
        Self { kind, stops }
    }

    /// Built-in sequential blue ramp.
    pub fn blues() -> Self {
        Self::new(RampKind::Sequential, vec![
            Color::rgb(239, 243, 255),
            Color::rgb(189, 215, 231),
            Color::rgb(107, 174, 214),
            Color::rgb(49, 130, 189),
            Color::rgb(8, 81, 156),
        ])
    }

    /// Built-in diverging red-blue ramp.
    pub fn red_blue() -> Self {
        Self::new(RampKind::Diverging, vec![
            Color::rgb(178, 24, 43),
            Color::rgb(239, 138, 98),
            Color::rgb(247, 247, 247),
            Color::rgb(103, 169, 207),
            Color::rgb(33, 102, 172),
        ])
    }

    /// Built-in qualitative palette (8 colors).
    pub fn qualitative8() -> Self {
        Self::new(RampKind::Qualitative, vec![
            Color::rgb(31, 119, 180),
            Color::rgb(255, 127, 14),
            Color::rgb(44, 160, 44),
            Color::rgb(214, 39, 40),
            Color::rgb(148, 103, 189),
            Color::rgb(140, 86, 75),
            Color::rgb(227, 119, 194),
            Color::rgb(127, 127, 127),
        ])
    }

    /// Sample the ramp at `t` in [0, 1].
    pub fn sample(&self, t: f64) -> Color {
        if self.stops.len() == 1 {
            return self.stops[0];
        }
        let t = t.clamp(0.0, 1.0);
        let n = self.stops.len() - 1;
        let idx = t * n as f64;
        let lo = (idx.floor() as usize).min(n.saturating_sub(1));
        let hi = (lo + 1).min(n);
        let frac = idx - lo as f64;
        self.stops[lo].lerp(&self.stops[hi], frac)
    }

    /// Return `n` evenly spaced colors.
    pub fn palette(&self, n: usize) -> Vec<Color> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![self.sample(0.5)];
        }
        (0..n).map(|i| self.sample(i as f64 / (n - 1) as f64)).collect()
    }
}

impl fmt::Display for ColorRamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ColorRamp({}, {} stops)", self.kind, self.stops.len())
    }
}

// ── Fill / Stroke ─────────────────────────────────────────────

/// Fill style for a feature.
#[derive(Debug, Clone)]
pub struct FillStyle {
    pub color: Color,
    pub opacity: f64,
}

impl Default for FillStyle {
    fn default() -> Self {
        Self { color: Color::rgb(100, 149, 237), opacity: 0.6 }
    }
}

impl fmt::Display for FillStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fill({}, {:.2})", self.color, self.opacity)
    }
}

/// Stroke style for a feature outline.
#[derive(Debug, Clone)]
pub struct StrokeStyle {
    pub color: Color,
    pub width: f64,
    pub opacity: f64,
    pub dash_array: Option<Vec<f64>>,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self { color: Color::black(), width: 1.0, opacity: 1.0, dash_array: None }
    }
}

impl fmt::Display for StrokeStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stroke({}, w={:.1})", self.color, self.width)
    }
}

// ── Data-Driven Style Rule ────────────────────────────────────

/// Comparison operator for property-based rules.
#[derive(Debug, Clone, PartialEq)]
pub enum Comparison {
    Equal(String),
    NotEqual(String),
    GreaterThan(f64),
    LessThan(f64),
    Between(f64, f64),
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Comparison::Equal(v) => write!(f, "== {v}"),
            Comparison::NotEqual(v) => write!(f, "!= {v}"),
            Comparison::GreaterThan(v) => write!(f, "> {v}"),
            Comparison::LessThan(v) => write!(f, "< {v}"),
            Comparison::Between(lo, hi) => write!(f, "in [{lo}, {hi}]"),
        }
    }
}

/// A rule that maps a feature property to a style override.
#[derive(Debug, Clone)]
pub struct DataDrivenRule {
    pub property: String,
    pub comparison: Comparison,
    pub fill: Option<FillStyle>,
    pub stroke: Option<StrokeStyle>,
}

impl DataDrivenRule {
    pub fn new(property: &str, comparison: Comparison) -> Self {
        Self { property: property.to_string(), comparison, fill: None, stroke: None }
    }

    pub fn with_fill(mut self, fill: FillStyle) -> Self {
        self.fill = Some(fill);
        self
    }

    pub fn with_stroke(mut self, stroke: StrokeStyle) -> Self {
        self.stroke = Some(stroke);
        self
    }

    /// Test whether a string value matches this rule.
    pub fn matches_str(&self, value: &str) -> bool {
        match &self.comparison {
            Comparison::Equal(v) => value == v,
            Comparison::NotEqual(v) => value != v,
            _ => false,
        }
    }

    /// Test whether a numeric value matches this rule.
    pub fn matches_f64(&self, value: f64) -> bool {
        match &self.comparison {
            Comparison::GreaterThan(v) => value > *v,
            Comparison::LessThan(v) => value < *v,
            Comparison::Between(lo, hi) => value >= *lo && value <= *hi,
            _ => false,
        }
    }
}

// ── Zoom-Level Styling ────────────────────────────────────────

/// A stop that associates a zoom level with a numeric value (for interpolation).
#[derive(Debug, Clone, Copy)]
pub struct ZoomStop {
    pub zoom: f64,
    pub value: f64,
}

/// Interpolate a value across zoom levels.
#[derive(Debug, Clone)]
pub struct ZoomInterpolation {
    pub stops: Vec<ZoomStop>,
}

impl ZoomInterpolation {
    pub fn new(stops: Vec<ZoomStop>) -> Self {
        Self { stops }
    }

    /// Evaluate the interpolated value at the given zoom level.
    pub fn evaluate(&self, zoom: f64) -> f64 {
        if self.stops.is_empty() {
            return 0.0;
        }
        if zoom <= self.stops[0].zoom {
            return self.stops[0].value;
        }
        let last = self.stops.len() - 1;
        if zoom >= self.stops[last].zoom {
            return self.stops[last].value;
        }
        for i in 0..last {
            let a = &self.stops[i];
            let b = &self.stops[i + 1];
            if zoom >= a.zoom && zoom <= b.zoom {
                let t = (zoom - a.zoom) / (b.zoom - a.zoom);
                return a.value + (b.value - a.value) * t;
            }
        }
        self.stops[last].value
    }
}

// ── Label Placement Rules ─────────────────────────────────────

/// Position preference for point labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelAnchor {
    TopLeft,
    Top,
    TopRight,
    Left,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}

impl fmt::Display for LabelAnchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            LabelAnchor::TopLeft => "top-left",
            LabelAnchor::Top => "top",
            LabelAnchor::TopRight => "top-right",
            LabelAnchor::Left => "left",
            LabelAnchor::Right => "right",
            LabelAnchor::BottomLeft => "bottom-left",
            LabelAnchor::Bottom => "bottom",
            LabelAnchor::BottomRight => "bottom-right",
        };
        write!(f, "{s}")
    }
}

/// Label placement configuration.
#[derive(Debug, Clone)]
pub struct LabelRule {
    pub property: String,
    pub font_size: f64,
    pub color: Color,
    pub anchor: LabelAnchor,
    pub max_width: f64,
    pub offset: (f64, f64),
}

impl Default for LabelRule {
    fn default() -> Self {
        Self {
            property: "name".to_string(),
            font_size: 12.0,
            color: Color::black(),
            anchor: LabelAnchor::Right,
            max_width: 120.0,
            offset: (4.0, 0.0),
        }
    }
}

// ── StyleConfig Builder ───────────────────────────────────────

/// Complete style configuration for a map layer.
#[derive(Debug, Clone)]
pub struct StyleConfig {
    pub fill: FillStyle,
    pub stroke: StrokeStyle,
    pub rules: Vec<DataDrivenRule>,
    pub label: Option<LabelRule>,
    pub zoom_width: Option<ZoomInterpolation>,
    pub zoom_opacity: Option<ZoomInterpolation>,
    pub min_zoom: f64,
    pub max_zoom: f64,
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            fill: FillStyle::default(),
            stroke: StrokeStyle::default(),
            rules: Vec::new(),
            label: None,
            zoom_width: None,
            zoom_opacity: None,
            min_zoom: 0.0,
            max_zoom: 22.0,
        }
    }
}

impl StyleConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_fill(mut self, fill: FillStyle) -> Self {
        self.fill = fill;
        self
    }

    pub fn with_stroke(mut self, stroke: StrokeStyle) -> Self {
        self.stroke = stroke;
        self
    }

    pub fn with_rule(mut self, rule: DataDrivenRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_label(mut self, label: LabelRule) -> Self {
        self.label = Some(label);
        self
    }

    pub fn with_zoom_width(mut self, interp: ZoomInterpolation) -> Self {
        self.zoom_width = Some(interp);
        self
    }

    pub fn with_zoom_opacity(mut self, interp: ZoomInterpolation) -> Self {
        self.zoom_opacity = Some(interp);
        self
    }

    pub fn with_zoom_range(mut self, min: f64, max: f64) -> Self {
        self.min_zoom = min;
        self.max_zoom = max;
        self
    }

    /// Whether the layer is visible at the given zoom level.
    pub fn visible_at(&self, zoom: f64) -> bool {
        zoom >= self.min_zoom && zoom <= self.max_zoom
    }

    /// Compute the effective stroke width at a zoom level.
    pub fn effective_width(&self, zoom: f64) -> f64 {
        match &self.zoom_width {
            Some(interp) => interp.evaluate(zoom),
            None => self.stroke.width,
        }
    }

    /// Compute the effective fill opacity at a zoom level.
    pub fn effective_opacity(&self, zoom: f64) -> f64 {
        match &self.zoom_opacity {
            Some(interp) => interp.evaluate(zoom).clamp(0.0, 1.0),
            None => self.fill.opacity,
        }
    }

    /// Find the first matching rule for a string property value.
    pub fn match_str_rule(&self, value: &str) -> Option<&DataDrivenRule> {
        self.rules.iter().find(|r| r.matches_str(value))
    }

    /// Find the first matching rule for a numeric property value.
    pub fn match_f64_rule(&self, value: f64) -> Option<&DataDrivenRule> {
        self.rules.iter().find(|r| r.matches_f64(value))
    }
}

impl fmt::Display for StyleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StyleConfig(z{}-{}, {} rules)", self.min_zoom, self.max_zoom, self.rules.len())
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_lerp_midpoint() {
        let a = Color::rgb(0, 0, 0);
        let b = Color::rgb(200, 100, 50);
        let mid = a.lerp(&b, 0.5);
        assert_eq!(mid.r, 100);
        assert_eq!(mid.g, 50);
        assert_eq!(mid.b, 25);
    }

    #[test]
    fn color_lerp_clamp() {
        let a = Color::rgb(10, 20, 30);
        let b = Color::rgb(50, 60, 70);
        assert_eq!(a.lerp(&b, -1.0), a);
        assert_eq!(a.lerp(&b, 2.0), b);
    }

    #[test]
    fn color_to_css_opaque() {
        assert_eq!(Color::rgb(255, 0, 128).to_css(), "rgb(255,0,128)");
    }

    #[test]
    fn color_to_css_alpha() {
        assert_eq!(Color::rgba(10, 20, 30, 0.5).to_css(), "rgba(10,20,30,0.50)");
    }

    #[test]
    fn color_display_hex() {
        assert_eq!(format!("{}", Color::rgb(255, 128, 0)), "#ff8000");
    }

    #[test]
    fn ramp_sample_endpoints() {
        let ramp = ColorRamp::blues();
        let first = ramp.sample(0.0);
        assert_eq!(first, ramp.stops[0]);
        let last = ramp.sample(1.0);
        assert_eq!(last, *ramp.stops.last().unwrap());
    }

    #[test]
    fn ramp_palette_count() {
        let ramp = ColorRamp::red_blue();
        assert_eq!(ramp.palette(5).len(), 5);
        assert_eq!(ramp.palette(0).len(), 0);
        assert_eq!(ramp.palette(1).len(), 1);
    }

    #[test]
    fn qualitative_palette_eight_colors() {
        let ramp = ColorRamp::qualitative8();
        assert_eq!(ramp.stops.len(), 8);
    }

    #[test]
    fn fill_default() {
        let f = FillStyle::default();
        assert!((f.opacity - 0.6).abs() < 1e-6);
    }

    #[test]
    fn stroke_default_width() {
        let s = StrokeStyle::default();
        assert!((s.width - 1.0).abs() < 1e-6);
    }

    #[test]
    fn data_driven_matches_str() {
        let rule = DataDrivenRule::new("type", Comparison::Equal("highway".to_string()));
        assert!(rule.matches_str("highway"));
        assert!(!rule.matches_str("path"));
    }

    #[test]
    fn data_driven_matches_f64() {
        let rule = DataDrivenRule::new("pop", Comparison::GreaterThan(1_000_000.0));
        assert!(rule.matches_f64(2_000_000.0));
        assert!(!rule.matches_f64(500_000.0));
    }

    #[test]
    fn data_driven_between() {
        let rule = DataDrivenRule::new("temp", Comparison::Between(20.0, 30.0));
        assert!(rule.matches_f64(25.0));
        assert!(!rule.matches_f64(35.0));
    }

    #[test]
    fn zoom_interpolation_clamp() {
        let z = ZoomInterpolation::new(vec![
            ZoomStop { zoom: 5.0, value: 1.0 },
            ZoomStop { zoom: 15.0, value: 5.0 },
        ]);
        assert!((z.evaluate(0.0) - 1.0).abs() < 1e-6);
        assert!((z.evaluate(20.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn zoom_interpolation_midpoint() {
        let z = ZoomInterpolation::new(vec![
            ZoomStop { zoom: 0.0, value: 0.0 },
            ZoomStop { zoom: 10.0, value: 100.0 },
        ]);
        assert!((z.evaluate(5.0) - 50.0).abs() < 1e-6);
    }

    #[test]
    fn style_config_visible_at() {
        let cfg = StyleConfig::new().with_zoom_range(3.0, 18.0);
        assert!(!cfg.visible_at(2.0));
        assert!(cfg.visible_at(10.0));
        assert!(!cfg.visible_at(19.0));
    }

    #[test]
    fn style_config_effective_width() {
        let z = ZoomInterpolation::new(vec![
            ZoomStop { zoom: 0.0, value: 1.0 },
            ZoomStop { zoom: 20.0, value: 10.0 },
        ]);
        let cfg = StyleConfig::new().with_zoom_width(z);
        assert!((cfg.effective_width(10.0) - 5.5).abs() < 1e-6);
    }

    #[test]
    fn style_config_match_rule() {
        let rule = DataDrivenRule::new("class", Comparison::Equal("water".to_string()))
            .with_fill(FillStyle { color: Color::rgb(0, 0, 200), opacity: 0.8 });
        let cfg = StyleConfig::new().with_rule(rule);
        assert!(cfg.match_str_rule("water").is_some());
        assert!(cfg.match_str_rule("land").is_none());
    }

    #[test]
    fn style_config_display() {
        let cfg = StyleConfig::new()
            .with_rule(DataDrivenRule::new("a", Comparison::GreaterThan(1.0)))
            .with_rule(DataDrivenRule::new("b", Comparison::LessThan(5.0)));
        assert!(format!("{cfg}").contains("2 rules"));
    }

    #[test]
    fn label_anchor_display() {
        assert_eq!(format!("{}", LabelAnchor::TopRight), "top-right");
    }
}
