//! Spacing scale system: base-unit multipliers, t-shirt sizes, semantic spacing,
//! margin/padding shorthand, responsive spacing, and gap utilities.
//!
//! All values are computed from a configurable base unit (default 4px) and
//! emitted as CSS strings — no runtime overhead.

use std::fmt;

// ── Base Unit ───────────────────────────────────────────────────

/// A spacing scale built from a base unit (in px).
#[derive(Debug, Clone)]
pub struct SpacingScale {
    /// The atomic unit in pixels (default: 4).
    pub base: f64,
}

impl Default for SpacingScale {
    fn default() -> Self {
        Self { base: 4.0 }
    }
}

impl SpacingScale {
    pub fn new(base: f64) -> Self {
        Self { base: base.max(1.0) }
    }

    /// Multiply the base unit by `n` (e.g., `scale.mult(2)` = 8px with base=4).
    pub fn mult(&self, n: f64) -> f64 {
        self.base * n
    }

    /// Get spacing for a t-shirt size.
    pub fn size(&self, size: TShirtSize) -> f64 {
        self.mult(size.multiplier())
    }

    /// Get spacing for a semantic level.
    pub fn semantic(&self, level: SemanticSpacing) -> f64 {
        self.mult(level.multiplier())
    }

    /// CSS px string for a multiplier.
    pub fn px(&self, n: f64) -> String {
        let v = self.mult(n);
        if v == 0.0 {
            "0".to_owned()
        } else {
            format!("{v}px")
        }
    }

    /// CSS rem string (assuming 16px root).
    pub fn rem(&self, n: f64) -> String {
        let v = self.mult(n) / 16.0;
        if v == 0.0 {
            "0".to_owned()
        } else {
            format!("{v}rem")
        }
    }
}

// ── T-Shirt Sizes ───────────────────────────────────────────────

/// Standard t-shirt spacing sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TShirtSize {
    Xs,
    Sm,
    Md,
    Lg,
    Xl,
    Xxl,
}

impl TShirtSize {
    /// Multiplier of the base unit for this size.
    pub fn multiplier(self) -> f64 {
        match self {
            TShirtSize::Xs => 1.0,   // 4px
            TShirtSize::Sm => 2.0,   // 8px
            TShirtSize::Md => 4.0,   // 16px
            TShirtSize::Lg => 6.0,   // 24px
            TShirtSize::Xl => 8.0,   // 32px
            TShirtSize::Xxl => 12.0, // 48px
        }
    }

    /// All sizes in order.
    pub fn all() -> &'static [TShirtSize] {
        &[
            TShirtSize::Xs,
            TShirtSize::Sm,
            TShirtSize::Md,
            TShirtSize::Lg,
            TShirtSize::Xl,
            TShirtSize::Xxl,
        ]
    }
}

impl fmt::Display for TShirtSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TShirtSize::Xs => write!(f, "xs"),
            TShirtSize::Sm => write!(f, "sm"),
            TShirtSize::Md => write!(f, "md"),
            TShirtSize::Lg => write!(f, "lg"),
            TShirtSize::Xl => write!(f, "xl"),
            TShirtSize::Xxl => write!(f, "2xl"),
        }
    }
}

// ── Semantic Spacing ────────────────────────────────────────────

/// Semantic spacing levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticSpacing {
    None,
    Tight,
    Normal,
    Loose,
}

impl SemanticSpacing {
    pub fn multiplier(self) -> f64 {
        match self {
            SemanticSpacing::None => 0.0,
            SemanticSpacing::Tight => 1.0,
            SemanticSpacing::Normal => 4.0,
            SemanticSpacing::Loose => 8.0,
        }
    }
}

impl fmt::Display for SemanticSpacing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemanticSpacing::None => write!(f, "none"),
            SemanticSpacing::Tight => write!(f, "tight"),
            SemanticSpacing::Normal => write!(f, "normal"),
            SemanticSpacing::Loose => write!(f, "loose"),
        }
    }
}

// ── Margin & Padding Shorthand ──────────────────────────────────

/// Four-side spacing (top, right, bottom, left).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxSpacing {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl BoxSpacing {
    /// All sides equal.
    pub fn all(v: f64) -> Self {
        Self { top: v, right: v, bottom: v, left: v }
    }

    /// Vertical and horizontal.
    pub fn symmetric(vertical: f64, horizontal: f64) -> Self {
        Self {
            top: vertical,
            right: horizontal,
            bottom: vertical,
            left: horizontal,
        }
    }

    /// All four sides.
    pub fn new(top: f64, right: f64, bottom: f64, left: f64) -> Self {
        Self { top, right, bottom, left }
    }

    /// Zero spacing.
    pub fn zero() -> Self {
        Self::all(0.0)
    }

    /// Emit CSS shorthand in px.
    pub fn to_css_px(&self) -> String {
        if self.top == self.right && self.right == self.bottom && self.bottom == self.left {
            if self.top == 0.0 {
                "0".to_owned()
            } else {
                format!("{}px", self.top)
            }
        } else if self.top == self.bottom && self.left == self.right {
            format!("{}px {}px", self.top, self.right)
        } else if self.left == self.right {
            format!("{}px {}px {}px", self.top, self.right, self.bottom)
        } else {
            format!(
                "{}px {}px {}px {}px",
                self.top, self.right, self.bottom, self.left
            )
        }
    }

    /// Emit as a CSS `margin` declaration.
    pub fn to_margin_css(&self) -> String {
        format!("margin: {};", self.to_css_px())
    }

    /// Emit as a CSS `padding` declaration.
    pub fn to_padding_css(&self) -> String {
        format!("padding: {};", self.to_css_px())
    }
}

// ── Responsive Spacing ──────────────────────────────────────────

/// A spacing value that changes at breakpoints.
#[derive(Debug, Clone)]
pub struct ResponsiveSpacing {
    pub entries: Vec<(f64, f64)>, // (min_width_px, spacing_px)
}

impl ResponsiveSpacing {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a breakpoint: at `min_width` viewport, use `spacing` px.
    pub fn at(mut self, min_width: f64, spacing: f64) -> Self {
        self.entries.push((min_width, spacing));
        self.entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self
    }

    /// Resolve for a given viewport width.
    pub fn resolve(&self, viewport_width: f64) -> f64 {
        let mut result = 0.0;
        for &(min_w, spacing) in &self.entries {
            if viewport_width >= min_w {
                result = spacing;
            } else {
                break;
            }
        }
        result
    }

    /// Generate CSS custom property with media queries.
    pub fn to_css(&self, property_name: &str) -> String {
        let mut css = String::new();
        for (i, &(min_w, spacing)) in self.entries.iter().enumerate() {
            if i == 0 {
                css.push_str(&format!(
                    ":root {{ --{property_name}: {spacing}px; }}\n"
                ));
            } else {
                css.push_str(&format!(
                    "@media (min-width: {min_w}px) {{ :root {{ --{property_name}: {spacing}px; }} }}\n"
                ));
            }
        }
        css
    }
}

impl Default for ResponsiveSpacing {
    fn default() -> Self {
        Self::new()
    }
}

// ── Gap Utilities ───────────────────────────────────────────────

/// CSS gap shorthand (row-gap, column-gap).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gap {
    pub row: f64,
    pub column: f64,
}

impl Gap {
    pub fn uniform(v: f64) -> Self {
        Self { row: v, column: v }
    }

    pub fn new(row: f64, column: f64) -> Self {
        Self { row, column }
    }

    pub fn from_scale(scale: &SpacingScale, row_mult: f64, col_mult: f64) -> Self {
        Self {
            row: scale.mult(row_mult),
            column: scale.mult(col_mult),
        }
    }

    pub fn to_css(&self) -> String {
        if (self.row - self.column).abs() < f64::EPSILON {
            if self.row == 0.0 {
                "gap: 0;".to_owned()
            } else {
                format!("gap: {}px;", self.row)
            }
        } else {
            format!("gap: {}px {}px;", self.row, self.column)
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_base() {
        let s = SpacingScale::default();
        assert_eq!(s.base, 4.0);
    }

    #[test]
    fn test_multiplier() {
        let s = SpacingScale::new(8.0);
        assert_eq!(s.mult(2.0), 16.0);
        assert_eq!(s.mult(0.5), 4.0);
    }

    #[test]
    fn test_tshirt_sizes() {
        let s = SpacingScale::default(); // base = 4
        assert_eq!(s.size(TShirtSize::Xs), 4.0);
        assert_eq!(s.size(TShirtSize::Sm), 8.0);
        assert_eq!(s.size(TShirtSize::Md), 16.0);
        assert_eq!(s.size(TShirtSize::Lg), 24.0);
        assert_eq!(s.size(TShirtSize::Xl), 32.0);
        assert_eq!(s.size(TShirtSize::Xxl), 48.0);
    }

    #[test]
    fn test_semantic_spacing() {
        let s = SpacingScale::default();
        assert_eq!(s.semantic(SemanticSpacing::None), 0.0);
        assert_eq!(s.semantic(SemanticSpacing::Tight), 4.0);
        assert_eq!(s.semantic(SemanticSpacing::Normal), 16.0);
        assert_eq!(s.semantic(SemanticSpacing::Loose), 32.0);
    }

    #[test]
    fn test_px_and_rem() {
        let s = SpacingScale::default();
        assert_eq!(s.px(4.0), "16px");
        assert_eq!(s.rem(4.0), "1rem");
        assert_eq!(s.px(0.0), "0");
    }

    #[test]
    fn test_box_spacing_shorthand_all() {
        let b = BoxSpacing::all(8.0);
        assert_eq!(b.to_css_px(), "8px");
    }

    #[test]
    fn test_box_spacing_shorthand_symmetric() {
        let b = BoxSpacing::symmetric(8.0, 16.0);
        assert_eq!(b.to_css_px(), "8px 16px");
    }

    #[test]
    fn test_box_spacing_four_values() {
        let b = BoxSpacing::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(b.to_css_px(), "1px 2px 3px 4px");
    }

    #[test]
    fn test_margin_padding_css() {
        let b = BoxSpacing::all(12.0);
        assert_eq!(b.to_margin_css(), "margin: 12px;");
        assert_eq!(b.to_padding_css(), "padding: 12px;");
    }

    #[test]
    fn test_responsive_spacing_resolve() {
        let rs = ResponsiveSpacing::new()
            .at(0.0, 8.0)
            .at(768.0, 16.0)
            .at(1200.0, 32.0);
        assert_eq!(rs.resolve(320.0), 8.0);
        assert_eq!(rs.resolve(800.0), 16.0);
        assert_eq!(rs.resolve(1400.0), 32.0);
    }

    #[test]
    fn test_responsive_spacing_css() {
        let rs = ResponsiveSpacing::new()
            .at(0.0, 8.0)
            .at(768.0, 16.0);
        let css = rs.to_css("spacing-md");
        assert!(css.contains("--spacing-md: 8px"));
        assert!(css.contains("@media (min-width: 768px)"));
    }

    #[test]
    fn test_gap_uniform() {
        let g = Gap::uniform(16.0);
        assert_eq!(g.to_css(), "gap: 16px;");
    }

    #[test]
    fn test_gap_different() {
        let g = Gap::new(8.0, 16.0);
        assert_eq!(g.to_css(), "gap: 8px 16px;");
    }

    #[test]
    fn test_gap_from_scale() {
        let scale = SpacingScale::default();
        let g = Gap::from_scale(&scale, 2.0, 4.0);
        assert_eq!(g.row, 8.0);
        assert_eq!(g.column, 16.0);
    }

    #[test]
    fn test_tshirt_display() {
        assert_eq!(TShirtSize::Xxl.to_string(), "2xl");
        assert_eq!(TShirtSize::Xs.to_string(), "xs");
    }

    #[test]
    fn test_all_tshirt_sizes() {
        assert_eq!(TShirtSize::all().len(), 6);
    }
}
