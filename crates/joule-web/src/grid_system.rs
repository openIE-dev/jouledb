//! Grid layout system: 12-column grid, column span, offset, push/pull,
//! nested grids, gutter size, responsive column counts, auto-fit/auto-fill,
//! named areas, and alignment.
//!
//! Pure CSS generation — no browser dependency.

use std::fmt;

// ── Alignment ───────────────────────────────────────────────────

/// Alignment for grid items and tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridAlign {
    Start,
    Center,
    End,
    Stretch,
}

impl fmt::Display for GridAlign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridAlign::Start => write!(f, "start"),
            GridAlign::Center => write!(f, "center"),
            GridAlign::End => write!(f, "end"),
            GridAlign::Stretch => write!(f, "stretch"),
        }
    }
}

// ── Auto Sizing Mode ────────────────────────────────────────────

/// CSS Grid auto-sizing keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoSizing {
    AutoFit,
    AutoFill,
}

impl fmt::Display for AutoSizing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutoSizing::AutoFit => write!(f, "auto-fit"),
            AutoSizing::AutoFill => write!(f, "auto-fill"),
        }
    }
}

// ── Grid Configuration ──────────────────────────────────────────

/// A CSS Grid container configuration.
#[derive(Debug, Clone)]
pub struct GridContainer {
    /// Total number of columns (default: 12).
    pub columns: u32,
    /// Gutter size in pixels.
    pub gutter: f64,
    /// Row gap in pixels (defaults to gutter if None).
    pub row_gap: Option<f64>,
    /// Justify items (column axis).
    pub justify_items: GridAlign,
    /// Align items (row axis).
    pub align_items: GridAlign,
    /// Max container width (None = fluid).
    pub max_width: Option<f64>,
}

impl Default for GridContainer {
    fn default() -> Self {
        Self {
            columns: 12,
            gutter: 16.0,
            row_gap: None,
            justify_items: GridAlign::Stretch,
            align_items: GridAlign::Stretch,
            max_width: None,
        }
    }
}

impl GridContainer {
    pub fn new(columns: u32) -> Self {
        Self {
            columns: columns.max(1),
            ..Default::default()
        }
    }

    pub fn with_gutter(mut self, gutter: f64) -> Self {
        self.gutter = gutter;
        self
    }

    pub fn with_row_gap(mut self, gap: f64) -> Self {
        self.row_gap = Some(gap);
        self
    }

    pub fn with_max_width(mut self, max_width: f64) -> Self {
        self.max_width = Some(max_width);
        self
    }

    pub fn with_justify(mut self, align: GridAlign) -> Self {
        self.justify_items = align;
        self
    }

    pub fn with_align(mut self, align: GridAlign) -> Self {
        self.align_items = align;
        self
    }

    /// Generate CSS for the grid container.
    pub fn to_css(&self, selector: &str) -> String {
        let row_gap = self.row_gap.unwrap_or(self.gutter);
        let mut css = format!(
            "{selector} {{\n  display: grid;\n  grid-template-columns: repeat({}, 1fr);\n  gap: {row_gap}px {}px;\n  justify-items: {};\n  align-items: {};\n",
            self.columns, self.gutter, self.justify_items, self.align_items
        );
        if let Some(mw) = self.max_width {
            css.push_str(&format!("  max-width: {mw}px;\n  margin-inline: auto;\n"));
        }
        css.push_str("}\n");
        css
    }

    /// Percentage width of `span` columns (including gutters).
    pub fn column_width_percent(&self, span: u32) -> f64 {
        let s = span.min(self.columns).max(1);
        (s as f64 / self.columns as f64) * 100.0
    }
}

// ── Grid Item ───────────────────────────────────────────────────

/// A single grid item placement.
#[derive(Debug, Clone)]
pub struct GridItem {
    /// Column span (1..=columns).
    pub span: u32,
    /// Column offset (push start by N columns).
    pub offset: u32,
    /// Push (move right by N columns visually, via order/grid-column).
    pub push: Option<u32>,
    /// Pull (move left by N columns visually).
    pub pull: Option<u32>,
    /// Row span.
    pub row_span: u32,
    /// Justify self override.
    pub justify_self: Option<GridAlign>,
    /// Align self override.
    pub align_self: Option<GridAlign>,
}

impl GridItem {
    pub fn span(columns: u32) -> Self {
        Self {
            span: columns.max(1),
            offset: 0,
            push: None,
            pull: None,
            row_span: 1,
            justify_self: None,
            align_self: None,
        }
    }

    pub fn with_offset(mut self, offset: u32) -> Self {
        self.offset = offset;
        self
    }

    pub fn with_push(mut self, push: u32) -> Self {
        self.push = Some(push);
        self
    }

    pub fn with_pull(mut self, pull: u32) -> Self {
        self.pull = Some(pull);
        self
    }

    pub fn with_row_span(mut self, row_span: u32) -> Self {
        self.row_span = row_span.max(1);
        self
    }

    pub fn with_justify(mut self, align: GridAlign) -> Self {
        self.justify_self = Some(align);
        self
    }

    pub fn with_align(mut self, align: GridAlign) -> Self {
        self.align_self = Some(align);
        self
    }

    /// Generate CSS for this item.
    pub fn to_css(&self, selector: &str) -> String {
        let col_start = self.offset + 1;
        // Apply push/pull as offset adjustment.
        let adjusted_start = if let Some(push) = self.push {
            col_start + push
        } else if let Some(pull) = self.pull {
            col_start.saturating_sub(pull)
        } else {
            col_start
        };

        let mut css = format!(
            "{selector} {{\n  grid-column: {} / span {};\n",
            adjusted_start, self.span
        );
        if self.row_span > 1 {
            css.push_str(&format!("  grid-row: span {};\n", self.row_span));
        }
        if let Some(js) = self.justify_self {
            css.push_str(&format!("  justify-self: {js};\n"));
        }
        if let Some(als) = self.align_self {
            css.push_str(&format!("  align-self: {als};\n"));
        }
        css.push_str("}\n");
        css
    }
}

// ── Auto-Sizing Grid ────────────────────────────────────────────

/// A grid with auto-fit or auto-fill columns.
#[derive(Debug, Clone)]
pub struct AutoGrid {
    pub mode: AutoSizing,
    /// Minimum column width in px.
    pub min_column_width: f64,
    /// Maximum column width (None = 1fr).
    pub max_column_width: Option<f64>,
    pub gap: f64,
}

impl AutoGrid {
    pub fn new(mode: AutoSizing, min_column_width: f64) -> Self {
        Self {
            mode,
            min_column_width,
            max_column_width: None,
            gap: 16.0,
        }
    }

    pub fn with_gap(mut self, gap: f64) -> Self {
        self.gap = gap;
        self
    }

    pub fn with_max_width(mut self, max: f64) -> Self {
        self.max_column_width = Some(max);
        self
    }

    pub fn to_css(&self, selector: &str) -> String {
        let max = self
            .max_column_width
            .map(|m| format!("{m}px"))
            .unwrap_or_else(|| "1fr".to_owned());

        format!(
            "{selector} {{\n  display: grid;\n  grid-template-columns: repeat({}, minmax({}px, {max}));\n  gap: {}px;\n}}\n",
            self.mode, self.min_column_width, self.gap
        )
    }
}

// ── Named Areas ─────────────────────────────────────────────────

/// A grid with named template areas.
#[derive(Debug, Clone)]
pub struct NamedAreaGrid {
    pub rows: Vec<Vec<String>>,
    pub column_sizes: Vec<String>,
    pub row_sizes: Vec<String>,
    pub gap: f64,
}

impl NamedAreaGrid {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            column_sizes: Vec::new(),
            row_sizes: Vec::new(),
            gap: 16.0,
        }
    }

    pub fn add_row(mut self, areas: Vec<impl Into<String>>) -> Self {
        self.rows.push(areas.into_iter().map(|a| a.into()).collect());
        self
    }

    pub fn with_column_sizes(mut self, sizes: Vec<impl Into<String>>) -> Self {
        self.column_sizes = sizes.into_iter().map(|s| s.into()).collect();
        self
    }

    pub fn with_row_sizes(mut self, sizes: Vec<impl Into<String>>) -> Self {
        self.row_sizes = sizes.into_iter().map(|s| s.into()).collect();
        self
    }

    pub fn with_gap(mut self, gap: f64) -> Self {
        self.gap = gap;
        self
    }

    pub fn to_css(&self, selector: &str) -> String {
        let areas: Vec<String> = self
            .rows
            .iter()
            .map(|row| format!("\"{}\"", row.join(" ")))
            .collect();

        let mut css = format!("{selector} {{\n  display: grid;\n");
        css.push_str(&format!(
            "  grid-template-areas:\n    {};\n",
            areas.join("\n    ")
        ));
        if !self.column_sizes.is_empty() {
            css.push_str(&format!(
                "  grid-template-columns: {};\n",
                self.column_sizes.join(" ")
            ));
        }
        if !self.row_sizes.is_empty() {
            css.push_str(&format!(
                "  grid-template-rows: {};\n",
                self.row_sizes.join(" ")
            ));
        }
        css.push_str(&format!("  gap: {}px;\n", self.gap));
        css.push_str("}\n");
        css
    }

    /// Generate a CSS rule to place an item in a named area.
    pub fn area_css(selector: &str, area_name: &str) -> String {
        format!("{selector} {{ grid-area: {area_name}; }}\n")
    }
}

impl Default for NamedAreaGrid {
    fn default() -> Self {
        Self::new()
    }
}

// ── Responsive Column Counts ────────────────────────────────────

/// Column count that changes at breakpoints.
#[derive(Debug, Clone)]
pub struct ResponsiveColumns {
    /// (min_viewport_width, column_count)
    pub entries: Vec<(f64, u32)>,
    pub gutter: f64,
}

impl ResponsiveColumns {
    pub fn new(gutter: f64) -> Self {
        Self {
            entries: Vec::new(),
            gutter,
        }
    }

    pub fn at(mut self, min_width: f64, columns: u32) -> Self {
        self.entries.push((min_width, columns));
        self.entries.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        self
    }

    /// Resolve column count for a viewport width.
    pub fn resolve(&self, viewport_width: f64) -> u32 {
        let mut result = 1;
        for &(min_w, cols) in &self.entries {
            if viewport_width >= min_w {
                result = cols;
            }
        }
        result
    }

    /// Generate CSS with media queries.
    pub fn to_css(&self, selector: &str) -> String {
        let mut css = String::new();
        for (i, &(min_w, cols)) in self.entries.iter().enumerate() {
            let rule = format!(
                "{selector} {{ grid-template-columns: repeat({cols}, 1fr); gap: {}px; }}\n",
                self.gutter
            );
            if i == 0 {
                css.push_str(&format!("{selector} {{ display: grid; }}\n{rule}"));
            } else {
                css.push_str(&format!(
                    "@media (min-width: {min_w}px) {{ {rule} }}\n"
                ));
            }
        }
        css
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_grid() {
        let g = GridContainer::default();
        assert_eq!(g.columns, 12);
        assert_eq!(g.gutter, 16.0);
    }

    #[test]
    fn test_grid_css() {
        let g = GridContainer::new(12).with_gutter(24.0);
        let css = g.to_css(".grid");
        assert!(css.contains("display: grid"));
        assert!(css.contains("repeat(12, 1fr)"));
        assert!(css.contains("24px"));
    }

    #[test]
    fn test_grid_max_width() {
        let g = GridContainer::new(12).with_max_width(1200.0);
        let css = g.to_css(".container");
        assert!(css.contains("max-width: 1200px"));
        assert!(css.contains("margin-inline: auto"));
    }

    #[test]
    fn test_column_width_percent() {
        let g = GridContainer::new(12);
        assert!((g.column_width_percent(6) - 50.0).abs() < 0.001);
        assert!((g.column_width_percent(4) - 33.333).abs() < 0.1);
        assert!((g.column_width_percent(12) - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_grid_item_span() {
        let item = GridItem::span(6);
        let css = item.to_css(".col-6");
        assert!(css.contains("grid-column: 1 / span 6"));
    }

    #[test]
    fn test_grid_item_offset() {
        let item = GridItem::span(4).with_offset(2);
        let css = item.to_css(".item");
        assert!(css.contains("grid-column: 3 / span 4"));
    }

    #[test]
    fn test_grid_item_push() {
        let item = GridItem::span(3).with_push(2);
        let css = item.to_css(".item");
        assert!(css.contains("grid-column: 3 / span 3"));
    }

    #[test]
    fn test_grid_item_pull() {
        let item = GridItem::span(3).with_offset(4).with_pull(2);
        let css = item.to_css(".item");
        assert!(css.contains("grid-column: 3 / span 3"));
    }

    #[test]
    fn test_auto_grid_fit() {
        let ag = AutoGrid::new(AutoSizing::AutoFit, 250.0).with_gap(20.0);
        let css = ag.to_css(".cards");
        assert!(css.contains("auto-fit"));
        assert!(css.contains("minmax(250px, 1fr)"));
        assert!(css.contains("20px"));
    }

    #[test]
    fn test_auto_grid_fill_with_max() {
        let ag = AutoGrid::new(AutoSizing::AutoFill, 200.0).with_max_width(400.0);
        let css = ag.to_css(".tiles");
        assert!(css.contains("auto-fill"));
        assert!(css.contains("minmax(200px, 400px)"));
    }

    #[test]
    fn test_named_area_grid() {
        let grid = NamedAreaGrid::new()
            .add_row(vec!["header", "header"])
            .add_row(vec!["sidebar", "main"])
            .add_row(vec!["footer", "footer"])
            .with_column_sizes(vec!["200px", "1fr"]);
        let css = grid.to_css(".layout");
        assert!(css.contains("\"header header\""));
        assert!(css.contains("\"sidebar main\""));
        assert!(css.contains("grid-template-columns: 200px 1fr"));
    }

    #[test]
    fn test_named_area_item() {
        let css = NamedAreaGrid::area_css(".nav", "sidebar");
        assert_eq!(css, ".nav { grid-area: sidebar; }\n");
    }

    #[test]
    fn test_responsive_columns_resolve() {
        let rc = ResponsiveColumns::new(16.0)
            .at(0.0, 1)
            .at(640.0, 2)
            .at(1024.0, 3)
            .at(1280.0, 4);
        assert_eq!(rc.resolve(320.0), 1);
        assert_eq!(rc.resolve(800.0), 2);
        assert_eq!(rc.resolve(1100.0), 3);
        assert_eq!(rc.resolve(1500.0), 4);
    }

    #[test]
    fn test_responsive_columns_css() {
        let rc = ResponsiveColumns::new(16.0)
            .at(0.0, 1)
            .at(768.0, 2);
        let css = rc.to_css(".grid");
        assert!(css.contains("display: grid"));
        assert!(css.contains("repeat(1, 1fr)"));
        assert!(css.contains("@media (min-width: 768px)"));
    }

    #[test]
    fn test_grid_align_display() {
        assert_eq!(GridAlign::Center.to_string(), "center");
        assert_eq!(GridAlign::Stretch.to_string(), "stretch");
    }

    #[test]
    fn test_grid_item_row_span() {
        let item = GridItem::span(4).with_row_span(2);
        let css = item.to_css(".tall");
        assert!(css.contains("grid-row: span 2"));
    }
}
