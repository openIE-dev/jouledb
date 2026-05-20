//! Parallel coordinates plot — multi-dimensional data visualization.
//!
//! Each axis is a vertical line; data rows are polylines connecting values
//! across axes. Supports brushing, axis reordering, group coloring, and SVG output.

use serde::{Deserialize, Serialize};

// ── Core Types ──────────────────────────────────────────────────

/// A single axis in the parallel coordinates plot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelAxis {
    pub label: String,
    pub min: f64,
    pub max: f64,
    /// Whether this axis is inverted (higher values at bottom).
    pub inverted: bool,
}

impl ParallelAxis {
    pub fn new(label: impl Into<String>, min: f64, max: f64) -> Self {
        Self {
            label: label.into(),
            min,
            max,
            inverted: false,
        }
    }

    pub fn inverted(mut self) -> Self {
        self.inverted = true;
        self
    }

    /// Normalize a raw value to 0.0–1.0 for this axis.
    pub fn normalize(&self, value: f64) -> f64 {
        if (self.max - self.min).abs() < f64::EPSILON {
            return 0.5;
        }
        let normalized = (value - self.min) / (self.max - self.min);
        let clamped = normalized.clamp(0.0, 1.0);
        if self.inverted {
            1.0 - clamped
        } else {
            clamped
        }
    }
}

/// A data row with one value per axis and an optional group label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRow {
    pub values: Vec<f64>,
    pub group: Option<String>,
    pub label: Option<String>,
}

impl DataRow {
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            group: None,
            label: None,
        }
    }

    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A brush selection on one axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brush {
    pub axis_index: usize,
    pub min_value: f64,
    pub max_value: f64,
}

impl Brush {
    pub fn new(axis_index: usize, min_value: f64, max_value: f64) -> Self {
        Self {
            axis_index,
            min_value,
            max_value,
        }
    }

    /// Whether a row's value on this axis falls within the brush range.
    pub fn matches(&self, row: &DataRow) -> bool {
        if let Some(&val) = row.values.get(self.axis_index) {
            val >= self.min_value && val <= self.max_value
        } else {
            false
        }
    }
}

// ── Parallel Coordinates Chart ──────────────────────────────────

/// Parallel coordinates chart model.
#[derive(Debug, Clone)]
pub struct ParallelCoordsChart {
    pub axes: Vec<ParallelAxis>,
    pub rows: Vec<DataRow>,
    pub axis_order: Vec<usize>,
    pub brushes: Vec<Brush>,
    /// Group name -> color mapping.
    pub group_colors: Vec<(String, String)>,
}

impl ParallelCoordsChart {
    pub fn new(axes: Vec<ParallelAxis>) -> Self {
        let axis_order: Vec<usize> = (0..axes.len()).collect();
        Self {
            axes,
            rows: Vec::new(),
            axis_order,
            brushes: Vec::new(),
            group_colors: Vec::new(),
        }
    }

    /// Add a data row.
    pub fn add_row(&mut self, row: DataRow) {
        self.rows.push(row);
    }

    /// Add multiple rows.
    pub fn add_rows(&mut self, rows: Vec<DataRow>) {
        self.rows.extend(rows);
    }

    /// Set axis display order.
    pub fn set_axis_order(&mut self, order: Vec<usize>) {
        self.axis_order = order;
    }

    /// Reorder axes by swapping two positions.
    pub fn swap_axes(&mut self, a: usize, b: usize) {
        if a < self.axis_order.len() && b < self.axis_order.len() {
            self.axis_order.swap(a, b);
        }
    }

    /// Add a brush selection.
    pub fn add_brush(&mut self, brush: Brush) {
        self.brushes.push(brush);
    }

    /// Clear all brushes.
    pub fn clear_brushes(&mut self) {
        self.brushes.clear();
    }

    /// Assign a color to a group.
    pub fn set_group_color(&mut self, group: impl Into<String>, color: impl Into<String>) {
        let group = group.into();
        let color = color.into();
        if let Some(entry) = self.group_colors.iter_mut().find(|(g, _)| g == &group) {
            entry.1 = color;
        } else {
            self.group_colors.push((group, color));
        }
    }

    /// Get color for a group.
    pub fn get_group_color(&self, group: &str) -> Option<&str> {
        self.group_colors
            .iter()
            .find(|(g, _)| g == group)
            .map(|(_, c)| c.as_str())
    }

    /// Filter rows that match all active brushes.
    pub fn filtered_rows(&self) -> Vec<&DataRow> {
        if self.brushes.is_empty() {
            return self.rows.iter().collect();
        }
        self.rows
            .iter()
            .filter(|row| self.brushes.iter().all(|brush| brush.matches(row)))
            .collect()
    }

    /// Compute normalized y-positions for a row across all axes (in axis_order).
    pub fn line_positions(&self, row: &DataRow) -> Vec<f64> {
        self.axis_order
            .iter()
            .map(|idx| {
                let axis: &_ = &self.axes[*idx];
                let val = row.values.get(*idx).copied().unwrap_or(0.0);
                axis.normalize(val)
            })
            .collect()
    }

    /// Generate SVG polyline points for a row.
    /// Axes are evenly spaced across `width`, values mapped to `height`.
    pub fn svg_polyline_points(
        &self,
        row: &DataRow,
        width: f64,
        height: f64,
        padding: f64,
    ) -> String {
        let n = self.axis_order.len();
        if n == 0 {
            return String::new();
        }
        let usable_width = width - 2.0 * padding;
        let usable_height = height - 2.0 * padding;
        let spacing = if n > 1 {
            usable_width / (n - 1) as f64
        } else {
            0.0
        };

        let positions = self.line_positions(row);
        positions
            .iter()
            .enumerate()
            .map(|(i, &norm)| {
                let x = padding + i as f64 * spacing;
                let y = padding + (1.0 - norm) * usable_height;
                format!("{:.1},{:.1}", x, y)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate complete SVG for the chart.
    pub fn to_svg(&self, width: f64, height: f64) -> String {
        let padding = 40.0;
        let n = self.axis_order.len();
        let usable_width = width - 2.0 * padding;
        let spacing = if n > 1 {
            usable_width / (n - 1) as f64
        } else {
            0.0
        };

        let default_color = "#4a90d9".to_string();
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">"#
        );

        // Draw axis lines
        for (i, &idx) in self.axis_order.iter().enumerate() {
            let x = padding + i as f64 * spacing;
            svg.push_str(&format!(
                r##"<line x1="{x:.1}" y1="{padding}" x2="{x:.1}" y2="{:.1}" stroke="#ccc" stroke-width="1"/>"##,
                height - padding
            ));
            // Axis label
            svg.push_str(&format!(
                r#"<text x="{x:.1}" y="{:.1}" text-anchor="middle" font-size="12">{}</text>"#,
                height - padding + 20.0,
                self.axes[idx].label
            ));
        }

        // Draw polylines for filtered rows
        let filtered = self.filtered_rows();
        for row in &filtered {
            let points = self.svg_polyline_points(row, width, height, padding);
            let color = row
                .group
                .as_ref()
                .and_then(|g| self.get_group_color(g))
                .unwrap_or(&default_color);
            svg.push_str(&format!(
                r#"<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.6"/>"#
            ));
        }

        svg.push_str("</svg>");
        svg
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chart() -> ParallelCoordsChart {
        let axes = vec![
            ParallelAxis::new("Speed", 0.0, 100.0),
            ParallelAxis::new("Weight", 1000.0, 5000.0),
            ParallelAxis::new("MPG", 10.0, 50.0),
        ];
        let mut chart = ParallelCoordsChart::new(axes);
        chart.add_row(DataRow::new(vec![80.0, 2000.0, 30.0]).with_group("sedan"));
        chart.add_row(DataRow::new(vec![60.0, 4000.0, 20.0]).with_group("truck"));
        chart.add_row(DataRow::new(vec![90.0, 1500.0, 40.0]).with_group("sedan"));
        chart
    }

    #[test]
    fn normalize_value() {
        let axis = ParallelAxis::new("X", 0.0, 100.0);
        assert!((axis.normalize(50.0) - 0.5).abs() < f64::EPSILON);
        assert!((axis.normalize(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((axis.normalize(100.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_inverted() {
        let axis = ParallelAxis::new("X", 0.0, 100.0).inverted();
        assert!((axis.normalize(0.0) - 1.0).abs() < f64::EPSILON);
        assert!((axis.normalize(100.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_clamped() {
        let axis = ParallelAxis::new("X", 0.0, 100.0);
        assert!((axis.normalize(150.0) - 1.0).abs() < f64::EPSILON);
        assert!((axis.normalize(-10.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_zero_range() {
        let axis = ParallelAxis::new("X", 5.0, 5.0);
        assert!((axis.normalize(5.0) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn line_positions() {
        let chart = sample_chart();
        let positions = chart.line_positions(&chart.rows[0]);
        assert_eq!(positions.len(), 3);
        assert!((positions[0] - 0.8).abs() < f64::EPSILON); // 80/100
        assert!((positions[1] - 0.25).abs() < f64::EPSILON); // (2000-1000)/4000
    }

    #[test]
    fn brush_filter() {
        let mut chart = sample_chart();
        chart.add_brush(Brush::new(0, 70.0, 100.0)); // Speed >= 70
        let filtered = chart.filtered_rows();
        assert_eq!(filtered.len(), 2); // 80 and 90
    }

    #[test]
    fn multiple_brushes() {
        let mut chart = sample_chart();
        chart.add_brush(Brush::new(0, 70.0, 100.0)); // Speed >= 70
        chart.add_brush(Brush::new(2, 25.0, 35.0)); // MPG 25-35
        let filtered = chart.filtered_rows();
        assert_eq!(filtered.len(), 1); // only the 80/2000/30 row
    }

    #[test]
    fn clear_brushes() {
        let mut chart = sample_chart();
        chart.add_brush(Brush::new(0, 90.0, 100.0));
        assert_eq!(chart.filtered_rows().len(), 1);
        chart.clear_brushes();
        assert_eq!(chart.filtered_rows().len(), 3);
    }

    #[test]
    fn axis_reorder() {
        let mut chart = sample_chart();
        chart.swap_axes(0, 2);
        assert_eq!(chart.axis_order, vec![2, 1, 0]);
    }

    #[test]
    fn group_colors() {
        let mut chart = sample_chart();
        chart.set_group_color("sedan", "#ff0000");
        chart.set_group_color("truck", "#0000ff");
        assert_eq!(chart.get_group_color("sedan"), Some("#ff0000"));
        assert_eq!(chart.get_group_color("truck"), Some("#0000ff"));
        assert_eq!(chart.get_group_color("suv"), None);
    }

    #[test]
    fn svg_polyline_points() {
        let chart = sample_chart();
        let points = chart.svg_polyline_points(&chart.rows[0], 400.0, 300.0, 40.0);
        assert!(!points.is_empty());
        let parts: Vec<&str> = points.split(' ').collect();
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn svg_output() {
        let mut chart = sample_chart();
        chart.set_group_color("sedan", "#e74c3c");
        let svg = chart.to_svg(600.0, 400.0);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("polyline"));
        assert!(svg.contains("#e74c3c"));
        assert!(svg.contains("Speed"));
    }

    #[test]
    fn set_axis_order() {
        let mut chart = sample_chart();
        chart.set_axis_order(vec![2, 0, 1]);
        assert_eq!(chart.axis_order, vec![2, 0, 1]);
        let positions = chart.line_positions(&chart.rows[0]);
        // First axis is now MPG (index 2): (30-10)/40 = 0.5
        assert!((positions[0] - 0.5).abs() < f64::EPSILON);
    }
}
