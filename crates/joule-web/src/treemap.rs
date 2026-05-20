//! Treemap layout: squarified algorithm, slice-and-dice, nested treemaps,
//! color mapping by value/category, label fitting, hover/selection state,
//! size normalization, padding between cells.  Pure Rust SVG output.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// A single leaf or branch in the treemap hierarchy.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub id: String,
    pub label: String,
    pub value: f64,
    pub category: Option<String>,
    pub children: Vec<TreeNode>,
    pub color: Option<String>,
}

impl TreeNode {
    pub fn leaf(id: impl Into<String>, label: impl Into<String>, value: f64) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            value,
            category: None,
            children: Vec::new(),
            color: None,
        }
    }

    pub fn branch(id: impl Into<String>, label: impl Into<String>, children: Vec<TreeNode>) -> Self {
        let value: f64 = children.iter().map(|c| c.total_value()).sum();
        Self {
            id: id.into(),
            label: label.into(),
            value,
            category: None,
            children,
            color: None,
        }
    }

    pub fn with_category(mut self, cat: impl Into<String>) -> Self {
        self.category = Some(cat.into());
        self
    }

    pub fn with_color(mut self, color: impl Into<String>) -> Self {
        self.color = Some(color.into());
        self
    }

    /// Total value including all descendants.
    pub fn total_value(&self) -> f64 {
        if self.children.is_empty() {
            self.value
        } else {
            self.children.iter().map(|c| c.total_value()).sum()
        }
    }

    /// Whether this is a leaf (no children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Count all leaves in the subtree.
    pub fn leaf_count(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.children.iter().map(|c| c.leaf_count()).sum()
        }
    }
}

/// A rectangle in 2D space.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }

    pub fn area(&self) -> f64 {
        self.w * self.h
    }

    pub fn aspect_ratio(&self) -> f64 {
        if self.h.abs() < f64::EPSILON {
            f64::INFINITY
        } else {
            (self.w / self.h).max(self.h / self.w)
        }
    }

    /// Apply inner padding.
    pub fn inset(&self, padding: f64) -> Rect {
        let p2 = padding * 2.0;
        Rect {
            x: self.x + padding,
            y: self.y + padding,
            w: (self.w - p2).max(0.0),
            h: (self.h - p2).max(0.0),
        }
    }

    pub fn shorter_side(&self) -> f64 {
        self.w.min(self.h)
    }

    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }
}

/// Layout algorithm choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutAlgorithm {
    Squarified,
    SliceAndDice,
}

/// Color mapping strategy.
#[derive(Debug, Clone)]
pub enum ColorMapping {
    /// Fixed color per node (uses node.color or fallback).
    Fixed(String),
    /// Map value to a gradient between two colors.
    ValueGradient { low: String, high: String },
    /// Map category string to a color palette.
    Category(HashMap<String, String>),
}

/// Interactive state for a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    Normal,
    Hovered,
    Selected,
}

/// A laid-out cell ready for rendering.
#[derive(Debug, Clone)]
pub struct LayoutCell {
    pub id: String,
    pub label: String,
    pub value: f64,
    pub rect: Rect,
    pub depth: usize,
    pub color: String,
    pub state: CellState,
    pub label_fits: bool,
}

/// Treemap configuration.
#[derive(Debug, Clone)]
pub struct TreemapConfig {
    pub width: f64,
    pub height: f64,
    pub algorithm: LayoutAlgorithm,
    pub padding: f64,
    pub color_mapping: ColorMapping,
    /// Minimum cell dimension to display labels.
    pub min_label_size: f64,
}

impl Default for TreemapConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            algorithm: LayoutAlgorithm::Squarified,
            padding: 2.0,
            color_mapping: ColorMapping::Fixed("#3498db".into()),
            min_label_size: 30.0,
        }
    }
}

// ── Layout engine ────────────────────────────────────────────────

/// Compute layout cells from a tree.
pub fn layout(root: &TreeNode, config: &TreemapConfig) -> Vec<LayoutCell> {
    let mut cells = Vec::new();
    let bounds = Rect::new(0.0, 0.0, config.width, config.height);
    layout_recursive(root, bounds, 0, config, &mut cells);
    cells
}

fn layout_recursive(
    node: &TreeNode,
    bounds: Rect,
    depth: usize,
    config: &TreemapConfig,
    out: &mut Vec<LayoutCell>,
) {
    if node.is_leaf() {
        let color = resolve_color(node, config);
        let label_fits =
            bounds.w >= config.min_label_size && bounds.h >= config.min_label_size;
        out.push(LayoutCell {
            id: node.id.clone(),
            label: node.label.clone(),
            value: node.value,
            rect: bounds,
            depth,
            color,
            state: CellState::Normal,
            label_fits,
        });
        return;
    }

    let inner = bounds.inset(config.padding);
    if inner.w <= 0.0 || inner.h <= 0.0 {
        return;
    }

    let total: f64 = node.children.iter().map(|c| c.total_value()).sum();
    if total <= 0.0 {
        return;
    }

    match config.algorithm {
        LayoutAlgorithm::Squarified => {
            let mut items: Vec<(&TreeNode, f64)> = node
                .children
                .iter()
                .map(|c| (c, c.total_value()))
                .collect();
            items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            squarify(&items, inner, total, depth + 1, config, out);
        }
        LayoutAlgorithm::SliceAndDice => {
            slice_and_dice(&node.children, inner, total, depth + 1, depth % 2 == 0, config, out);
        }
    }
}

fn slice_and_dice(
    children: &[TreeNode],
    bounds: Rect,
    total: f64,
    depth: usize,
    horizontal: bool,
    config: &TreemapConfig,
    out: &mut Vec<LayoutCell>,
) {
    let mut offset = 0.0;
    for child in children {
        let frac = child.total_value() / total.max(f64::EPSILON);
        let r = if horizontal {
            Rect::new(bounds.x + offset, bounds.y, bounds.w * frac, bounds.h)
        } else {
            Rect::new(bounds.x, bounds.y + offset, bounds.w, bounds.h * frac)
        };
        if horizontal {
            offset += bounds.w * frac;
        } else {
            offset += bounds.h * frac;
        }
        layout_recursive(&child, r, depth, config, out);
    }
}

fn squarify(
    items: &[(&TreeNode, f64)],
    bounds: Rect,
    total: f64,
    depth: usize,
    config: &TreemapConfig,
    out: &mut Vec<LayoutCell>,
) {
    if items.is_empty() || bounds.w <= 0.0 || bounds.h <= 0.0 {
        return;
    }
    if items.len() == 1 {
        layout_recursive(items[0].0, bounds, depth, config, out);
        return;
    }

    // Greedy: fill rows along the shorter side.
    let side = bounds.shorter_side();
    let area_scale = bounds.area() / total.max(f64::EPSILON);

    let mut row: Vec<usize> = Vec::new();
    let mut row_area = 0.0;
    let mut best_ratio = f64::INFINITY;

    for i in 0..items.len() {
        let a = items[i].1 * area_scale;
        let trial_area = row_area + a;
        let ratio = worst_ratio_for_row(
            &items[..=i]
                .iter()
                .map(|(_, v)| v * area_scale)
                .collect::<Vec<_>>(),
            side,
            trial_area,
        );
        if ratio <= best_ratio || row.is_empty() {
            row.push(i);
            row_area = trial_area;
            best_ratio = ratio;
        } else {
            break;
        }
    }

    // Lay out the row.
    let horizontal = bounds.w >= bounds.h;
    let row_span = row_area / side.max(f64::EPSILON);
    let mut offset = 0.0;

    for &idx in &row {
        let a = items[idx].1 * area_scale;
        let cell_span = a / row_span.max(f64::EPSILON);
        let r = if horizontal {
            Rect::new(bounds.x, bounds.y + offset, row_span, cell_span)
        } else {
            Rect::new(bounds.x + offset, bounds.y, cell_span, row_span)
        };
        offset += cell_span;
        layout_recursive(items[idx].0, r, depth, config, out);
    }

    // Remaining items go in the leftover rectangle.
    let remaining = &items[row.len()..];
    if !remaining.is_empty() {
        let remaining_total: f64 = remaining.iter().map(|(_, v)| *v).sum();
        let leftover = if horizontal {
            Rect::new(bounds.x + row_span, bounds.y, bounds.w - row_span, bounds.h)
        } else {
            Rect::new(bounds.x, bounds.y + row_span, bounds.w, bounds.h - row_span)
        };
        squarify(remaining, leftover, remaining_total, depth, config, out);
    }
}

fn worst_ratio_for_row(areas: &[f64], side: f64, total_area: f64) -> f64 {
    let side2 = side * side;
    let mut worst = 0.0_f64;
    for &a in areas {
        if a <= 0.0 {
            continue;
        }
        let r1 = (side2 * a) / (total_area * total_area);
        let r2 = (total_area * total_area) / (side2 * a);
        worst = worst.max(r1.max(r2));
    }
    worst
}

fn resolve_color(node: &TreeNode, config: &TreemapConfig) -> String {
    if let Some(c) = &node.color {
        return c.clone();
    }
    match &config.color_mapping {
        ColorMapping::Fixed(c) => c.clone(),
        ColorMapping::ValueGradient { low, high } => {
            // Simple linear interpolation placeholder — blend two hex colors.
            let _ = (low, high);
            format!("#{:02x}{:02x}{:02x}", 52, 152, 219)
        }
        ColorMapping::Category(map) => {
            if let Some(cat) = &node.category {
                map.get(cat).cloned().unwrap_or_else(|| "#95a5a6".into())
            } else {
                "#95a5a6".into()
            }
        }
    }
}

// ── Size normalization ───────────────────────────────────────────

/// Normalize all leaf values in a tree so they sum to `target_sum`.
pub fn normalize_values(root: &mut TreeNode, target_sum: f64) {
    let total = root.total_value();
    if total <= 0.0 {
        return;
    }
    let scale = target_sum / total;
    normalize_recursive(root, scale);
}

fn normalize_recursive(node: &mut TreeNode, scale: f64) {
    if node.is_leaf() {
        node.value *= scale;
    } else {
        for child in &mut node.children {
            normalize_recursive(child, scale);
        }
        node.value = node.children.iter().map(|c| c.total_value()).sum();
    }
}

// ── Hit testing ──────────────────────────────────────────────────

/// Find which cell contains the given point.
pub fn hit_test(cells: &[LayoutCell], x: f64, y: f64) -> Option<&LayoutCell> {
    // Return the deepest (last) cell that contains the point.
    cells.iter().rev().find(|c| c.rect.contains(x, y))
}

/// Set hover state on the cell under (x, y), clearing others.
pub fn apply_hover(cells: &mut [LayoutCell], x: f64, y: f64) {
    let hit_id = cells
        .iter()
        .rev()
        .find(|c| c.rect.contains(x, y))
        .map(|c| c.id.clone());
    for cell in cells.iter_mut() {
        if cell.state == CellState::Hovered {
            cell.state = CellState::Normal;
        }
        if let Some(hid) = &hit_id {
            if cell.id == *hid {
                cell.state = CellState::Hovered;
            }
        }
    }
}

/// Toggle selection on the cell under (x, y).
pub fn toggle_selection(cells: &mut [LayoutCell], x: f64, y: f64) {
    if let Some(hit_id) = cells
        .iter()
        .rev()
        .find(|c| c.rect.contains(x, y))
        .map(|c| c.id.clone())
    {
        for cell in cells.iter_mut() {
            if cell.id == hit_id {
                cell.state = match cell.state {
                    CellState::Selected => CellState::Normal,
                    _ => CellState::Selected,
                };
            }
        }
    }
}

// ── SVG rendering ────────────────────────────────────────────────

/// Render laid-out cells to an SVG string.
pub fn render_svg(cells: &[LayoutCell], width: f64, height: f64) -> String {
    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">"#
    );
    for cell in cells {
        let opacity = match cell.state {
            CellState::Normal => 1.0,
            CellState::Hovered => 0.8,
            CellState::Selected => 0.6,
        };
        let stroke = match cell.state {
            CellState::Selected => "stroke=\"#000\" stroke-width=\"2\"",
            _ => "stroke=\"#fff\" stroke-width=\"1\"",
        };
        let _ = write!(
            svg,
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" opacity="{opacity}" {stroke}/>"#,
            cell.rect.x, cell.rect.y, cell.rect.w, cell.rect.h, cell.color,
        );
        if cell.label_fits {
            let tx = cell.rect.x + cell.rect.w / 2.0;
            let ty = cell.rect.y + cell.rect.h / 2.0;
            let _ = write!(
                svg,
                r#"<text x="{tx:.1}" y="{ty:.1}" text-anchor="middle" dominant-baseline="central" font-size="12">{}</text>"#,
                cell.label,
            );
        }
    }
    svg.push_str("</svg>");
    svg
}

// ── Tooltip ──────────────────────────────────────────────────────

/// Generate a tooltip string for the cell at (x, y).
pub fn tooltip(cells: &[LayoutCell], x: f64, y: f64) -> Option<String> {
    hit_test(cells, x, y).map(|c| format!("{}: {:.2}", c.label, c.value))
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> TreeNode {
        TreeNode::branch(
            "root",
            "Root",
            vec![
                TreeNode::leaf("a", "A", 6.0),
                TreeNode::leaf("b", "B", 3.0),
                TreeNode::leaf("c", "C", 2.0),
                TreeNode::leaf("d", "D", 1.0),
            ],
        )
    }

    #[test]
    fn total_value_flat() {
        let tree = sample_tree();
        assert!((tree.total_value() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn total_value_nested() {
        let inner = TreeNode::branch("inner", "Inner", vec![
            TreeNode::leaf("x", "X", 5.0),
            TreeNode::leaf("y", "Y", 3.0),
        ]);
        let root = TreeNode::branch("root", "Root", vec![inner, TreeNode::leaf("z", "Z", 2.0)]);
        assert!((root.total_value() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn leaf_count() {
        let tree = sample_tree();
        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn squarified_layout_covers_area() {
        let tree = sample_tree();
        let config = TreemapConfig::default();
        let cells = layout(&tree, &config);
        assert_eq!(cells.len(), 4);
        let total_area: f64 = cells.iter().map(|c| c.rect.area()).sum();
        let expected = config.width * config.height;
        // With padding the total is slightly less.
        assert!(total_area <= expected + 1.0);
        assert!(total_area > expected * 0.8);
    }

    #[test]
    fn slice_and_dice_layout() {
        let tree = sample_tree();
        let config = TreemapConfig {
            algorithm: LayoutAlgorithm::SliceAndDice,
            padding: 0.0,
            ..Default::default()
        };
        let cells = layout(&tree, &config);
        assert_eq!(cells.len(), 4);
        let total_area: f64 = cells.iter().map(|c| c.rect.area()).sum();
        let expected = config.width * config.height;
        assert!((total_area - expected).abs() < 1.0);
    }

    #[test]
    fn hit_test_finds_cell() {
        let tree = sample_tree();
        let config = TreemapConfig {
            padding: 0.0,
            ..Default::default()
        };
        let cells = layout(&tree, &config);
        let hit = hit_test(&cells, 10.0, 10.0);
        assert!(hit.is_some());
    }

    #[test]
    fn hit_test_outside() {
        let tree = sample_tree();
        let config = TreemapConfig::default();
        let cells = layout(&tree, &config);
        let hit = hit_test(&cells, -10.0, -10.0);
        assert!(hit.is_none());
    }

    #[test]
    fn hover_state() {
        let tree = sample_tree();
        let config = TreemapConfig { padding: 0.0, ..Default::default() };
        let mut cells = layout(&tree, &config);
        apply_hover(&mut cells, 10.0, 10.0);
        assert!(cells.iter().any(|c| c.state == CellState::Hovered));
    }

    #[test]
    fn selection_toggle() {
        let tree = sample_tree();
        let config = TreemapConfig { padding: 0.0, ..Default::default() };
        let mut cells = layout(&tree, &config);
        toggle_selection(&mut cells, 10.0, 10.0);
        assert!(cells.iter().any(|c| c.state == CellState::Selected));
        toggle_selection(&mut cells, 10.0, 10.0);
        assert!(!cells.iter().any(|c| c.state == CellState::Selected));
    }

    #[test]
    fn normalize_values_rescales() {
        let mut tree = sample_tree();
        normalize_values(&mut tree, 100.0);
        assert!((tree.total_value() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn svg_output_valid() {
        let tree = sample_tree();
        let config = TreemapConfig::default();
        let cells = layout(&tree, &config);
        let svg = render_svg(&cells, config.width, config.height);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn tooltip_generation() {
        let tree = sample_tree();
        let config = TreemapConfig { padding: 0.0, ..Default::default() };
        let cells = layout(&tree, &config);
        let tip = tooltip(&cells, 10.0, 10.0);
        assert!(tip.is_some());
    }

    #[test]
    fn category_color_mapping() {
        let mut map = HashMap::new();
        map.insert("tech".into(), "#e74c3c".into());
        let config = TreemapConfig {
            color_mapping: ColorMapping::Category(map),
            padding: 0.0,
            ..Default::default()
        };
        let tree = TreeNode::branch("r", "R", vec![
            TreeNode::leaf("a", "A", 5.0).with_category("tech"),
            TreeNode::leaf("b", "B", 3.0).with_category("other"),
        ]);
        let cells = layout(&tree, &config);
        let a_cell = cells.iter().find(|c| c.id == "a").unwrap();
        assert_eq!(a_cell.color, "#e74c3c");
        let b_cell = cells.iter().find(|c| c.id == "b").unwrap();
        assert_eq!(b_cell.color, "#95a5a6"); // fallback
    }

    #[test]
    fn nested_treemap_layout() {
        let inner = TreeNode::branch("grp", "Group", vec![
            TreeNode::leaf("x", "X", 4.0),
            TreeNode::leaf("y", "Y", 2.0),
        ]);
        let root = TreeNode::branch("root", "Root", vec![
            inner,
            TreeNode::leaf("z", "Z", 4.0),
        ]);
        let config = TreemapConfig { padding: 0.0, ..Default::default() };
        let cells = layout(&root, &config);
        assert_eq!(cells.len(), 3);
    }

    #[test]
    fn rect_inset() {
        let r = Rect::new(0.0, 0.0, 100.0, 80.0);
        let inner = r.inset(5.0);
        assert!((inner.x - 5.0).abs() < 1e-9);
        assert!((inner.w - 90.0).abs() < 1e-9);
        assert!((inner.h - 70.0).abs() < 1e-9);
    }
}
