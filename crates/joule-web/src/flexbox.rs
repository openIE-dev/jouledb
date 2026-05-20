//! Flexbox layout engine: flex containers, items, line wrapping, and alignment.
//!
//! Pure math — no browser dependency. Implements the core flexbox algorithm:
//! collect into lines, resolve flexible lengths, cross-axis alignment,
//! and final rect computation.

use std::cmp::Ordering;

// ── Direction & Alignment ───────────────────────────────────────

/// Main axis direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlexDirection {
    Row,
    RowReverse,
    Column,
    ColumnReverse,
}

impl FlexDirection {
    fn is_row(self) -> bool {
        matches!(self, FlexDirection::Row | FlexDirection::RowReverse)
    }

    fn is_reversed(self) -> bool {
        matches!(self, FlexDirection::RowReverse | FlexDirection::ColumnReverse)
    }
}

/// Wrapping behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlexWrap {
    NoWrap,
    Wrap,
    WrapReverse,
}

/// Main-axis content distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JustifyContent {
    FlexStart,
    FlexEnd,
    Center,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

/// Cross-axis item alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignItems {
    FlexStart,
    FlexEnd,
    Center,
    Stretch,
    Baseline,
}

/// Cross-axis content distribution (multi-line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignContent {
    FlexStart,
    FlexEnd,
    Center,
    SpaceBetween,
    SpaceAround,
    Stretch,
}

// ── Container & Item ────────────────────────────────────────────

/// Flex container configuration.
#[derive(Debug, Clone)]
pub struct FlexContainer {
    pub direction: FlexDirection,
    pub wrap: FlexWrap,
    pub justify_content: JustifyContent,
    pub align_items: AlignItems,
    pub align_content: AlignContent,
    pub gap: f64,
    pub cross_gap: f64,
}

impl Default for FlexContainer {
    fn default() -> Self {
        Self {
            direction: FlexDirection::Row,
            wrap: FlexWrap::NoWrap,
            justify_content: JustifyContent::FlexStart,
            align_items: AlignItems::Stretch,
            align_content: AlignContent::Stretch,
            gap: 0.0,
            cross_gap: 0.0,
        }
    }
}

/// A flex item with sizing hints.
#[derive(Debug, Clone)]
pub struct FlexItem {
    pub basis: f64,
    pub grow: f64,
    pub shrink: f64,
    pub align_self: Option<AlignItems>,
    pub min_size: f64,
    pub max_size: f64,
    pub order: i32,
}

impl Default for FlexItem {
    fn default() -> Self {
        Self {
            basis: 0.0,
            grow: 0.0,
            shrink: 1.0,
            align_self: None,
            min_size: 0.0,
            max_size: f64::INFINITY,
            order: 0,
        }
    }
}

/// Computed rectangle for a flex item.
#[derive(Debug, Clone, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

// ── Internal structures ─────────────────────────────────────────

struct IndexedItem {
    original_index: usize,
    item: FlexItem,
}

struct FlexLine {
    items: Vec<(usize, f64)>, // (original_index, resolved main size)
    main_size: f64,
    cross_size: f64,
}

// ── Layout Algorithm ────────────────────────────────────────────

/// Compute final rects for all flex items.
pub fn layout(
    container: &FlexContainer,
    items: &[FlexItem],
    container_width: f64,
    container_height: f64,
) -> Vec<Rect> {
    if items.is_empty() {
        return vec![];
    }

    let main_available = if container.direction.is_row() {
        container_width
    } else {
        container_height
    };
    let cross_available = if container.direction.is_row() {
        container_height
    } else {
        container_width
    };

    // Sort by order (stable)
    let mut indexed: Vec<IndexedItem> = items
        .iter()
        .enumerate()
        .map(|(i, item)| IndexedItem { original_index: i, item: item.clone() })
        .collect();
    indexed.sort_by(|a, b| a.item.order.cmp(&b.item.order).then(Ordering::Equal));

    // Collect into lines
    let lines = collect_lines(&indexed, container, main_available);

    // Distribute cross space among lines
    let total_cross_gap = container.cross_gap * (lines.len().saturating_sub(1)) as f64;
    let total_line_cross: f64 = lines.iter().map(|l| l.cross_size).sum();
    let remaining_cross = (cross_available - total_line_cross - total_cross_gap).max(0.0);

    let line_cross_sizes = distribute_cross_space(
        &lines,
        container.align_content,
        remaining_cross,
        cross_available,
    );

    // Compute cross offsets
    let mut cross_offsets = Vec::with_capacity(lines.len());
    let cross_start_offset = cross_start_offset(
        container.align_content,
        remaining_cross,
        lines.len(),
    );
    let mut cross_pos = cross_start_offset;
    for (i, cs) in line_cross_sizes.iter().enumerate() {
        cross_offsets.push(cross_pos);
        cross_pos += cs;
        if i + 1 < lines.len() {
            cross_pos += container.cross_gap
                + cross_between_space(container.align_content, remaining_cross, lines.len());
        }
    }

    // Reverse cross if wrap-reverse
    if container.wrap == FlexWrap::WrapReverse {
        cross_offsets.reverse();
    }

    // Compute final rects
    let mut rects = vec![Rect { x: 0.0, y: 0.0, width: 0.0, height: 0.0 }; items.len()];

    for (line_idx, line) in lines.iter().enumerate() {
        let line_cross = line_cross_sizes[line_idx];
        let line_cross_offset = cross_offsets[line_idx];

        // Justify main axis
        let total_main: f64 = line.items.iter().map(|(_, s)| *s).sum();
        let total_gap = container.gap * (line.items.len().saturating_sub(1)) as f64;
        let free_space = (main_available - total_main - total_gap).max(0.0);

        let (mut main_pos, between, before_first) =
            justify_offsets(container.justify_content, free_space, line.items.len());

        main_pos += before_first;

        let items_iter: Box<dyn Iterator<Item = &(usize, f64)>> =
            if container.direction.is_reversed() {
                Box::new(line.items.iter().rev())
            } else {
                Box::new(line.items.iter())
            };

        for (orig_idx, main_size) in items_iter {
            let align = items[*orig_idx]
                .align_self
                .unwrap_or(container.align_items);

            let cross_item_size = match align {
                AlignItems::Stretch => line_cross,
                _ => 0.0_f64.max(line_cross.min(items[*orig_idx].basis)),
            };

            let cross_offset = match align {
                AlignItems::FlexStart | AlignItems::Baseline | AlignItems::Stretch => 0.0,
                AlignItems::FlexEnd => line_cross - cross_item_size,
                AlignItems::Center => (line_cross - cross_item_size) / 2.0,
            };

            let (x, y, w, h) = if container.direction.is_row() {
                (main_pos, line_cross_offset + cross_offset, *main_size, cross_item_size)
            } else {
                (line_cross_offset + cross_offset, main_pos, cross_item_size, *main_size)
            };

            rects[*orig_idx] = Rect { x, y, width: w, height: h };
            main_pos += main_size + container.gap + between;
        }
    }

    rects
}

fn collect_lines(
    indexed: &[IndexedItem],
    container: &FlexContainer,
    main_available: f64,
) -> Vec<FlexLine> {
    let mut lines: Vec<FlexLine> = Vec::new();
    let mut current_items: Vec<(usize, FlexItem)> = Vec::new();
    let mut current_main = 0.0;

    for ii in indexed {
        let item_main = ii.item.basis;
        let gap_cost = if current_items.is_empty() { 0.0 } else { container.gap };

        if container.wrap != FlexWrap::NoWrap
            && !current_items.is_empty()
            && current_main + gap_cost + item_main > main_available
        {
            lines.push(resolve_line(&current_items, main_available, container.gap));
            current_items.clear();
            current_main = 0.0;
        }
        current_main += if current_items.is_empty() { 0.0 } else { container.gap } + item_main;
        current_items.push((ii.original_index, ii.item.clone()));
    }
    if !current_items.is_empty() {
        lines.push(resolve_line(&current_items, main_available, container.gap));
    }
    lines
}

fn resolve_line(
    items: &[(usize, FlexItem)],
    main_available: f64,
    gap: f64,
) -> FlexLine {
    let total_basis: f64 = items.iter().map(|(_, it)| it.basis).sum();
    let total_gap = gap * (items.len().saturating_sub(1)) as f64;
    let free = main_available - total_basis - total_gap;

    let mut sizes: Vec<(usize, f64)> = Vec::with_capacity(items.len());

    if free >= 0.0 {
        // Grow
        let total_grow: f64 = items.iter().map(|(_, it)| it.grow).sum();
        for (idx, item) in items {
            let extra = if total_grow > 0.0 {
                free * (item.grow / total_grow)
            } else {
                0.0
            };
            let size = (item.basis + extra).clamp(item.min_size, item.max_size);
            sizes.push((*idx, size));
        }
    } else {
        // Shrink
        let total_shrink: f64 = items.iter().map(|(_, it)| it.shrink * it.basis).sum();
        for (idx, item) in items {
            let reduction = if total_shrink > 0.0 {
                (-free) * (item.shrink * item.basis / total_shrink)
            } else {
                0.0
            };
            let size = (item.basis - reduction).clamp(item.min_size, item.max_size);
            sizes.push((*idx, size));
        }
    }

    let main_size: f64 = sizes.iter().map(|(_, s)| *s).sum::<f64>() + total_gap;
    // Cross size = max of item basis (simplified; real impl uses intrinsic cross)
    let cross_size: f64 = items.iter().map(|(_, it)| it.basis).fold(0.0_f64, f64::max);

    FlexLine { items: sizes, main_size, cross_size }
}

fn distribute_cross_space(
    lines: &[FlexLine],
    align: AlignContent,
    remaining: f64,
    _total: f64,
) -> Vec<f64> {
    match align {
        AlignContent::Stretch => {
            let extra_each = if lines.is_empty() {
                0.0
            } else {
                remaining / lines.len() as f64
            };
            lines.iter().map(|l| l.cross_size + extra_each).collect()
        }
        _ => lines.iter().map(|l| l.cross_size).collect(),
    }
}

fn cross_start_offset(align: AlignContent, remaining: f64, line_count: usize) -> f64 {
    match align {
        AlignContent::FlexEnd => remaining,
        AlignContent::Center => remaining / 2.0,
        AlignContent::SpaceAround if line_count > 0 => remaining / (line_count as f64 * 2.0),
        _ => 0.0,
    }
}

fn cross_between_space(align: AlignContent, remaining: f64, line_count: usize) -> f64 {
    if line_count <= 1 {
        return 0.0;
    }
    match align {
        AlignContent::SpaceBetween => remaining / (line_count - 1) as f64,
        AlignContent::SpaceAround => remaining / line_count as f64,
        _ => 0.0,
    }
}

fn justify_offsets(
    justify: JustifyContent,
    free_space: f64,
    item_count: usize,
) -> (f64, f64, f64) {
    // Returns (start_pos, between_extra, before_first)
    if item_count == 0 {
        return (0.0, 0.0, 0.0);
    }
    match justify {
        JustifyContent::FlexStart => (0.0, 0.0, 0.0),
        JustifyContent::FlexEnd => (free_space, 0.0, 0.0),
        JustifyContent::Center => (0.0, 0.0, free_space / 2.0),
        JustifyContent::SpaceBetween => {
            if item_count <= 1 {
                (0.0, 0.0, 0.0)
            } else {
                (0.0, free_space / (item_count - 1) as f64, 0.0)
            }
        }
        JustifyContent::SpaceAround => {
            let around = free_space / item_count as f64;
            (0.0, around, around / 2.0)
        }
        JustifyContent::SpaceEvenly => {
            let even = free_space / (item_count + 1) as f64;
            (0.0, even, even)
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn basic_row_layout() {
        let container = FlexContainer::default();
        let items = vec![
            FlexItem { basis: 100.0, ..Default::default() },
            FlexItem { basis: 100.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 300.0, 50.0);
        assert!(approx(rects[0].x, 0.0));
        assert!(approx(rects[0].width, 100.0));
        assert!(approx(rects[1].x, 100.0));
        assert!(approx(rects[1].width, 100.0));
    }

    #[test]
    fn grow_distributes_space() {
        let container = FlexContainer::default();
        let items = vec![
            FlexItem { basis: 50.0, grow: 1.0, ..Default::default() },
            FlexItem { basis: 50.0, grow: 3.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 200.0, 50.0);
        assert!(approx(rects[0].width, 75.0));  // 50 + 25
        assert!(approx(rects[1].width, 125.0)); // 50 + 75
    }

    #[test]
    fn shrink_reduces_items() {
        let container = FlexContainer::default();
        let items = vec![
            FlexItem { basis: 200.0, shrink: 1.0, ..Default::default() },
            FlexItem { basis: 200.0, shrink: 1.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 300.0, 50.0);
        assert!(approx(rects[0].width, 150.0));
        assert!(approx(rects[1].width, 150.0));
    }

    #[test]
    fn column_direction() {
        let container = FlexContainer {
            direction: FlexDirection::Column,
            ..Default::default()
        };
        let items = vec![
            FlexItem { basis: 50.0, ..Default::default() },
            FlexItem { basis: 50.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 100.0, 200.0);
        assert!(approx(rects[0].y, 0.0));
        assert!(approx(rects[0].height, 50.0));
        assert!(approx(rects[1].y, 50.0));
        assert!(approx(rects[1].height, 50.0));
    }

    #[test]
    fn justify_center() {
        let container = FlexContainer {
            justify_content: JustifyContent::Center,
            ..Default::default()
        };
        let items = vec![
            FlexItem { basis: 50.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 200.0, 50.0);
        assert!(approx(rects[0].x, 75.0));
    }

    #[test]
    fn justify_space_between() {
        let container = FlexContainer {
            justify_content: JustifyContent::SpaceBetween,
            ..Default::default()
        };
        let items = vec![
            FlexItem { basis: 50.0, ..Default::default() },
            FlexItem { basis: 50.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 200.0, 50.0);
        assert!(approx(rects[0].x, 0.0));
        assert!(approx(rects[1].x, 150.0));
    }

    #[test]
    fn wrap_creates_lines() {
        let container = FlexContainer {
            wrap: FlexWrap::Wrap,
            align_content: AlignContent::FlexStart,
            ..Default::default()
        };
        let items = vec![
            FlexItem { basis: 60.0, ..Default::default() },
            FlexItem { basis: 60.0, ..Default::default() },
            FlexItem { basis: 60.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 100.0, 200.0);
        // First line: item 0 (60px), can't fit item 1 (60+60=120 > 100)
        // Second line: item 1
        // Third line: item 2
        assert!(approx(rects[0].y, 0.0));
        assert!(approx(rects[1].y, 60.0));
        assert!(approx(rects[2].y, 120.0));
    }

    #[test]
    fn gap_support() {
        let container = FlexContainer {
            gap: 10.0,
            ..Default::default()
        };
        let items = vec![
            FlexItem { basis: 50.0, ..Default::default() },
            FlexItem { basis: 50.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 200.0, 50.0);
        assert!(approx(rects[0].x, 0.0));
        assert!(approx(rects[1].x, 60.0)); // 50 + 10 gap
    }

    #[test]
    fn order_reorders_items() {
        let container = FlexContainer::default();
        let items = vec![
            FlexItem { basis: 50.0, order: 2, ..Default::default() },
            FlexItem { basis: 50.0, order: 1, ..Default::default() },
        ];
        let rects = layout(&container, &items, 200.0, 50.0);
        // Item 1 (order=1) renders first, item 0 (order=2) renders second
        assert!(approx(rects[1].x, 0.0));
        assert!(approx(rects[0].x, 50.0));
    }

    #[test]
    fn min_max_clamping() {
        let container = FlexContainer::default();
        let items = vec![
            FlexItem { basis: 50.0, grow: 1.0, max_size: 80.0, ..Default::default() },
            FlexItem { basis: 50.0, grow: 1.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 200.0, 50.0);
        assert!(rects[0].width <= 80.0 + 0.01);
    }

    #[test]
    fn empty_container() {
        let container = FlexContainer::default();
        let rects = layout(&container, &[], 200.0, 100.0);
        assert!(rects.is_empty());
    }

    #[test]
    fn align_items_center() {
        let container = FlexContainer {
            align_items: AlignItems::Center,
            ..Default::default()
        };
        let items = vec![
            FlexItem { basis: 20.0, ..Default::default() },
        ];
        let rects = layout(&container, &items, 200.0, 100.0);
        // Cross axis centering: (100 - 20) / 2 = 40
        assert!(approx(rects[0].y, 40.0));
    }
}
