//! Virtual scrolling engine for massive lists and grids.
//!
//! Replaces react-virtualized, TanStack Virtual with pure-Rust math.
//! No DOM dependency — computes visible ranges and offsets only.

// ── VirtualRange ─────────────────────────────────────────────────

/// Range of visible items and the pixel offset for the first visible item.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VirtualRange {
    pub start_index: usize,
    pub end_index: usize,
    pub offset_y: f64,
}

// ── VirtualListConfig ────────────────────────────────────────────

/// Configuration for a fixed-height virtual list.
#[derive(Debug, Clone, Copy)]
pub struct VirtualListConfig {
    pub total_items: usize,
    pub item_height: f64,
    pub container_height: f64,
    /// Extra items rendered above/below viewport.
    pub overscan: usize,
}

// ── VirtualList (fixed height) ───────────────────────────────────

/// Fixed-height virtual list. Pure math, no allocation per scroll.
pub struct VirtualList {
    config: VirtualListConfig,
    scroll_top: f64,
}

impl VirtualList {
    pub fn new(config: VirtualListConfig) -> Self {
        Self {
            config,
            scroll_top: 0.0,
        }
    }

    /// Scroll to a pixel offset; returns the visible range.
    pub fn scroll_to(&mut self, scroll_top: f64) -> VirtualRange {
        self.scroll_top = scroll_top.max(0.0);
        self.compute_range()
    }

    /// Total scrollable height.
    pub fn total_height(&self) -> f64 {
        self.config.total_items as f64 * self.config.item_height
    }

    /// Number of items visible in the viewport (excluding overscan).
    pub fn visible_count(&self) -> usize {
        if self.config.item_height <= 0.0 {
            return 0;
        }
        (self.config.container_height / self.config.item_height).ceil() as usize
    }

    /// Scroll so that a given index is at the top of the viewport.
    pub fn scroll_to_index(&mut self, index: usize) -> VirtualRange {
        let clamped = index.min(self.config.total_items.saturating_sub(1));
        let top = clamped as f64 * self.config.item_height;
        self.scroll_to(top)
    }

    /// Update total item count (e.g., after data load).
    pub fn update_total(&mut self, total: usize) {
        self.config.total_items = total;
    }

    fn compute_range(&self) -> VirtualRange {
        if self.config.total_items == 0 || self.config.item_height <= 0.0 {
            return VirtualRange {
                start_index: 0,
                end_index: 0,
                offset_y: 0.0,
            };
        }

        let first_visible = (self.scroll_top / self.config.item_height).floor() as usize;
        let start = first_visible.saturating_sub(self.config.overscan);

        let visible = (self.config.container_height / self.config.item_height).ceil() as usize;
        let end_raw = first_visible + visible + self.config.overscan;
        let end = end_raw.min(self.config.total_items);

        VirtualRange {
            start_index: start,
            end_index: end,
            offset_y: start as f64 * self.config.item_height,
        }
    }
}

// ── VirtualListVariable (variable heights) ───────────────────────

/// Variable-height virtual list using prefix sums and binary search.
pub struct VirtualListVariable {
    heights: Vec<f64>,
    container_height: f64,
    overscan: usize,
    prefix_sums: Vec<f64>,
}

impl VirtualListVariable {
    pub fn new(heights: Vec<f64>, container_height: f64) -> Self {
        let prefix_sums = Self::build_prefix_sums(&heights);
        Self {
            heights,
            container_height,
            overscan: 2,
            prefix_sums,
        }
    }

    /// Create with explicit overscan.
    pub fn with_overscan(mut self, overscan: usize) -> Self {
        self.overscan = overscan;
        self
    }

    /// Scroll to a pixel offset.
    pub fn scroll_to(&self, scroll_top: f64) -> VirtualRange {
        let scroll = scroll_top.max(0.0);
        if self.heights.is_empty() {
            return VirtualRange {
                start_index: 0,
                end_index: 0,
                offset_y: 0.0,
            };
        }

        // Binary search for first visible item
        let first_visible = self.find_index(scroll);
        let start = first_visible.saturating_sub(self.overscan);

        // Scan forward to find end
        let bottom = scroll + self.container_height;
        let mut end = first_visible;
        while end < self.heights.len() && self.prefix_sums[end] < bottom {
            end += 1;
        }
        end = (end + self.overscan).min(self.heights.len());

        let offset_y = if start == 0 {
            0.0
        } else {
            self.prefix_sums[start]
        };

        VirtualRange {
            start_index: start,
            end_index: end,
            offset_y,
        }
    }

    /// Total scrollable height.
    pub fn total_height(&self) -> f64 {
        self.prefix_sums
            .last()
            .map(|last| last + self.heights.last().copied().unwrap_or(0.0))
            .unwrap_or(0.0)
    }

    /// Update a single item's height and rebuild prefix sums.
    pub fn update_height(&mut self, index: usize, height: f64) {
        if index < self.heights.len() {
            self.heights[index] = height;
            self.prefix_sums = Self::build_prefix_sums(&self.heights);
        }
    }

    /// Scroll to a specific index.
    pub fn scroll_to_index(&self, index: usize) -> VirtualRange {
        if index >= self.heights.len() {
            return self.scroll_to(self.total_height());
        }
        let top = self.prefix_sums[index];
        self.scroll_to(top)
    }

    // ── helpers ──────────────────────────────────────────────────

    /// Build prefix sums: prefix_sums[i] = sum of heights[0..i].
    fn build_prefix_sums(heights: &[f64]) -> Vec<f64> {
        let mut sums = Vec::with_capacity(heights.len());
        let mut acc = 0.0;
        for &h in heights {
            sums.push(acc);
            acc += h;
        }
        sums
    }

    /// Binary search for the first item whose top is <= scroll_top.
    fn find_index(&self, scroll_top: f64) -> usize {
        let mut lo = 0usize;
        let mut hi = self.prefix_sums.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.prefix_sums[mid] <= scroll_top {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.saturating_sub(1)
    }
}

// ── VirtualGridRange ─────────────────────────────────────────────

/// Range of visible cells in a 2-D grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VirtualGridRange {
    pub start_row: usize,
    pub end_row: usize,
    pub start_col: usize,
    pub end_col: usize,
    pub offset_x: f64,
    pub offset_y: f64,
}

// ── VirtualGrid ──────────────────────────────────────────────────

/// Fixed-cell virtual grid (rows × columns).
pub struct VirtualGrid {
    pub rows: usize,
    pub cols: usize,
    pub row_height: f64,
    pub col_width: f64,
    pub viewport_width: f64,
    pub viewport_height: f64,
    pub overscan: usize,
}

impl VirtualGrid {
    pub fn new(
        rows: usize,
        cols: usize,
        row_height: f64,
        col_width: f64,
        viewport_width: f64,
        viewport_height: f64,
        overscan: usize,
    ) -> Self {
        Self {
            rows,
            cols,
            row_height,
            col_width,
            viewport_width,
            viewport_height,
            overscan,
        }
    }

    /// Given scroll offsets, compute the visible cell range.
    pub fn scroll_to(&self, scroll_top: f64, scroll_left: f64) -> VirtualGridRange {
        let first_row = (scroll_top / self.row_height).floor() as usize;
        let first_col = (scroll_left / self.col_width).floor() as usize;

        let visible_rows = (self.viewport_height / self.row_height).ceil() as usize;
        let visible_cols = (self.viewport_width / self.col_width).ceil() as usize;

        let start_row = first_row.saturating_sub(self.overscan);
        let start_col = first_col.saturating_sub(self.overscan);
        let end_row = (first_row + visible_rows + self.overscan).min(self.rows);
        let end_col = (first_col + visible_cols + self.overscan).min(self.cols);

        VirtualGridRange {
            start_row,
            end_row,
            start_col,
            end_col,
            offset_x: start_col as f64 * self.col_width,
            offset_y: start_row as f64 * self.row_height,
        }
    }

    /// Total scrollable width.
    pub fn total_width(&self) -> f64 {
        self.cols as f64 * self.col_width
    }

    /// Total scrollable height.
    pub fn total_height(&self) -> f64 {
        self.rows as f64 * self.row_height
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_height_visible_range() {
        let mut vl = VirtualList::new(VirtualListConfig {
            total_items: 100,
            item_height: 40.0,
            container_height: 200.0,
            overscan: 0,
        });
        let range = vl.scroll_to(0.0);
        assert_eq!(range.start_index, 0);
        assert_eq!(range.end_index, 5); // 200/40 = 5
        assert_eq!(range.offset_y, 0.0);
    }

    #[test]
    fn scroll_to_middle() {
        let mut vl = VirtualList::new(VirtualListConfig {
            total_items: 100,
            item_height: 40.0,
            container_height: 200.0,
            overscan: 0,
        });
        let range = vl.scroll_to(400.0); // item 10
        assert_eq!(range.start_index, 10);
        assert_eq!(range.end_index, 15);
        assert_eq!(range.offset_y, 400.0);
    }

    #[test]
    fn overscan_adds_extra() {
        let mut vl = VirtualList::new(VirtualListConfig {
            total_items: 100,
            item_height: 40.0,
            container_height: 200.0,
            overscan: 2,
        });
        let range = vl.scroll_to(400.0); // first visible = 10
        assert_eq!(range.start_index, 8); // 10 - 2
        assert_eq!(range.end_index, 17); // 10 + 5 + 2
    }

    #[test]
    fn total_height_correct() {
        let vl = VirtualList::new(VirtualListConfig {
            total_items: 1000,
            item_height: 30.0,
            container_height: 300.0,
            overscan: 0,
        });
        assert_eq!(vl.total_height(), 30000.0);
    }

    #[test]
    fn scroll_to_index() {
        let mut vl = VirtualList::new(VirtualListConfig {
            total_items: 100,
            item_height: 50.0,
            container_height: 200.0,
            overscan: 0,
        });
        let range = vl.scroll_to_index(20);
        assert_eq!(range.start_index, 20);
    }

    #[test]
    fn variable_heights_binary_search() {
        // 10 items, heights 10..100
        let heights: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        let vl = VirtualListVariable::new(heights, 50.0);
        // Total height = 10+20+30+40+50+60+70+80+90+100 = 550
        assert_eq!(vl.total_height(), 550.0);

        // Scroll to 60 → first visible should be index 3 (prefix: 0,10,30,60,...)
        let range = vl.scroll_to(60.0);
        assert!(range.start_index <= 3);
        assert!(range.end_index >= 4);
    }

    #[test]
    fn variable_update_height() {
        let heights = vec![10.0, 20.0, 30.0];
        let mut vl = VirtualListVariable::new(heights, 100.0);
        assert_eq!(vl.total_height(), 60.0);
        vl.update_height(1, 50.0); // 20→50
        assert_eq!(vl.total_height(), 90.0); // 10+50+30
    }

    #[test]
    fn grid_visible_cells() {
        let grid = VirtualGrid::new(100, 50, 30.0, 100.0, 400.0, 300.0, 0);
        let range = grid.scroll_to(0.0, 0.0);
        assert_eq!(range.start_row, 0);
        assert_eq!(range.end_row, 10); // 300/30
        assert_eq!(range.start_col, 0);
        assert_eq!(range.end_col, 4); // 400/100
    }

    #[test]
    fn empty_list() {
        let mut vl = VirtualList::new(VirtualListConfig {
            total_items: 0,
            item_height: 40.0,
            container_height: 200.0,
            overscan: 0,
        });
        let range = vl.scroll_to(0.0);
        assert_eq!(range.start_index, 0);
        assert_eq!(range.end_index, 0);
        assert_eq!(vl.total_height(), 0.0);
    }

    #[test]
    fn large_list_no_allocation_per_item() {
        // 100K items — should be instant, no per-item alloc
        let mut vl = VirtualList::new(VirtualListConfig {
            total_items: 100_000,
            item_height: 20.0,
            container_height: 600.0,
            overscan: 5,
        });
        let range = vl.scroll_to(50_000.0 * 20.0);
        assert_eq!(range.start_index, 50_000 - 5);
        assert_eq!(range.end_index, 50_000 + 30 + 5); // visible = 600/20 = 30
        assert_eq!(vl.total_height(), 2_000_000.0);
    }

    #[test]
    fn grid_total_dimensions() {
        let grid = VirtualGrid::new(50, 20, 40.0, 120.0, 480.0, 400.0, 0);
        assert_eq!(grid.total_height(), 2000.0);
        assert_eq!(grid.total_width(), 2400.0);
    }

    #[test]
    fn variable_scroll_to_index() {
        let heights = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let vl = VirtualListVariable::new(heights, 40.0).with_overscan(0);
        // prefix sums: [0, 10, 30, 60, 100]
        let range = vl.scroll_to_index(2);
        // scroll_top = 30 → first visible = index 2
        assert!(range.start_index <= 2);
        assert!(range.end_index >= 3);
    }
}
