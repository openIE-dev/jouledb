//! CSS Grid layout engine: track sizing, item placement, and final rect computation.
//!
//! Pure math — no browser dependency. Resolves grid templates, places items
//! (explicit + auto-placement), and computes pixel rects for each item.

// ── Track Sizing ────────────────────────────────────────────────

/// How a grid track (column or row) is sized.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackSize {
    /// Fixed pixel width.
    Fixed(f64),
    /// Fractional unit — share of remaining space.
    Fr(f64),
    /// `minmax(min, max)` — clamp between two sizes.
    MinMax(Box<TrackSize>, Box<TrackSize>),
    /// `auto` — fit content.
    Auto,
}

impl TrackSize {
    /// The minimum contribution of this track.
    fn min_size(&self) -> f64 {
        match self {
            TrackSize::Fixed(px) => *px,
            TrackSize::Fr(_) => 0.0,
            TrackSize::Auto => 0.0,
            TrackSize::MinMax(min, _) => min.min_size(),
        }
    }

    /// Whether this track participates in fr distribution.
    fn is_flexible(&self) -> bool {
        match self {
            TrackSize::Fr(_) => true,
            TrackSize::MinMax(_, max) => max.is_flexible(),
            _ => false,
        }
    }

    fn fr_value(&self) -> f64 {
        match self {
            TrackSize::Fr(v) => *v,
            TrackSize::MinMax(_, max) => max.fr_value(),
            _ => 0.0,
        }
    }
}

// ── Grid Template ───────────────────────────────────────────────

/// Definition of a grid container.
#[derive(Debug, Clone)]
pub struct GridTemplate {
    pub columns: Vec<TrackSize>,
    pub rows: Vec<TrackSize>,
    pub column_gap: f64,
    pub row_gap: f64,
}

impl GridTemplate {
    pub fn new(columns: Vec<TrackSize>, rows: Vec<TrackSize>) -> Self {
        Self { columns, rows, column_gap: 0.0, row_gap: 0.0 }
    }

    pub fn with_gap(mut self, column_gap: f64, row_gap: f64) -> Self {
        self.column_gap = column_gap;
        self.row_gap = row_gap;
        self
    }
}

// ── Grid Item ───────────────────────────────────────────────────

/// Placement of an item within the grid.
#[derive(Debug, Clone)]
pub struct GridItem {
    /// 1-based line numbers (CSS convention). 0 = auto.
    pub col_start: usize,
    pub col_end: usize,
    pub row_start: usize,
    pub row_end: usize,
}

impl GridItem {
    pub fn new(col_start: usize, col_end: usize, row_start: usize, row_end: usize) -> Self {
        Self { col_start, col_end, row_start, row_end }
    }

    /// Automatic placement — will be resolved later.
    pub fn auto_place() -> Self {
        Self { col_start: 0, col_end: 0, row_start: 0, row_end: 0 }
    }

    fn is_auto(&self) -> bool {
        self.col_start == 0 && self.col_end == 0 && self.row_start == 0 && self.row_end == 0
    }
}

// ── Computed Rect ───────────────────────────────────────────────

/// Final computed rectangle of a grid item in pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

// ── Track Resolution ────────────────────────────────────────────

/// Resolve a list of track sizes into pixel widths given available space.
pub fn resolve_tracks(tracks: &[TrackSize], available: f64, gap: f64) -> Vec<f64> {
    if tracks.is_empty() {
        return vec![];
    }
    let total_gap = gap * (tracks.len().saturating_sub(1)) as f64;
    let usable = (available - total_gap).max(0.0);

    // Pass 1: allocate fixed / auto tracks
    let mut sizes: Vec<f64> = tracks.iter().map(|t| t.min_size()).collect();
    let fixed_total: f64 = tracks
        .iter()
        .zip(sizes.iter())
        .filter(|(t, _)| !t.is_flexible())
        .map(|(_, s)| *s)
        .sum();

    // Pass 2: auto tracks get 0 (content-based sizing deferred)
    // In headless mode, auto = 0 unless flexible.

    let remaining = (usable - fixed_total).max(0.0);

    // Pass 3: distribute remaining to fr units
    let total_fr: f64 = tracks.iter().map(|t| t.fr_value()).sum();
    if total_fr > 0.0 {
        let per_fr = remaining / total_fr;
        for (i, track) in tracks.iter().enumerate() {
            if track.is_flexible() {
                let fr = track.fr_value();
                let min = track.min_size();
                sizes[i] = (per_fr * fr).max(min);
            }
        }
    }

    // Pass 4: auto tracks share leftover after fr
    let auto_count = tracks.iter().filter(|t| matches!(t, TrackSize::Auto)).count();
    if auto_count > 0 && total_fr == 0.0 {
        let auto_share = remaining / auto_count as f64;
        for (i, track) in tracks.iter().enumerate() {
            if matches!(track, TrackSize::Auto) {
                sizes[i] = auto_share.max(0.0);
            }
        }
    }

    sizes
}

// ── Auto-placement ──────────────────────────────────────────────

fn auto_place_items(items: &mut [GridItem], num_cols: usize, num_rows: usize) {
    // Bitmap of occupied cells
    let effective_rows = num_rows.max(items.len());
    let mut occupied = vec![vec![false; num_cols]; effective_rows];

    // Mark explicitly placed items
    for item in items.iter() {
        if !item.is_auto() {
            let rs = item.row_start.saturating_sub(1);
            let re = item.row_end.saturating_sub(1);
            let cs = item.col_start.saturating_sub(1);
            let ce = item.col_end.saturating_sub(1);
            for r in rs..re.min(effective_rows) {
                for c in cs..ce.min(num_cols) {
                    occupied[r][c] = true;
                }
            }
        }
    }

    // Place auto items
    let mut cursor_row = 0usize;
    let mut cursor_col = 0usize;
    for item in items.iter_mut() {
        if !item.is_auto() {
            continue;
        }
        // Find next free cell
        loop {
            if cursor_row >= effective_rows {
                // Extend grid implicitly
                break;
            }
            if cursor_col >= num_cols {
                cursor_col = 0;
                cursor_row += 1;
                continue;
            }
            if !occupied.get(cursor_row).and_then(|r| r.get(cursor_col)).copied().unwrap_or(true) {
                break;
            }
            cursor_col += 1;
        }
        item.col_start = cursor_col + 1;
        item.col_end = cursor_col + 2;
        item.row_start = cursor_row + 1;
        item.row_end = cursor_row + 2;
        if cursor_row < effective_rows && cursor_col < num_cols {
            occupied[cursor_row][cursor_col] = true;
        }
        cursor_col += 1;
    }
}

// ── Layout ──────────────────────────────────────────────────────

/// Compute rects for all grid items.
pub fn layout(
    template: &GridTemplate,
    items: &[GridItem],
    container_width: f64,
    container_height: f64,
) -> Vec<Rect> {
    let col_sizes = resolve_tracks(&template.columns, container_width, template.column_gap);
    let row_sizes = resolve_tracks(&template.rows, container_height, template.row_gap);

    let mut items = items.to_vec();
    auto_place_items(&mut items, col_sizes.len(), row_sizes.len());

    // Precompute offsets
    let col_offsets = track_offsets(&col_sizes, template.column_gap);
    let row_offsets = track_offsets(&row_sizes, template.row_gap);

    items
        .iter()
        .map(|item| {
            let cs = item.col_start.saturating_sub(1).min(col_sizes.len());
            let ce = item.col_end.saturating_sub(1).min(col_sizes.len());
            let rs = item.row_start.saturating_sub(1).min(row_sizes.len());
            let re = item.row_end.saturating_sub(1).min(row_sizes.len());

            let x = col_offsets.get(cs).copied().unwrap_or(0.0);
            let x_end = if ce > 0 {
                col_offsets.get(ce - 1).copied().unwrap_or(0.0)
                    + col_sizes.get(ce - 1).copied().unwrap_or(0.0)
            } else {
                x
            };

            let y = row_offsets.get(rs).copied().unwrap_or(0.0);
            let y_end = if re > 0 {
                row_offsets.get(re - 1).copied().unwrap_or(0.0)
                    + row_sizes.get(re - 1).copied().unwrap_or(0.0)
            } else {
                y
            };

            Rect {
                x,
                y,
                width: (x_end - x).max(0.0),
                height: (y_end - y).max(0.0),
            }
        })
        .collect()
}

fn track_offsets(sizes: &[f64], gap: f64) -> Vec<f64> {
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut pos = 0.0;
    for (i, size) in sizes.iter().enumerate() {
        offsets.push(pos);
        pos += size;
        if i + 1 < sizes.len() {
            pos += gap;
        }
    }
    offsets
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.001
    }

    #[test]
    fn resolve_fixed_tracks() {
        let sizes = resolve_tracks(
            &[TrackSize::Fixed(100.0), TrackSize::Fixed(200.0)],
            400.0,
            0.0,
        );
        assert!(approx(sizes[0], 100.0));
        assert!(approx(sizes[1], 200.0));
    }

    #[test]
    fn resolve_fr_tracks() {
        let sizes = resolve_tracks(
            &[TrackSize::Fr(1.0), TrackSize::Fr(2.0)],
            300.0,
            0.0,
        );
        assert!(approx(sizes[0], 100.0));
        assert!(approx(sizes[1], 200.0));
    }

    #[test]
    fn resolve_mixed_fixed_and_fr() {
        let sizes = resolve_tracks(
            &[TrackSize::Fixed(100.0), TrackSize::Fr(1.0), TrackSize::Fr(1.0)],
            500.0,
            0.0,
        );
        assert!(approx(sizes[0], 100.0));
        assert!(approx(sizes[1], 200.0));
        assert!(approx(sizes[2], 200.0));
    }

    #[test]
    fn resolve_with_gap() {
        // 2 columns, 10px gap → usable = 300 - 10 = 290
        let sizes = resolve_tracks(
            &[TrackSize::Fr(1.0), TrackSize::Fr(1.0)],
            300.0,
            10.0,
        );
        assert!(approx(sizes[0], 145.0));
        assert!(approx(sizes[1], 145.0));
    }

    #[test]
    fn resolve_auto_tracks() {
        let sizes = resolve_tracks(
            &[TrackSize::Auto, TrackSize::Auto],
            200.0,
            0.0,
        );
        assert!(approx(sizes[0], 100.0));
        assert!(approx(sizes[1], 100.0));
    }

    #[test]
    fn resolve_minmax_tracks() {
        // minmax(50, 1fr) with 300px → fr gets 300
        let sizes = resolve_tracks(
            &[TrackSize::MinMax(
                Box::new(TrackSize::Fixed(50.0)),
                Box::new(TrackSize::Fr(1.0)),
            )],
            300.0,
            0.0,
        );
        assert!(approx(sizes[0], 300.0));
    }

    #[test]
    fn layout_explicit_placement() {
        let template = GridTemplate::new(
            vec![TrackSize::Fixed(100.0), TrackSize::Fixed(100.0)],
            vec![TrackSize::Fixed(50.0), TrackSize::Fixed(50.0)],
        );
        let items = vec![
            GridItem::new(1, 2, 1, 2), // top-left
            GridItem::new(2, 3, 2, 3), // bottom-right
        ];
        let rects = layout(&template, &items, 200.0, 100.0);
        assert!(approx(rects[0].x, 0.0));
        assert!(approx(rects[0].y, 0.0));
        assert!(approx(rects[0].width, 100.0));
        assert!(approx(rects[0].height, 50.0));

        assert!(approx(rects[1].x, 100.0));
        assert!(approx(rects[1].y, 50.0));
        assert!(approx(rects[1].width, 100.0));
        assert!(approx(rects[1].height, 50.0));
    }

    #[test]
    fn layout_with_gap() {
        let template = GridTemplate::new(
            vec![TrackSize::Fr(1.0), TrackSize::Fr(1.0)],
            vec![TrackSize::Fixed(50.0)],
        ).with_gap(10.0, 0.0);

        let items = vec![
            GridItem::new(1, 2, 1, 2),
            GridItem::new(2, 3, 1, 2),
        ];
        let rects = layout(&template, &items, 210.0, 50.0);
        assert!(approx(rects[0].x, 0.0));
        assert!(approx(rects[0].width, 100.0));
        assert!(approx(rects[1].x, 110.0));
        assert!(approx(rects[1].width, 100.0));
    }

    #[test]
    fn layout_spanning_item() {
        let template = GridTemplate::new(
            vec![TrackSize::Fixed(100.0), TrackSize::Fixed(100.0)],
            vec![TrackSize::Fixed(50.0)],
        );
        let items = vec![
            GridItem::new(1, 3, 1, 2), // spans both columns
        ];
        let rects = layout(&template, &items, 200.0, 50.0);
        assert!(approx(rects[0].width, 200.0));
    }

    #[test]
    fn auto_placement_fills_grid() {
        let template = GridTemplate::new(
            vec![TrackSize::Fixed(100.0), TrackSize::Fixed(100.0)],
            vec![TrackSize::Fixed(50.0), TrackSize::Fixed(50.0)],
        );
        let items = vec![
            GridItem::auto_place(),
            GridItem::auto_place(),
            GridItem::auto_place(),
        ];
        let rects = layout(&template, &items, 200.0, 100.0);
        // Item 0: (0,0), Item 1: (100,0), Item 2: (0,50)
        assert!(approx(rects[0].x, 0.0));
        assert!(approx(rects[0].y, 0.0));
        assert!(approx(rects[1].x, 100.0));
        assert!(approx(rects[1].y, 0.0));
        assert!(approx(rects[2].x, 0.0));
        assert!(approx(rects[2].y, 50.0));
    }

    #[test]
    fn auto_placement_skips_occupied() {
        let template = GridTemplate::new(
            vec![TrackSize::Fixed(100.0), TrackSize::Fixed(100.0)],
            vec![TrackSize::Fixed(50.0), TrackSize::Fixed(50.0)],
        );
        let items = vec![
            GridItem::new(1, 2, 1, 2), // occupies (0,0)
            GridItem::auto_place(),     // should go to (1,0) = col2, row1
        ];
        let rects = layout(&template, &items, 200.0, 100.0);
        assert!(approx(rects[1].x, 100.0));
        assert!(approx(rects[1].y, 0.0));
    }

    #[test]
    fn empty_grid() {
        let template = GridTemplate::new(vec![], vec![]);
        let rects = layout(&template, &[], 100.0, 100.0);
        assert!(rects.is_empty());
    }
}
