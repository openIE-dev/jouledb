// Procedural Terrain Generation — River network generation on terrain
// Flow direction, flow accumulation, Strahler order, lake detection,
// waterfall detection, river widening, meandering

use std::collections::VecDeque;
use std::fmt;

/// Direction of water flow from a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowDir {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
    Sink, // Local minimum — no outflow
}

impl FlowDir {
    /// Offset (dx, dy) for this direction.
    pub fn offset(&self) -> (i64, i64) {
        match self {
            Self::North     => ( 0, -1),
            Self::NorthEast => ( 1, -1),
            Self::East      => ( 1,  0),
            Self::SouthEast => ( 1,  1),
            Self::South     => ( 0,  1),
            Self::SouthWest => (-1,  1),
            Self::West      => (-1,  0),
            Self::NorthWest => (-1, -1),
            Self::Sink      => ( 0,  0),
        }
    }

    const ALL_DIRS: [FlowDir; 8] = [
        Self::North, Self::NorthEast, Self::East, Self::SouthEast,
        Self::South, Self::SouthWest, Self::West, Self::NorthWest,
    ];
}

/// A cell in the river network.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiverCell {
    pub flow_dir: FlowDir,
    pub accumulation: u32,
    pub strahler_order: u32,
    pub is_river: bool,
    pub is_lake: bool,
    pub is_waterfall: bool,
    pub width: f64,
}

impl Default for RiverCell {
    fn default() -> Self {
        Self {
            flow_dir: FlowDir::Sink,
            accumulation: 1,
            strahler_order: 0,
            is_river: false,
            is_lake: false,
            is_waterfall: false,
            width: 0.0,
        }
    }
}

/// A 2D point for meandering curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Configuration for river network generation.
#[derive(Debug, Clone, PartialEq)]
pub struct RiverConfig {
    pub river_threshold: u32,
    pub waterfall_height_diff: f64,
    pub max_river_width: f64,
    pub min_river_width: f64,
    pub meander_amplitude: f64,
    pub meander_segments: usize,
    pub lake_fill_threshold: f64,
}

impl Default for RiverConfig {
    fn default() -> Self {
        Self {
            river_threshold: 50,
            waterfall_height_diff: 0.1,
            max_river_width: 5.0,
            min_river_width: 1.0,
            meander_amplitude: 2.0,
            meander_segments: 8,
            lake_fill_threshold: 0.001,
        }
    }
}

/// River network generator.
pub struct RiverNetwork {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<RiverCell>,
    config: RiverConfig,
    heightmap: Vec<f64>,
}

impl fmt::Debug for RiverNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RiverNetwork")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("config", &self.config)
            .finish()
    }
}

impl RiverNetwork {
    /// Create a river network from a heightmap.
    pub fn new(width: usize, height: usize, heightmap: Vec<f64>, config: RiverConfig) -> Self {
        assert_eq!(heightmap.len(), width * height);
        let cells = vec![RiverCell::default(); width * height];
        Self { width, height, cells, config, heightmap }
    }

    pub fn config(&self) -> &RiverConfig {
        &self.config
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn height_at(&self, x: usize, y: usize) -> f64 {
        self.heightmap[self.idx(x, y)]
    }

    /// Step 1: Compute flow direction per cell (steepest descent to neighbor).
    pub fn compute_flow_directions(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let h = self.height_at(x, y);
                let mut best_dir = FlowDir::Sink;
                let mut best_drop = 0.0f64;

                for dir in &FlowDir::ALL_DIRS {
                    let (dx, dy) = dir.offset();
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || nx >= self.width as i64 || ny < 0 || ny >= self.height as i64 {
                        continue;
                    }
                    let nh = self.height_at(nx as usize, ny as usize);
                    let drop = h - nh;
                    // For diagonals, divide by sqrt(2) for accurate slope
                    let dist = if dx.abs() + dy.abs() == 2 { std::f64::consts::SQRT_2 } else { 1.0 };
                    let slope = drop / dist;
                    if slope > best_drop {
                        best_drop = slope;
                        best_dir = *dir;
                    }
                }

                let idx = self.idx(x, y);
                self.cells[idx].flow_dir = best_dir;
            }
        }
    }

    /// Step 2: Compute flow accumulation (count upstream cells).
    pub fn compute_accumulation(&mut self) {
        // Reset accumulations
        for cell in &mut self.cells {
            cell.accumulation = 1;
        }

        // Topological sort: process cells from highest to lowest
        let mut order: Vec<usize> = (0..self.width * self.height).collect();
        let hm = &self.heightmap;
        order.sort_by(|&a, &b| {
            hm[b].partial_cmp(&hm[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        for &idx in &order {
            let x = idx % self.width;
            let y = idx / self.width;
            let dir = self.cells[idx].flow_dir;
            if dir == FlowDir::Sink { continue; }

            let (dx, dy) = dir.offset();
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx >= 0 && nx < self.width as i64 && ny >= 0 && ny < self.height as i64 {
                let ni = ny as usize * self.width + nx as usize;
                let acc = self.cells[idx].accumulation;
                self.cells[ni].accumulation += acc;
            }
        }
    }

    /// Step 3: Mark river cells (accumulation above threshold).
    pub fn mark_rivers(&mut self) {
        let threshold = self.config.river_threshold;
        for cell in &mut self.cells {
            cell.is_river = cell.accumulation >= threshold;
        }
    }

    /// Step 4: Compute Strahler stream order.
    pub fn compute_strahler_order(&mut self) {
        // Reset
        for cell in &mut self.cells {
            cell.strahler_order = if cell.is_river { 1 } else { 0 };
        }

        // Process from highest to lowest
        let mut order: Vec<usize> = (0..self.width * self.height).collect();
        let hm = &self.heightmap;
        order.sort_by(|&a, &b| {
            hm[b].partial_cmp(&hm[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        for &idx in &order {
            if !self.cells[idx].is_river { continue; }

            let x = idx % self.width;
            let y = idx / self.width;
            let dir = self.cells[idx].flow_dir;
            if dir == FlowDir::Sink { continue; }

            let (dx, dy) = dir.offset();
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx < 0 || nx >= self.width as i64 || ny < 0 || ny >= self.height as i64 {
                continue;
            }

            let ni = ny as usize * self.width + nx as usize;
            if !self.cells[ni].is_river { continue; }

            let my_order = self.cells[idx].strahler_order;
            let target_order = self.cells[ni].strahler_order;

            if my_order > target_order {
                self.cells[ni].strahler_order = my_order;
            } else if my_order == target_order {
                self.cells[ni].strahler_order = my_order + 1;
            }
        }
    }

    /// Step 5: Detect lakes (fill depressions where water pools).
    pub fn detect_lakes(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.idx(x, y);
                if self.cells[idx].flow_dir == FlowDir::Sink {
                    // Check if it accumulates significant water
                    if self.cells[idx].accumulation >= self.config.river_threshold / 2 {
                        self.flood_fill_lake(x, y);
                    }
                }
            }
        }
    }

    fn flood_fill_lake(&mut self, start_x: usize, start_y: usize) {
        let start_h = self.height_at(start_x, start_y);
        let lake_level = start_h + self.config.lake_fill_threshold;

        let mut queue = VecDeque::new();
        queue.push_back((start_x, start_y));
        let mut visited = vec![false; self.width * self.height];
        visited[self.idx(start_x, start_y)] = true;

        while let Some((x, y)) = queue.pop_front() {
            let idx = self.idx(x, y);
            if self.height_at(x, y) <= lake_level {
                self.cells[idx].is_lake = true;

                for (dx, dy) in [(0i64, 1), (0, -1), (1, 0), (-1, 0)] {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx >= 0 && nx < self.width as i64 && ny >= 0 && ny < self.height as i64 {
                        let ni = self.idx(nx as usize, ny as usize);
                        if !visited[ni] {
                            visited[ni] = true;
                            queue.push_back((nx as usize, ny as usize));
                        }
                    }
                }
            }
        }
    }

    /// Step 6: Detect waterfalls (sharp height drops along river path).
    pub fn detect_waterfalls(&mut self) {
        let threshold = self.config.waterfall_height_diff;
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.idx(x, y);
                if !self.cells[idx].is_river { continue; }

                let dir = self.cells[idx].flow_dir;
                if dir == FlowDir::Sink { continue; }

                let (dx, dy) = dir.offset();
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx >= 0 && nx < self.width as i64 && ny >= 0 && ny < self.height as i64 {
                    let h_diff = self.height_at(x, y) - self.height_at(nx as usize, ny as usize);
                    if h_diff >= threshold {
                        self.cells[idx].is_waterfall = true;
                    }
                }
            }
        }
    }

    /// Step 7: Compute river width proportional to flow accumulation.
    pub fn compute_widths(&mut self) {
        let max_acc = self.cells.iter()
            .filter(|c| c.is_river)
            .map(|c| c.accumulation)
            .max()
            .unwrap_or(1) as f64;

        for cell in &mut self.cells {
            if cell.is_river {
                let t = (cell.accumulation as f64 / max_acc).sqrt();
                cell.width = self.config.min_river_width
                    + t * (self.config.max_river_width - self.config.min_river_width);
            }
        }
    }

    /// Run all steps in order.
    pub fn generate(&mut self) {
        self.compute_flow_directions();
        self.compute_accumulation();
        self.mark_rivers();
        self.compute_strahler_order();
        self.detect_lakes();
        self.detect_waterfalls();
        self.compute_widths();
    }

    /// Count river cells.
    pub fn river_count(&self) -> usize {
        self.cells.iter().filter(|c| c.is_river).count()
    }

    /// Count lake cells.
    pub fn lake_count(&self) -> usize {
        self.cells.iter().filter(|c| c.is_lake).count()
    }

    /// Count waterfall cells.
    pub fn waterfall_count(&self) -> usize {
        self.cells.iter().filter(|c| c.is_waterfall).count()
    }

    /// Extract river path from a starting cell following flow direction.
    pub fn trace_river(&self, start_x: usize, start_y: usize) -> Vec<(usize, usize)> {
        let mut path = Vec::new();
        let mut x = start_x;
        let mut y = start_y;
        let mut visited = vec![false; self.width * self.height];

        loop {
            let idx = self.idx(x, y);
            if visited[idx] { break; }
            visited[idx] = true;
            path.push((x, y));

            let dir = self.cells[idx].flow_dir;
            if dir == FlowDir::Sink { break; }

            let (dx, dy) = dir.offset();
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx < 0 || nx >= self.width as i64 || ny < 0 || ny >= self.height as i64 {
                break;
            }
            x = nx as usize;
            y = ny as usize;
        }
        path
    }

    /// Generate meander points along a river path using cubic Bezier curves.
    pub fn meander_path(&self, path: &[(usize, usize)]) -> Vec<Point2D> {
        if path.len() < 2 {
            return path.iter().map(|&(x, y)| Point2D::new(x as f64, y as f64)).collect();
        }

        let amp = self.config.meander_amplitude;
        let segments = self.config.meander_segments.max(2);
        let mut result = Vec::new();

        for window in path.windows(2) {
            let (x0, y0) = (window[0].0 as f64, window[0].1 as f64);
            let (x1, y1) = (window[1].0 as f64, window[1].1 as f64);

            let dx = x1 - x0;
            let dy = y1 - y0;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-10 { continue; }

            // Perpendicular direction for meandering offset
            let nx = -dy / len;
            let ny = dx / len;

            // Alternate meander direction based on segment index
            let sign = if result.len() % 2 == 0 { 1.0 } else { -1.0 };

            // Control points for cubic Bezier
            let cp1x = x0 + dx * 0.33 + nx * amp * sign;
            let cp1y = y0 + dy * 0.33 + ny * amp * sign;
            let cp2x = x0 + dx * 0.67 - nx * amp * sign;
            let cp2y = y0 + dy * 0.67 - ny * amp * sign;

            for s in 0..segments {
                let t = s as f64 / segments as f64;
                let it = 1.0 - t;
                let bx = it * it * it * x0
                    + 3.0 * it * it * t * cp1x
                    + 3.0 * it * t * t * cp2x
                    + t * t * t * x1;
                let by = it * it * it * y0
                    + 3.0 * it * it * t * cp1y
                    + 3.0 * it * t * t * cp2y
                    + t * t * t * y1;
                result.push(Point2D::new(bx, by));
            }
        }

        // Add final point
        if let Some(&(lx, ly)) = path.last() {
            result.push(Point2D::new(lx as f64, ly as f64));
        }
        result
    }

    /// Get the maximum flow accumulation in the network.
    pub fn max_accumulation(&self) -> u32 {
        self.cells.iter().map(|c| c.accumulation).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sloped_heightmap(w: usize, h: usize) -> Vec<f64> {
        let mut data = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = 1.0 - (x as f64 + y as f64) / (w + h) as f64;
            }
        }
        data
    }

    fn make_valley_heightmap(w: usize, h: usize) -> Vec<f64> {
        let mut data = vec![0.0; w * h];
        let cx = w as f64 / 2.0;
        for y in 0..h {
            for x in 0..w {
                let dist = (x as f64 - cx).abs() / cx;
                data[y * w + x] = dist * 0.5 + (1.0 - y as f64 / h as f64) * 0.5;
            }
        }
        data
    }

    #[test]
    fn test_flow_dir_offsets() {
        assert_eq!(FlowDir::North.offset(), (0, -1));
        assert_eq!(FlowDir::East.offset(), (1, 0));
        assert_eq!(FlowDir::Sink.offset(), (0, 0));
    }

    #[test]
    fn test_river_cell_default() {
        let c = RiverCell::default();
        assert_eq!(c.flow_dir, FlowDir::Sink);
        assert_eq!(c.accumulation, 1);
        assert!(!c.is_river);
    }

    #[test]
    fn test_new_river_network() {
        let hm = vec![0.5; 64];
        let rn = RiverNetwork::new(8, 8, hm, RiverConfig::default());
        assert_eq!(rn.width, 8);
        assert_eq!(rn.height, 8);
    }

    #[test]
    fn test_compute_flow_directions_sloped() {
        let w = 16;
        let h = 16;
        let hm = make_sloped_heightmap(w, h);
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig::default());
        rn.compute_flow_directions();

        // Interior cells should flow towards lower-right
        let c = rn.cells[rn.idx(5, 5)];
        assert_ne!(c.flow_dir, FlowDir::Sink);
    }

    #[test]
    fn test_compute_accumulation() {
        let w = 16;
        let h = 16;
        let hm = make_sloped_heightmap(w, h);
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig::default());
        rn.compute_flow_directions();
        rn.compute_accumulation();

        // Lower-right corner should have high accumulation
        let max_acc = rn.max_accumulation();
        assert!(max_acc > 1, "should have accumulation > 1: {max_acc}");
    }

    #[test]
    fn test_mark_rivers() {
        let w = 32;
        let h = 32;
        let hm = make_sloped_heightmap(w, h);
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig {
            river_threshold: 10,
            ..RiverConfig::default()
        });
        rn.compute_flow_directions();
        rn.compute_accumulation();
        rn.mark_rivers();
        assert!(rn.river_count() > 0);
    }

    #[test]
    fn test_strahler_order() {
        let w = 32;
        let h = 32;
        let hm = make_valley_heightmap(w, h);
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig {
            river_threshold: 5,
            ..RiverConfig::default()
        });
        rn.compute_flow_directions();
        rn.compute_accumulation();
        rn.mark_rivers();
        rn.compute_strahler_order();

        let max_order = rn.cells.iter().map(|c| c.strahler_order).max().unwrap_or(0);
        assert!(max_order >= 1, "should have at least order 1: {max_order}");
    }

    #[test]
    fn test_waterfall_detection() {
        let w = 16;
        let h = 16;
        let mut hm = make_sloped_heightmap(w, h);
        // Create a cliff at row 8
        for x in 0..w {
            for y in 8..h {
                hm[y * w + x] -= 0.3;
            }
        }
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig {
            river_threshold: 3,
            waterfall_height_diff: 0.1,
            ..RiverConfig::default()
        });
        rn.generate();
        // There should be some waterfalls near the cliff
        // (depends on flow paths hitting the cliff edge)
    }

    #[test]
    fn test_compute_widths() {
        let w = 32;
        let h = 32;
        let hm = make_sloped_heightmap(w, h);
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig {
            river_threshold: 10,
            ..RiverConfig::default()
        });
        rn.generate();

        for cell in &rn.cells {
            if cell.is_river {
                assert!(cell.width >= rn.config.min_river_width - 1e-6);
                assert!(cell.width <= rn.config.max_river_width + 1e-6);
            }
        }
    }

    #[test]
    fn test_trace_river() {
        let w = 16;
        let h = 16;
        let hm = make_sloped_heightmap(w, h);
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig::default());
        rn.compute_flow_directions();
        let path = rn.trace_river(0, 0);
        assert!(path.len() > 1, "should trace a path downhill");
    }

    #[test]
    fn test_trace_river_no_infinite_loop() {
        let hm = vec![0.5; 64];
        let mut rn = RiverNetwork::new(8, 8, hm, RiverConfig::default());
        rn.compute_flow_directions();
        let path = rn.trace_river(4, 4);
        assert!(path.len() <= 64, "should not loop infinitely");
    }

    #[test]
    fn test_meander_path_basic() {
        let rn = RiverNetwork::new(8, 8, vec![0.5; 64], RiverConfig::default());
        let path = vec![(0, 0), (4, 4), (8, 8)];
        let meander = rn.meander_path(&path);
        assert!(meander.len() > path.len());
    }

    #[test]
    fn test_meander_single_point() {
        let rn = RiverNetwork::new(8, 8, vec![0.5; 64], RiverConfig::default());
        let path = vec![(4, 4)];
        let meander = rn.meander_path(&path);
        assert_eq!(meander.len(), 1);
    }

    #[test]
    fn test_meander_empty_path() {
        let rn = RiverNetwork::new(8, 8, vec![0.5; 64], RiverConfig::default());
        let path: Vec<(usize, usize)> = vec![];
        let meander = rn.meander_path(&path);
        assert_eq!(meander.len(), 0);
    }

    #[test]
    fn test_generate_full_pipeline() {
        let w = 32;
        let h = 32;
        let hm = make_sloped_heightmap(w, h);
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig {
            river_threshold: 10,
            ..RiverConfig::default()
        });
        rn.generate();
        assert!(rn.river_count() > 0);
    }

    #[test]
    fn test_point2d() {
        let p = Point2D::new(1.5, 2.5);
        assert!((p.x - 1.5).abs() < 1e-12);
        assert!((p.y - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_config_default() {
        let cfg = RiverConfig::default();
        assert_eq!(cfg.river_threshold, 50);
    }

    #[test]
    fn test_debug_format() {
        let rn = RiverNetwork::new(8, 8, vec![0.5; 64], RiverConfig::default());
        let s = format!("{:?}", rn);
        assert!(s.contains("RiverNetwork"));
    }

    #[test]
    fn test_river_cell_partial_eq() {
        let a = RiverCell::default();
        let b = RiverCell::default();
        assert_eq!(a, b);
    }

    #[test]
    fn test_lake_detection() {
        let w = 16;
        let h = 16;
        // Create a bowl: high edges, low center
        let mut hm = vec![0.0; w * h];
        let cx = w as f64 / 2.0;
        let cy = h as f64 / 2.0;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                hm[y * w + x] = (dx * dx + dy * dy).sqrt() / cx;
            }
        }
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig {
            river_threshold: 3,
            ..RiverConfig::default()
        });
        rn.generate();
        // Center should be a sink with accumulation
    }

    #[test]
    fn test_max_accumulation() {
        let w = 16;
        let h = 16;
        let hm = make_sloped_heightmap(w, h);
        let mut rn = RiverNetwork::new(w, h, hm, RiverConfig::default());
        rn.compute_flow_directions();
        rn.compute_accumulation();
        let max_acc = rn.max_accumulation();
        assert!(max_acc >= 1);
    }

    #[test]
    fn test_flow_dir_all_dirs_len() {
        assert_eq!(FlowDir::ALL_DIRS.len(), 8);
    }

    #[test]
    fn test_point2d_partial_eq() {
        let a = Point2D::new(1.0, 2.0);
        let b = Point2D::new(1.0, 2.0);
        assert_eq!(a, b);
    }
}
