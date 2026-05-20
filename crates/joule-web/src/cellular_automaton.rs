//! Cellular automata — 1D/2D grid, rule-based evolution, pattern detection.
//!
//! Replaces p5.js / NetLogo / Wolfram.js cellular automaton libraries with
//! pure Rust. Supports Game of Life, 1D elementary automata (e.g. Rule 110),
//! configurable neighborhoods (Moore / Von Neumann), boundary conditions
//! (wrap / dead), generation stepping, pattern detection, and population stats.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for cellular automata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaError {
    /// Grid dimensions are zero.
    ZeroDimension,
    /// Coordinates out of bounds.
    OutOfBounds { x: usize, y: usize, width: usize, height: usize },
    /// Invalid rule number for elementary automaton.
    InvalidRule(u16),
    /// Grid size mismatch.
    SizeMismatch { expected: usize, got: usize },
}

impl fmt::Display for CaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "grid dimensions must be non-zero"),
            Self::OutOfBounds { x, y, width, height } => {
                write!(f, "({x}, {y}) out of bounds for {width}x{height} grid")
            }
            Self::InvalidRule(r) => write!(f, "invalid rule number: {r} (must be 0..=255)"),
            Self::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for CaError {}

// ── Types ───────────────────────────────────────────────────────

/// Cell state: alive (1) or dead (0).
pub type CellState = u8;

/// Neighborhood type for 2D automata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neighborhood {
    /// 8 neighbors (orthogonal + diagonal).
    Moore,
    /// 4 neighbors (orthogonal only).
    VonNeumann,
}

/// Boundary condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Edges wrap around (torus).
    Wrap,
    /// Cells outside the grid are dead.
    Dead,
}

// ── 1D Elementary Automaton ─────────────────────────────────────

/// A 1D elementary cellular automaton (256 rules).
#[derive(Debug, Clone)]
pub struct ElementaryAutomaton {
    cells: Vec<CellState>,
    rule: u8,
    boundary: Boundary,
    generation: u64,
    history: Vec<Vec<CellState>>,
    max_history: usize,
}

impl ElementaryAutomaton {
    /// Create a new 1D automaton with given width and rule.
    pub fn new(width: usize, rule: u8, boundary: Boundary) -> Result<Self, CaError> {
        if width == 0 {
            return Err(CaError::ZeroDimension);
        }
        Ok(Self {
            cells: vec![0; width],
            rule,
            boundary,
            generation: 0,
            history: Vec::new(),
            max_history: 256,
        })
    }

    /// Create with a single seed in the center.
    pub fn with_center_seed(width: usize, rule: u8, boundary: Boundary) -> Result<Self, CaError> {
        let mut ca = Self::new(width, rule, boundary)?;
        ca.cells[width / 2] = 1;
        Ok(ca)
    }

    /// Set a cell state at position.
    pub fn set(&mut self, pos: usize, state: CellState) -> Result<(), CaError> {
        if pos >= self.cells.len() {
            return Err(CaError::OutOfBounds {
                x: pos, y: 0, width: self.cells.len(), height: 1,
            });
        }
        self.cells[pos] = state;
        Ok(())
    }

    /// Get the cell state.
    pub fn get(&self, pos: usize) -> Option<CellState> {
        self.cells.get(pos).copied()
    }

    /// Width of the automaton.
    pub fn width(&self) -> usize {
        self.cells.len()
    }

    /// Current generation count.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Current cells as a slice.
    pub fn cells(&self) -> &[CellState] {
        &self.cells
    }

    /// History of past generations.
    pub fn history(&self) -> &[Vec<CellState>] {
        &self.history
    }

    /// Population count (alive cells).
    pub fn population(&self) -> usize {
        self.cells.iter().filter(|&&c| c != 0).count()
    }

    /// Advance by one generation using the elementary rule.
    pub fn step(&mut self) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(self.cells.clone());

        let w = self.cells.len();
        let old = self.cells.clone();
        for i in 0..w {
            let left = if i == 0 {
                match self.boundary {
                    Boundary::Wrap => old[w - 1],
                    Boundary::Dead => 0,
                }
            } else {
                old[i - 1]
            };
            let center = old[i];
            let right = if i == w - 1 {
                match self.boundary {
                    Boundary::Wrap => old[0],
                    Boundary::Dead => 0,
                }
            } else {
                old[i + 1]
            };
            let index = ((left & 1) << 2) | ((center & 1) << 1) | (right & 1);
            self.cells[i] = (self.rule >> index) & 1;
        }
        self.generation += 1;
    }

    /// Step multiple generations.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Detect if the automaton has reached a cycle (pattern repeats).
    pub fn detect_cycle(&self) -> Option<u64> {
        let current = &self.cells;
        for (i, past) in self.history.iter().enumerate().rev() {
            if past == current {
                return Some((self.history.len() - i) as u64);
            }
        }
        None
    }

    /// Render the current row as a string ('.' dead, '#' alive).
    pub fn render_row(&self) -> String {
        self.cells.iter().map(|c| if *c != 0 { '#' } else { '.' }).collect()
    }
}

// ── 2D Grid Automaton ───────────────────────────────────────────

/// A 2D cellular automaton (e.g. Conway's Game of Life).
#[derive(Debug, Clone)]
pub struct Grid2D {
    width: usize,
    height: usize,
    cells: Vec<CellState>,
    neighborhood: Neighborhood,
    boundary: Boundary,
    /// Birth rule: number of alive neighbors that cause birth.
    birth: Vec<u8>,
    /// Survival rule: number of alive neighbors that keep a cell alive.
    survival: Vec<u8>,
    generation: u64,
}

impl Grid2D {
    /// Create a new 2D grid with given dimensions and rules.
    pub fn new(
        width: usize,
        height: usize,
        neighborhood: Neighborhood,
        boundary: Boundary,
    ) -> Result<Self, CaError> {
        if width == 0 || height == 0 {
            return Err(CaError::ZeroDimension);
        }
        Ok(Self {
            width,
            height,
            cells: vec![0; width * height],
            neighborhood,
            boundary,
            birth: vec![3],
            survival: vec![2, 3],
            generation: 0,
        })
    }

    /// Create a standard Game of Life (B3/S23, Moore, Wrap).
    pub fn game_of_life(width: usize, height: usize) -> Result<Self, CaError> {
        Self::new(width, height, Neighborhood::Moore, Boundary::Wrap)
    }

    /// Set custom birth/survival rules (B/S notation).
    pub fn with_rules(mut self, birth: Vec<u8>, survival: Vec<u8>) -> Self {
        self.birth = birth;
        self.survival = survival;
        self
    }

    /// Dimensions.
    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }

    /// Current generation.
    pub fn generation(&self) -> u64 { self.generation }

    /// Get cell state at (x, y).
    pub fn get(&self, x: usize, y: usize) -> Result<CellState, CaError> {
        if x >= self.width || y >= self.height {
            return Err(CaError::OutOfBounds { x, y, width: self.width, height: self.height });
        }
        Ok(self.cells[y * self.width + x])
    }

    /// Set cell state at (x, y).
    pub fn set(&mut self, x: usize, y: usize, state: CellState) -> Result<(), CaError> {
        if x >= self.width || y >= self.height {
            return Err(CaError::OutOfBounds { x, y, width: self.width, height: self.height });
        }
        self.cells[y * self.width + x] = state;
        Ok(())
    }

    /// Toggle a cell.
    pub fn toggle(&mut self, x: usize, y: usize) -> Result<(), CaError> {
        let current = self.get(x, y)?;
        self.set(x, y, if current == 0 { 1 } else { 0 })
    }

    /// Raw cell data.
    pub fn cells(&self) -> &[CellState] { &self.cells }

    /// Load cells from a flat slice.
    pub fn load(&mut self, data: &[CellState]) -> Result<(), CaError> {
        if data.len() != self.width * self.height {
            return Err(CaError::SizeMismatch {
                expected: self.width * self.height,
                got: data.len(),
            });
        }
        self.cells.copy_from_slice(data);
        Ok(())
    }

    /// Population count.
    pub fn population(&self) -> usize {
        self.cells.iter().filter(|&&c| c != 0).count()
    }

    /// Density = population / total.
    pub fn density(&self) -> f64 {
        self.population() as f64 / (self.width * self.height) as f64
    }

    /// Count alive neighbors for cell at (x, y).
    pub fn count_neighbors(&self, x: usize, y: usize) -> u8 {
        let offsets: &[(i32, i32)] = match self.neighborhood {
            Neighborhood::Moore => &[
                (-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1),
            ],
            Neighborhood::VonNeumann => &[
                         (0, -1),
                (-1,  0),          (1,  0),
                         (0,  1),
            ],
        };

        let mut count = 0u8;
        for &(dx, dy) in offsets {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            let (cx, cy) = match self.boundary {
                Boundary::Wrap => {
                    let w = self.width as i32;
                    let h = self.height as i32;
                    (((nx % w) + w) % w, ((ny % h) + h) % h)
                }
                Boundary::Dead => {
                    if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                        continue;
                    }
                    (nx, ny)
                }
            };
            if self.cells[cy as usize * self.width + cx as usize] != 0 {
                count += 1;
            }
        }
        count
    }

    /// Advance one generation.
    pub fn step(&mut self) {
        let mut next = vec![0u8; self.width * self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                let n = self.count_neighbors(x, y);
                let alive = self.cells[y * self.width + x] != 0;
                next[y * self.width + x] = if alive {
                    if self.survival.contains(&n) { 1 } else { 0 }
                } else if self.birth.contains(&n) {
                    1
                } else {
                    0
                };
            }
        }
        self.cells = next;
        self.generation += 1;
    }

    /// Step multiple generations.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Clear all cells.
    pub fn clear(&mut self) {
        self.cells.fill(0);
        self.generation = 0;
    }

    /// Place a pattern at (ox, oy). Pattern is a list of (dx, dy) offsets.
    pub fn place_pattern(&mut self, ox: usize, oy: usize, pattern: &[(usize, usize)]) -> Result<(), CaError> {
        for &(dx, dy) in pattern {
            let x = ox + dx;
            let y = oy + dy;
            self.set(x, y, 1)?;
        }
        Ok(())
    }

    /// Render as a multi-line string.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity((self.width + 1) * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                out.push(if self.cells[y * self.width + x] != 0 { '#' } else { '.' });
            }
            if y < self.height - 1 {
                out.push('\n');
            }
        }
        out
    }

    /// Compute a hash of the current state for cycle detection.
    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &c in &self.cells {
            h ^= c as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Detect if the grid has reached a cycle by checking against recorded hashes.
    pub fn detect_cycle_with_history(&self, history: &[u64]) -> Option<usize> {
        let current = self.state_hash();
        for (i, &h) in history.iter().enumerate().rev() {
            if h == current {
                return Some(history.len() - i);
            }
        }
        None
    }

    /// Get population histogram: how many cells have each neighbor count.
    pub fn neighbor_histogram(&self) -> HashMap<u8, usize> {
        let mut hist = HashMap::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let n = self.count_neighbors(x, y);
                *hist.entry(n).or_insert(0) += 1;
            }
        }
        hist
    }
}

/// Well-known patterns for Game of Life.
pub struct Patterns;

impl Patterns {
    /// Glider pattern offsets.
    pub fn glider() -> Vec<(usize, usize)> {
        vec![(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
    }

    /// Blinker (period-2 oscillator).
    pub fn blinker() -> Vec<(usize, usize)> {
        vec![(0, 0), (1, 0), (2, 0)]
    }

    /// Block (still life).
    pub fn block() -> Vec<(usize, usize)> {
        vec![(0, 0), (1, 0), (0, 1), (1, 1)]
    }

    /// Beacon (period-2 oscillator).
    pub fn beacon() -> Vec<(usize, usize)> {
        vec![(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)]
    }

    /// Toad (period-2 oscillator).
    pub fn toad() -> Vec<(usize, usize)> {
        vec![(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)]
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1D Elementary tests ──

    #[test]
    fn test_1d_creation() {
        let ca = ElementaryAutomaton::new(10, 110, Boundary::Dead).unwrap();
        assert_eq!(ca.width(), 10);
        assert_eq!(ca.generation(), 0);
        assert_eq!(ca.population(), 0);
    }

    #[test]
    fn test_1d_zero_width() {
        let result = ElementaryAutomaton::new(0, 30, Boundary::Dead);
        assert!(result.is_err());
    }

    #[test]
    fn test_1d_center_seed() {
        let ca = ElementaryAutomaton::with_center_seed(11, 30, Boundary::Dead).unwrap();
        assert_eq!(ca.get(5), Some(1));
        assert_eq!(ca.population(), 1);
    }

    #[test]
    fn test_1d_set_get() {
        let mut ca = ElementaryAutomaton::new(5, 30, Boundary::Dead).unwrap();
        ca.set(2, 1).unwrap();
        assert_eq!(ca.get(2), Some(1));
        assert_eq!(ca.get(0), Some(0));
    }

    #[test]
    fn test_1d_out_of_bounds() {
        let mut ca = ElementaryAutomaton::new(5, 30, Boundary::Dead).unwrap();
        assert!(ca.set(10, 1).is_err());
    }

    #[test]
    fn test_rule_110_step() {
        // Rule 110 with a single cell should produce specific patterns.
        let mut ca = ElementaryAutomaton::with_center_seed(7, 110, Boundary::Dead).unwrap();
        ca.step();
        assert_eq!(ca.generation(), 1);
        // The center cell and its right neighbor should be alive after rule 110 step.
        assert!(ca.population() >= 1);
    }

    #[test]
    fn test_rule_30_produces_complexity() {
        let mut ca = ElementaryAutomaton::with_center_seed(21, 30, Boundary::Dead).unwrap();
        ca.step_n(5);
        assert_eq!(ca.generation(), 5);
        assert!(ca.population() > 1);
    }

    #[test]
    fn test_1d_wrap_boundary() {
        let mut ca = ElementaryAutomaton::new(5, 30, Boundary::Wrap).unwrap();
        ca.set(0, 1).unwrap();
        ca.set(4, 1).unwrap();
        ca.step();
        // With wrap, cell 0 sees cell 4 as its left neighbor.
        assert_eq!(ca.generation(), 1);
    }

    #[test]
    fn test_1d_history() {
        let mut ca = ElementaryAutomaton::with_center_seed(5, 30, Boundary::Dead).unwrap();
        ca.step();
        ca.step();
        assert_eq!(ca.history().len(), 2);
    }

    #[test]
    fn test_1d_render_row() {
        let mut ca = ElementaryAutomaton::new(5, 30, Boundary::Dead).unwrap();
        ca.set(1, 1).unwrap();
        ca.set(3, 1).unwrap();
        assert_eq!(ca.render_row(), ".#.#.");
    }

    #[test]
    fn test_1d_cycle_detection() {
        // Rule 0: all cells die. After one step from all-dead, stays dead.
        let mut ca = ElementaryAutomaton::new(5, 0, Boundary::Dead).unwrap();
        ca.step();
        ca.step();
        // After two steps of all zeros, cycle should be detected.
        assert!(ca.detect_cycle().is_some());
    }

    // ── 2D Grid tests ──

    #[test]
    fn test_2d_creation() {
        let grid = Grid2D::game_of_life(10, 10).unwrap();
        assert_eq!(grid.width(), 10);
        assert_eq!(grid.height(), 10);
        assert_eq!(grid.generation(), 0);
        assert_eq!(grid.population(), 0);
    }

    #[test]
    fn test_2d_zero_dimension() {
        assert!(Grid2D::new(0, 5, Neighborhood::Moore, Boundary::Wrap).is_err());
        assert!(Grid2D::new(5, 0, Neighborhood::Moore, Boundary::Wrap).is_err());
    }

    #[test]
    fn test_2d_set_get() {
        let mut grid = Grid2D::game_of_life(5, 5).unwrap();
        grid.set(2, 3, 1).unwrap();
        assert_eq!(grid.get(2, 3).unwrap(), 1);
        assert_eq!(grid.get(0, 0).unwrap(), 0);
    }

    #[test]
    fn test_2d_out_of_bounds() {
        let grid = Grid2D::game_of_life(5, 5).unwrap();
        assert!(grid.get(10, 10).is_err());
    }

    #[test]
    fn test_2d_toggle() {
        let mut grid = Grid2D::game_of_life(5, 5).unwrap();
        grid.toggle(1, 1).unwrap();
        assert_eq!(grid.get(1, 1).unwrap(), 1);
        grid.toggle(1, 1).unwrap();
        assert_eq!(grid.get(1, 1).unwrap(), 0);
    }

    #[test]
    fn test_block_is_still_life() {
        let mut grid = Grid2D::game_of_life(6, 6).unwrap();
        grid.place_pattern(1, 1, &Patterns::block()).unwrap();
        let before = grid.cells().to_vec();
        grid.step();
        assert_eq!(grid.cells(), &before[..]);
    }

    #[test]
    fn test_blinker_oscillates() {
        let mut grid = Grid2D::game_of_life(5, 5).unwrap();
        grid.place_pattern(1, 2, &Patterns::blinker()).unwrap();
        let state0 = grid.cells().to_vec();
        grid.step();
        let state1 = grid.cells().to_vec();
        assert_ne!(state0, state1);
        grid.step();
        let state2 = grid.cells().to_vec();
        assert_eq!(state0, state2);
    }

    #[test]
    fn test_moore_neighbor_count() {
        let mut grid = Grid2D::game_of_life(5, 5).unwrap();
        // Place a 3x3 block around (2,2).
        for dy in 0..3 {
            for dx in 0..3 {
                grid.set(1 + dx, 1 + dy, 1).unwrap();
            }
        }
        assert_eq!(grid.count_neighbors(2, 2), 8); // All 8 Moore neighbors alive.
    }

    #[test]
    fn test_von_neumann_neighbor_count() {
        let mut grid = Grid2D::new(5, 5, Neighborhood::VonNeumann, Boundary::Dead).unwrap();
        grid.set(2, 1, 1).unwrap();
        grid.set(1, 2, 1).unwrap();
        grid.set(3, 2, 1).unwrap();
        grid.set(2, 3, 1).unwrap();
        assert_eq!(grid.count_neighbors(2, 2), 4);
    }

    #[test]
    fn test_dead_boundary() {
        let mut grid = Grid2D::new(3, 3, Neighborhood::Moore, Boundary::Dead).unwrap();
        grid.set(0, 0, 1).unwrap();
        // Corner cell with dead boundary has at most 3 neighbors.
        assert_eq!(grid.count_neighbors(0, 0), 0);
    }

    #[test]
    fn test_wrap_boundary() {
        let mut grid = Grid2D::new(3, 3, Neighborhood::Moore, Boundary::Wrap).unwrap();
        grid.set(2, 2, 1).unwrap();
        // (0,0) wraps to see (2,2) as diagonal neighbor.
        assert_eq!(grid.count_neighbors(0, 0), 1);
    }

    #[test]
    fn test_density() {
        let mut grid = Grid2D::game_of_life(10, 10).unwrap();
        for i in 0..25 {
            grid.set(i % 10, i / 10, 1).unwrap();
        }
        let d = grid.density();
        assert!((d - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_load_cells() {
        let mut grid = Grid2D::game_of_life(3, 3).unwrap();
        let data = vec![1, 0, 1, 0, 1, 0, 1, 0, 1];
        grid.load(&data).unwrap();
        assert_eq!(grid.population(), 5);
    }

    #[test]
    fn test_load_wrong_size() {
        let mut grid = Grid2D::game_of_life(3, 3).unwrap();
        assert!(grid.load(&[1, 0]).is_err());
    }

    #[test]
    fn test_clear() {
        let mut grid = Grid2D::game_of_life(5, 5).unwrap();
        grid.set(2, 2, 1).unwrap();
        grid.step();
        grid.clear();
        assert_eq!(grid.population(), 0);
        assert_eq!(grid.generation(), 0);
    }

    #[test]
    fn test_render() {
        let mut grid = Grid2D::game_of_life(3, 3).unwrap();
        grid.set(0, 0, 1).unwrap();
        grid.set(2, 2, 1).unwrap();
        let rendered = grid.render();
        assert_eq!(rendered, "#..\n...\n..#");
    }

    #[test]
    fn test_state_hash_differs() {
        let mut g1 = Grid2D::game_of_life(5, 5).unwrap();
        let h1 = g1.state_hash();
        g1.set(2, 2, 1).unwrap();
        let h2 = g1.state_hash();
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_neighbor_histogram() {
        let grid = Grid2D::game_of_life(3, 3).unwrap();
        let hist = grid.neighbor_histogram();
        // All cells dead, all neighbor counts are 0.
        let total: usize = hist.values().sum();
        assert_eq!(total, 9);
    }

    #[test]
    fn test_custom_rules() {
        // Highlife: B36/S23.
        let mut grid = Grid2D::game_of_life(10, 10).unwrap()
            .with_rules(vec![3, 6], vec![2, 3]);
        grid.set(2, 2, 1).unwrap();
        grid.step();
        assert_eq!(grid.generation(), 1);
    }

    #[test]
    fn test_glider_moves() {
        let mut grid = Grid2D::game_of_life(20, 20).unwrap();
        grid.place_pattern(1, 1, &Patterns::glider()).unwrap();
        let pop0 = grid.population();
        grid.step_n(4);
        // Glider preserves population after 4 steps.
        assert_eq!(grid.population(), pop0);
    }

    #[test]
    fn test_cycle_detection_with_history() {
        let mut grid = Grid2D::game_of_life(6, 6).unwrap();
        grid.place_pattern(1, 1, &Patterns::block()).unwrap();
        let mut hashes = Vec::new();
        hashes.push(grid.state_hash());
        grid.step();
        // Block is a still life, so hash after step should match.
        let cycle = grid.detect_cycle_with_history(&hashes);
        assert!(cycle.is_some());
        assert_eq!(cycle.unwrap(), 1);
    }
}
