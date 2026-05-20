//! 2D cellular automata framework — configurable states, neighborhoods, boundaries.
//!
//! Replaces CA.js / CellPyLib / Wolfram-style 2D automaton libraries. Supports
//! multi-state cells (u8), Moore and Von Neumann neighborhoods, wrap/dead/mirror
//! boundary conditions, double-buffered simultaneous update, totalistic rule
//! encoding, custom rule functions, grid visualization, and generation tracking.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for 2D cellular automata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ca2dError {
    /// Grid dimensions are zero.
    ZeroDimension,
    /// Coordinates out of bounds.
    OutOfBounds { x: usize, y: usize, width: usize, height: usize },
    /// Grid data size does not match dimensions.
    SizeMismatch { expected: usize, got: usize },
    /// Invalid number of states.
    InvalidStates(u8),
}

impl fmt::Display for Ca2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "grid dimensions must be non-zero"),
            Self::OutOfBounds { x, y, width, height } => {
                write!(f, "({x}, {y}) out of bounds for {width}x{height} grid")
            }
            Self::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {expected}, got {got}")
            }
            Self::InvalidStates(n) => write!(f, "invalid state count: {n} (must be >= 2)"),
        }
    }
}

impl std::error::Error for Ca2dError {}

// ── Neighborhood ────────────────────────────────────────────────

/// Neighborhood type for 2D automata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neighborhood2d {
    /// 8 neighbors (orthogonal + diagonal).
    Moore,
    /// 4 neighbors (orthogonal only).
    VonNeumann,
}

impl Neighborhood2d {
    /// Returns the (dx, dy) offsets for this neighborhood.
    pub fn offsets(&self) -> &[(i32, i32)] {
        match self {
            Self::Moore => &[
                (-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1),
            ],
            Self::VonNeumann => &[
                          (0, -1),
                (-1,  0),          (1,  0),
                          (0,  1),
            ],
        }
    }

    /// Number of neighbors.
    pub fn count(&self) -> usize {
        match self {
            Self::Moore => 8,
            Self::VonNeumann => 4,
        }
    }
}

// ── Boundary ────────────────────────────────────────────────────

/// Boundary condition for 2D grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary2d {
    /// Edges wrap around (torus topology).
    Wrap,
    /// Cells outside grid have a fixed dead state.
    Dead(u8),
    /// Edges mirror the interior cells.
    Mirror,
}

// ── Rule types ──────────────────────────────────────────────────

/// A totalistic rule: maps (cell_state, neighbor_sum) -> new_state.
/// `neighbor_sum` is the sum of all neighbor cell values.
#[derive(Debug, Clone)]
pub struct TotalisticRule {
    /// Number of possible cell states (2..=255).
    pub num_states: u8,
    /// birth[sum] = true means a dead cell with this neighbor sum becomes alive.
    pub birth: Vec<bool>,
    /// survival[sum] = true means a living cell with this sum survives.
    pub survival: Vec<bool>,
}

impl TotalisticRule {
    /// Create a new totalistic rule with given birth and survival sums.
    pub fn new(num_states: u8, birth_sums: &[usize], survival_sums: &[usize], max_sum: usize) -> Self {
        let mut birth = vec![false; max_sum + 1];
        let mut survival = vec![false; max_sum + 1];
        for &s in birth_sums {
            if s <= max_sum {
                birth[s] = true;
            }
        }
        for &s in survival_sums {
            if s <= max_sum {
                survival[s] = true;
            }
        }
        Self { num_states, birth, survival }
    }

    /// Apply the totalistic rule.
    pub fn apply(&self, cell: u8, neighbor_sum: usize) -> u8 {
        let sum_idx = neighbor_sum.min(self.birth.len().saturating_sub(1));
        if cell == 0 {
            if sum_idx < self.birth.len() && self.birth[sum_idx] { 1 } else { 0 }
        } else if cell == 1 {
            if sum_idx < self.survival.len() && self.survival[sum_idx] { 1 } else { 0 }
        } else {
            // Multi-state decay: non-zero non-one cells decay toward 0
            cell.saturating_sub(1)
        }
    }
}

/// Rule specification for the automaton.
#[derive(Clone)]
pub enum Rule2d {
    /// Totalistic rule (birth/survival based on neighbor sum).
    Totalistic(TotalisticRule),
    /// Custom rule function: (cell_state, neighbors_slice) -> new_state.
    Custom(fn(u8, &[u8]) -> u8),
}

impl fmt::Debug for Rule2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Totalistic(r) => write!(f, "Totalistic({:?})", r),
            Self::Custom(_) => write!(f, "Custom(fn)"),
        }
    }
}

// ── Grid2d ──────────────────────────────────────────────────────

/// A 2D cellular automaton grid with configurable rules.
#[derive(Debug, Clone)]
pub struct Grid2d {
    width: usize,
    height: usize,
    cells: Vec<u8>,
    buffer: Vec<u8>,
    neighborhood: Neighborhood2d,
    boundary: Boundary2d,
    rule: Rule2d,
    generation: u64,
}

impl Grid2d {
    /// Create a new grid filled with zeros.
    pub fn new(
        width: usize,
        height: usize,
        neighborhood: Neighborhood2d,
        boundary: Boundary2d,
        rule: Rule2d,
    ) -> Result<Self, Ca2dError> {
        if width == 0 || height == 0 {
            return Err(Ca2dError::ZeroDimension);
        }
        let size = width * height;
        Ok(Self {
            width,
            height,
            cells: vec![0; size],
            buffer: vec![0; size],
            neighborhood,
            boundary,
            rule,
            generation: 0,
        })
    }

    /// Create from existing cell data.
    pub fn from_cells(
        width: usize,
        height: usize,
        cells: Vec<u8>,
        neighborhood: Neighborhood2d,
        boundary: Boundary2d,
        rule: Rule2d,
    ) -> Result<Self, Ca2dError> {
        if width == 0 || height == 0 {
            return Err(Ca2dError::ZeroDimension);
        }
        if cells.len() != width * height {
            return Err(Ca2dError::SizeMismatch {
                expected: width * height,
                got: cells.len(),
            });
        }
        let buffer = vec![0; cells.len()];
        Ok(Self {
            width,
            height,
            cells,
            buffer,
            neighborhood,
            boundary,
            rule,
            generation: 0,
        })
    }

    /// Width of the grid.
    pub fn width(&self) -> usize { self.width }

    /// Height of the grid.
    pub fn height(&self) -> usize { self.height }

    /// Current generation count.
    pub fn generation(&self) -> u64 { self.generation }

    /// Get cell at (x, y).
    pub fn get(&self, x: usize, y: usize) -> Result<u8, Ca2dError> {
        if x >= self.width || y >= self.height {
            return Err(Ca2dError::OutOfBounds { x, y, width: self.width, height: self.height });
        }
        Ok(self.cells[y * self.width + x])
    }

    /// Set cell at (x, y).
    pub fn set(&mut self, x: usize, y: usize, state: u8) -> Result<(), Ca2dError> {
        if x >= self.width || y >= self.height {
            return Err(Ca2dError::OutOfBounds { x, y, width: self.width, height: self.height });
        }
        self.cells[y * self.width + x] = state;
        Ok(())
    }

    /// Read the cell at potentially out-of-bounds coordinates, applying boundary.
    fn read_cell(&self, x: i32, y: i32) -> u8 {
        let w = self.width as i32;
        let h = self.height as i32;

        match self.boundary {
            Boundary2d::Wrap => {
                let nx = ((x % w) + w) % w;
                let ny = ((y % h) + h) % h;
                self.cells[ny as usize * self.width + nx as usize]
            }
            Boundary2d::Dead(state) => {
                if x < 0 || x >= w || y < 0 || y >= h {
                    state
                } else {
                    self.cells[y as usize * self.width + x as usize]
                }
            }
            Boundary2d::Mirror => {
                let nx = if x < 0 {
                    (-x - 1).min(w - 1)
                } else if x >= w {
                    (2 * w - x - 1).max(0)
                } else {
                    x
                };
                let ny = if y < 0 {
                    (-y - 1).min(h - 1)
                } else if y >= h {
                    (2 * h - y - 1).max(0)
                } else {
                    y
                };
                let nx = nx.clamp(0, w - 1) as usize;
                let ny = ny.clamp(0, h - 1) as usize;
                self.cells[ny * self.width + nx]
            }
        }
    }

    /// Gather neighbor states for cell at (x, y).
    fn gather_neighbors(&self, x: usize, y: usize) -> Vec<u8> {
        let offsets = self.neighborhood.offsets();
        let cx = x as i32;
        let cy = y as i32;
        offsets.iter().map(|&(dx, dy)| self.read_cell(cx + dx, cy + dy)).collect()
    }

    /// Advance by one generation (double-buffered simultaneous update).
    pub fn step(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = self.cells[y * self.width + x];
                let neighbors = self.gather_neighbors(x, y);
                let new_state = match &self.rule {
                    Rule2d::Totalistic(tr) => {
                        let sum: usize = neighbors.iter().map(|n| *n as usize).sum();
                        tr.apply(cell, sum)
                    }
                    Rule2d::Custom(func) => func(cell, &neighbors),
                };
                self.buffer[y * self.width + x] = new_state;
            }
        }
        std::mem::swap(&mut self.cells, &mut self.buffer);
        self.generation += 1;
    }

    /// Advance by multiple generations.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Population count (cells with non-zero state).
    pub fn population(&self) -> usize {
        self.cells.iter().filter(|&&c| c != 0).count()
    }

    /// Count cells in a specific state.
    pub fn count_state(&self, state: u8) -> usize {
        self.cells.iter().filter(|&&c| c == state).count()
    }

    /// All cells as a flat slice (row-major).
    pub fn cells(&self) -> &[u8] {
        &self.cells
    }

    /// Render grid to a string with character mapping.
    pub fn to_string_with(&self, map: &dyn Fn(u8) -> char) -> String {
        let mut s = String::with_capacity((self.width + 1) * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                s.push(map(self.cells[y * self.width + x]));
            }
            if y + 1 < self.height {
                s.push('\n');
            }
        }
        s
    }

    /// Default visualization: '.' for 0, '#' for non-zero.
    pub fn to_display_string(&self) -> String {
        self.to_string_with(&|c| if c == 0 { '.' } else { '#' })
    }

    /// Bounding box of non-zero cells: (min_x, min_y, max_x, max_y), or None if empty.
    pub fn bounding_box(&self) -> Option<(usize, usize, usize, usize)> {
        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                if self.cells[y * self.width + x] != 0 {
                    found = true;
                    if x < min_x { min_x = x; }
                    if y < min_y { min_y = y; }
                    if x > max_x { max_x = x; }
                    if y > max_y { max_y = y; }
                }
            }
        }

        if found { Some((min_x, min_y, max_x, max_y)) } else { None }
    }

    /// Check if the grid is all dead (all zeros).
    pub fn is_dead(&self) -> bool {
        self.cells.iter().all(|c| *c == 0)
    }

    /// Clear the grid (set all to 0), reset generation.
    pub fn clear(&mut self) {
        self.cells.fill(0);
        self.generation = 0;
    }

    /// Randomize the grid with given density (0.0 to 1.0) for binary states.
    /// Uses a simple LCG seeded by `seed`.
    pub fn randomize(&mut self, density: f64, seed: u64) {
        let mut rng = seed;
        for cell in &mut self.cells {
            // LCG: x' = (a*x + c) mod m
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = ((rng >> 33) as f64) / ((1u64 << 31) as f64);
            *cell = if val < density { 1 } else { 0 };
        }
    }

    /// Load a pattern at the given offset. Pattern is a slice of (dx, dy, state).
    pub fn load_pattern(&mut self, ox: usize, oy: usize, pattern: &[(usize, usize, u8)]) -> Result<(), Ca2dError> {
        for &(dx, dy, state) in pattern {
            let x = ox + dx;
            let y = oy + dy;
            self.set(x, y, state)?;
        }
        Ok(())
    }

    /// Create a Game-of-Life style rule (B3/S23).
    pub fn life_rule() -> Rule2d {
        Rule2d::Totalistic(TotalisticRule::new(2, &[3], &[2, 3], 8))
    }

    /// Create a HighLife rule (B36/S23).
    pub fn highlife_rule() -> Rule2d {
        Rule2d::Totalistic(TotalisticRule::new(2, &[3, 6], &[2, 3], 8))
    }

    /// Create a Seeds rule (B2/S — nothing survives).
    pub fn seeds_rule() -> Rule2d {
        Rule2d::Totalistic(TotalisticRule::new(2, &[2], &[], 8))
    }

    /// Create a Day & Night rule (B3678/S34678).
    pub fn day_night_rule() -> Rule2d {
        Rule2d::Totalistic(TotalisticRule::new(2, &[3, 6, 7, 8], &[3, 4, 6, 7, 8], 8))
    }
}

impl fmt::Display for Grid2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

// ── Wolfram 1D reference rule encoding ─────────────────────────

/// Encode a Wolfram-style elementary 1D rule number (0..255).
/// Returns the lookup table: index = (left, center, right) as 3-bit number.
pub fn wolfram_1d_rule_table(rule: u8) -> [u8; 8] {
    let mut table = [0u8; 8];
    for i in 0..8u8 {
        table[i as usize] = (rule >> i) & 1;
    }
    table
}

/// Apply a Wolfram 1D rule on a row, returning the next row.
pub fn wolfram_1d_step(row: &[u8], rule: u8, wrap: bool) -> Vec<u8> {
    let table = wolfram_1d_rule_table(rule);
    let n = row.len();
    let mut next = vec![0u8; n];
    for i in 0..n {
        let left = if i == 0 {
            if wrap { row[n - 1] } else { 0 }
        } else {
            row[i - 1]
        };
        let center = row[i];
        let right = if i + 1 >= n {
            if wrap { row[0] } else { 0 }
        } else {
            row[i + 1]
        };
        let idx = ((left & 1) << 2) | ((center & 1) << 1) | (right & 1);
        next[i] = table[idx as usize];
    }
    next
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn life_grid(w: usize, h: usize) -> Grid2d {
        Grid2d::new(w, h, Neighborhood2d::Moore, Boundary2d::Wrap, Grid2d::life_rule()).unwrap()
    }

    #[test]
    fn test_new_grid() {
        let g = life_grid(10, 10);
        assert_eq!(g.width(), 10);
        assert_eq!(g.height(), 10);
        assert_eq!(g.generation(), 0);
        assert_eq!(g.population(), 0);
    }

    #[test]
    fn test_zero_dimension_error() {
        let r = Grid2d::new(0, 5, Neighborhood2d::Moore, Boundary2d::Wrap, Grid2d::life_rule());
        assert_eq!(r.unwrap_err(), Ca2dError::ZeroDimension);
    }

    #[test]
    fn test_set_get() {
        let mut g = life_grid(5, 5);
        g.set(2, 3, 1).unwrap();
        assert_eq!(g.get(2, 3).unwrap(), 1);
        assert_eq!(g.get(0, 0).unwrap(), 0);
    }

    #[test]
    fn test_out_of_bounds() {
        let g = life_grid(5, 5);
        assert!(g.get(5, 0).is_err());
        assert!(g.get(0, 5).is_err());
    }

    #[test]
    fn test_blinker_oscillates() {
        // Blinker: horizontal line of 3
        let mut g = life_grid(5, 5);
        g.set(1, 2, 1).unwrap();
        g.set(2, 2, 1).unwrap();
        g.set(3, 2, 1).unwrap();

        let state0 = g.cells().to_vec();
        g.step(); // Becomes vertical
        assert_eq!(g.generation(), 1);
        assert_eq!(g.population(), 3);
        g.step(); // Back to horizontal
        assert_eq!(g.cells(), state0.as_slice());
        assert_eq!(g.generation(), 2);
    }

    #[test]
    fn test_block_is_still() {
        // 2x2 block is a still life
        let mut g = life_grid(6, 6);
        g.set(2, 2, 1).unwrap();
        g.set(3, 2, 1).unwrap();
        g.set(2, 3, 1).unwrap();
        g.set(3, 3, 1).unwrap();

        let state0 = g.cells().to_vec();
        g.step();
        assert_eq!(g.cells(), state0.as_slice());
    }

    #[test]
    fn test_dead_boundary() {
        let mut g = Grid2d::new(
            3, 3, Neighborhood2d::Moore, Boundary2d::Dead(0), Grid2d::life_rule(),
        ).unwrap();
        g.set(0, 0, 1).unwrap();
        // Corner cell with dead boundary: only 3 neighbors in-bounds
        let nbrs = g.gather_neighbors(0, 0);
        assert_eq!(nbrs.len(), 8);
        // 5 neighbors are out of bounds -> dead(0)
        let alive_nbrs: usize = nbrs.iter().map(|n| *n as usize).sum();
        assert_eq!(alive_nbrs, 0);
    }

    #[test]
    fn test_mirror_boundary() {
        let mut g = Grid2d::new(
            3, 3, Neighborhood2d::Moore, Boundary2d::Mirror, Grid2d::life_rule(),
        ).unwrap();
        g.set(0, 0, 1).unwrap();
        // Mirror: (-1,-1) mirrors to (0,0) which is 1
        let val = g.read_cell(-1, -1);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_wrap_boundary() {
        let mut g = life_grid(5, 5);
        g.set(0, 0, 1).unwrap();
        // Wrap: (-1, 0) wraps to (4, 0)
        let val = g.read_cell(-1, 0);
        assert_eq!(val, 0); // (4,0) is 0
        g.set(4, 0, 7).unwrap();
        let val = g.read_cell(-1, 0);
        assert_eq!(val, 7);
    }

    #[test]
    fn test_von_neumann_neighborhood() {
        let g = Grid2d::new(
            5, 5, Neighborhood2d::VonNeumann, Boundary2d::Wrap, Grid2d::life_rule(),
        ).unwrap();
        let nbrs = g.gather_neighbors(2, 2);
        assert_eq!(nbrs.len(), 4);
    }

    #[test]
    fn test_moore_neighborhood() {
        let g = life_grid(5, 5);
        let nbrs = g.gather_neighbors(2, 2);
        assert_eq!(nbrs.len(), 8);
    }

    #[test]
    fn test_from_cells() {
        let cells = vec![0, 1, 1, 0, 1, 0, 0, 0, 1];
        let g = Grid2d::from_cells(3, 3, cells.clone(), Neighborhood2d::Moore, Boundary2d::Wrap, Grid2d::life_rule()).unwrap();
        assert_eq!(g.population(), 4);
    }

    #[test]
    fn test_from_cells_size_mismatch() {
        let cells = vec![0, 1, 1];
        let r = Grid2d::from_cells(3, 3, cells, Neighborhood2d::Moore, Boundary2d::Wrap, Grid2d::life_rule());
        assert!(matches!(r.unwrap_err(), Ca2dError::SizeMismatch { .. }));
    }

    #[test]
    fn test_randomize() {
        let mut g = life_grid(20, 20);
        g.randomize(0.5, 42);
        let pop = g.population();
        // With 400 cells and 50% density, we expect roughly 200
        assert!(pop > 100 && pop < 300, "population was {pop}");
    }

    #[test]
    fn test_clear() {
        let mut g = life_grid(5, 5);
        g.randomize(0.5, 99);
        g.step();
        g.clear();
        assert_eq!(g.population(), 0);
        assert_eq!(g.generation(), 0);
    }

    #[test]
    fn test_step_n() {
        let mut g = life_grid(5, 5);
        g.set(1, 2, 1).unwrap();
        g.set(2, 2, 1).unwrap();
        g.set(3, 2, 1).unwrap();
        g.step_n(4);
        assert_eq!(g.generation(), 4);
    }

    #[test]
    fn test_display_string() {
        let mut g = life_grid(3, 3);
        g.set(1, 1, 1).unwrap();
        let s = g.to_display_string();
        assert!(s.contains('#'));
        assert!(s.contains('.'));
    }

    #[test]
    fn test_bounding_box() {
        let mut g = life_grid(10, 10);
        g.set(3, 2, 1).unwrap();
        g.set(7, 8, 1).unwrap();
        let bb = g.bounding_box().unwrap();
        assert_eq!(bb, (3, 2, 7, 8));
    }

    #[test]
    fn test_bounding_box_empty() {
        let g = life_grid(5, 5);
        assert!(g.bounding_box().is_none());
    }

    #[test]
    fn test_custom_rule() {
        // Custom rule: always set to 1
        let rule = Rule2d::Custom(|_cell, _nbrs| 1);
        let mut g = Grid2d::new(3, 3, Neighborhood2d::Moore, Boundary2d::Wrap, rule).unwrap();
        g.step();
        assert_eq!(g.population(), 9);
    }

    #[test]
    fn test_count_state() {
        let mut g = life_grid(5, 5);
        g.set(0, 0, 1).unwrap();
        g.set(1, 0, 1).unwrap();
        g.set(2, 0, 1).unwrap();
        assert_eq!(g.count_state(1), 3);
        assert_eq!(g.count_state(0), 22);
    }

    #[test]
    fn test_load_pattern() {
        let mut g = life_grid(10, 10);
        let glider = [(0, 0, 1u8), (1, 1, 1), (2, 1, 1), (0, 2, 1), (1, 2, 1)];
        g.load_pattern(2, 2, &glider).unwrap();
        assert_eq!(g.population(), 5);
    }

    #[test]
    fn test_wolfram_1d_rule_30() {
        let row = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
        let next = wolfram_1d_step(&row, 30, false);
        // Rule 30 center seed: generation 1 should have 3 alive cells
        let pop: u8 = next.iter().sum();
        assert!(pop >= 2, "Rule 30 pop = {pop}");
    }

    #[test]
    fn test_wolfram_1d_rule_110() {
        let row = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
        let next = wolfram_1d_step(&row, 110, false);
        let pop: u8 = next.iter().sum();
        assert!(pop >= 1, "Rule 110 pop = {pop}");
    }

    #[test]
    fn test_totalistic_rule_decay() {
        let tr = TotalisticRule::new(4, &[3], &[2, 3], 8);
        // Multi-state decay: state 3 decays to 2
        assert_eq!(tr.apply(3, 0), 2);
        assert_eq!(tr.apply(2, 0), 1);
    }

    #[test]
    fn test_seeds_rule() {
        let mut g = Grid2d::new(
            7, 7, Neighborhood2d::Moore, Boundary2d::Dead(0), Grid2d::seeds_rule(),
        ).unwrap();
        // Seeds: B2/S — two adjacent alive cells birth new cells, nothing survives
        g.set(3, 3, 1).unwrap();
        g.set(4, 3, 1).unwrap();
        g.step();
        // Original cells die (no survival), but their neighbors with sum=2 are born
        assert_eq!(g.get(3, 3).unwrap(), 0); // died
        assert_eq!(g.get(4, 3).unwrap(), 0); // died
    }

    #[test]
    fn test_is_dead() {
        let g = life_grid(5, 5);
        assert!(g.is_dead());
    }

    #[test]
    fn test_display_trait() {
        let g = life_grid(3, 3);
        let s = format!("{g}");
        assert_eq!(s, "...\n...\n...");
    }
}
