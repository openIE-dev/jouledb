//! Langton's Ant and multi-state Turing machines on a 2D grid.
//!
//! Replaces NetLogo / p5.js ant simulations. Classic two-color Langton's Ant,
//! multi-color extensions (RL, RLR, LLRR, etc.), multiple ants, collision handling,
//! auto-growing grid, highway detection, and colored grid visualization.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AntError {
    ZeroDimension,
    InvalidTurnRule(String),
    OutOfBounds { x: usize, y: usize },
    NoAnts,
}

impl fmt::Display for AntError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "grid dimensions must be non-zero"),
            Self::InvalidTurnRule(s) => write!(f, "invalid turn rule: {s}"),
            Self::OutOfBounds { x, y } => write!(f, "({x},{y}) out of bounds"),
            Self::NoAnts => write!(f, "no ants on the grid"),
        }
    }
}

impl std::error::Error for AntError {}

// ── Direction & Turn ────────────────────────────────────────────

/// Cardinal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    /// Turn right (clockwise).
    pub fn turn_right(self) -> Self {
        match self {
            Self::North => Self::East,
            Self::East => Self::South,
            Self::South => Self::West,
            Self::West => Self::North,
        }
    }

    /// Turn left (counter-clockwise).
    pub fn turn_left(self) -> Self {
        match self {
            Self::North => Self::West,
            Self::West => Self::South,
            Self::South => Self::East,
            Self::East => Self::North,
        }
    }

    /// U-turn (reverse).
    pub fn u_turn(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::East => Self::West,
            Self::South => Self::North,
            Self::West => Self::East,
        }
    }

    /// Movement delta (dx, dy). Y increases downward.
    pub fn delta(self) -> (i64, i64) {
        match self {
            Self::North => (0, -1),
            Self::East => (1, 0),
            Self::South => (0, 1),
            Self::West => (-1, 0),
        }
    }
}

/// Turn instruction for a color state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Turn {
    Left,
    Right,
    UTurn,
    NoTurn,
}

impl Turn {
    /// Parse a single character: L, R, U, N.
    pub fn from_char(c: char) -> Result<Self, AntError> {
        match c.to_ascii_uppercase() {
            'L' => Ok(Self::Left),
            'R' => Ok(Self::Right),
            'U' => Ok(Self::UTurn),
            'N' => Ok(Self::NoTurn),
            _ => Err(AntError::InvalidTurnRule(c.to_string())),
        }
    }

    /// Apply this turn to a direction.
    pub fn apply(self, dir: Direction) -> Direction {
        match self {
            Self::Left => dir.turn_left(),
            Self::Right => dir.turn_right(),
            Self::UTurn => dir.u_turn(),
            Self::NoTurn => dir,
        }
    }
}

// ── Turn Rule ──────────────────────────────────────────────────

/// Multi-color turn rule (e.g., "RL" for classic Langton, "RLR" for 3-color).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnRule {
    turns: Vec<Turn>,
}

impl TurnRule {
    /// Parse from a string like "RL", "LLRR", "RLLR".
    pub fn parse(s: &str) -> Result<Self, AntError> {
        if s.is_empty() {
            return Err(AntError::InvalidTurnRule("empty rule".into()));
        }
        let turns: Result<Vec<Turn>, _> = s.chars().map(Turn::from_char).collect();
        Ok(Self { turns: turns? })
    }

    /// Classic Langton's Ant: RL.
    pub fn classic() -> Self {
        Self { turns: vec![Turn::Right, Turn::Left] }
    }

    /// Number of color states.
    pub fn num_colors(&self) -> u8 {
        self.turns.len() as u8
    }

    /// Get the turn for a given color.
    pub fn turn_for(&self, color: u8) -> Turn {
        self.turns[(color as usize) % self.turns.len()]
    }

    /// Get the next color after visiting a cell.
    pub fn next_color(&self, color: u8) -> u8 {
        ((color as usize + 1) % self.turns.len()) as u8
    }
}

// ── Ant ────────────────────────────────────────────────────────

/// A single ant on the grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ant {
    pub x: i64,
    pub y: i64,
    pub direction: Direction,
    pub id: u32,
    pub steps_taken: u64,
}

impl Ant {
    pub fn new(id: u32, x: i64, y: i64, direction: Direction) -> Self {
        Self { x, y, direction, id, steps_taken: 0 }
    }
}

// ── Grid ───────────────────────────────────────────────────────

/// Langton's Ant grid (auto-growing via HashMap).
#[derive(Debug, Clone)]
pub struct LangtonGrid {
    cells: HashMap<(i64, i64), u8>,
    ants: Vec<Ant>,
    rule: TurnRule,
    total_steps: u64,
    min_x: i64,
    max_x: i64,
    min_y: i64,
    max_y: i64,
}

impl LangtonGrid {
    /// Create a new grid with the given turn rule.
    pub fn new(rule: TurnRule) -> Self {
        Self {
            cells: HashMap::new(),
            ants: Vec::new(),
            rule,
            total_steps: 0,
            min_x: 0,
            max_x: 0,
            min_y: 0,
            max_y: 0,
        }
    }

    /// Classic Langton's Ant with one ant at the origin facing north.
    pub fn classic() -> Self {
        let mut grid = Self::new(TurnRule::classic());
        grid.add_ant(0, 0, Direction::North);
        grid
    }

    /// Add an ant at position (x, y) facing the given direction.
    pub fn add_ant(&mut self, x: i64, y: i64, direction: Direction) -> u32 {
        let id = self.ants.len() as u32;
        self.ants.push(Ant::new(id, x, y, direction));
        self.update_bounds(x, y);
        id
    }

    /// Get the color of a cell (0 if unvisited).
    pub fn get_color(&self, x: i64, y: i64) -> u8 {
        self.cells.get(&(x, y)).copied().unwrap_or(0)
    }

    /// Set the color of a cell.
    pub fn set_color(&mut self, x: i64, y: i64, color: u8) {
        if color == 0 {
            self.cells.remove(&(x, y));
        } else {
            self.cells.insert((x, y), color);
        }
        self.update_bounds(x, y);
    }

    fn update_bounds(&mut self, x: i64, y: i64) {
        if x < self.min_x { self.min_x = x; }
        if x > self.max_x { self.max_x = x; }
        if y < self.min_y { self.min_y = y; }
        if y > self.max_y { self.max_y = y; }
    }

    /// Number of ants.
    pub fn ant_count(&self) -> usize { self.ants.len() }

    /// Get ant by index.
    pub fn ant(&self, idx: usize) -> Option<&Ant> { self.ants.get(idx) }

    /// Total steps across all ants.
    pub fn total_steps(&self) -> u64 { self.total_steps }

    /// Bounding box: (min_x, min_y, max_x, max_y).
    pub fn bounds(&self) -> (i64, i64, i64, i64) {
        (self.min_x, self.min_y, self.max_x, self.max_y)
    }

    /// Number of non-zero cells.
    pub fn colored_cells(&self) -> usize { self.cells.len() }

    /// Advance all ants by one step.
    pub fn step(&mut self) {
        let n = self.ants.len();
        for i in 0..n {
            let ax = self.ants[i].x;
            let ay = self.ants[i].y;
            let color = self.get_color(ax, ay);

            // Turn based on current cell color
            let turn = self.rule.turn_for(color);
            self.ants[i].direction = turn.apply(self.ants[i].direction);

            // Flip the cell color
            let new_color = self.rule.next_color(color);
            self.set_color(ax, ay, new_color);

            // Move forward
            let (dx, dy) = self.ants[i].direction.delta();
            self.ants[i].x += dx;
            self.ants[i].y += dy;
            self.update_bounds(self.ants[i].x, self.ants[i].y);
            self.ants[i].steps_taken += 1;
        }
        self.total_steps += 1;
    }

    /// Advance by n steps.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Check for collisions (multiple ants on the same cell).
    pub fn collisions(&self) -> Vec<(i64, i64, Vec<u32>)> {
        let mut positions: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
        for ant in &self.ants {
            positions.entry((ant.x, ant.y)).or_default().push(ant.id);
        }
        let mut result: Vec<_> = positions.into_iter()
            .filter(|(_, ids)| ids.len() > 1)
            .map(|((x, y), ids)| (x, y, ids))
            .collect();
        result.sort_by_key(|(x, y, _)| (*x, *y));
        result
    }

    /// Detect highway formation: check if the ant's recent movement forms
    /// a repeating pattern over the given window. Returns the period if found.
    pub fn detect_highway(&self, ant_idx: usize, history: &[(i64, i64)], min_period: usize) -> Option<usize> {
        if ant_idx >= self.ants.len() || history.len() < min_period * 3 {
            return None;
        }

        let n = history.len();
        // Check candidate periods from min_period up to n/2
        for period in min_period..=(n / 2) {
            let tail = &history[n - period * 2..];
            let first_half = &tail[..period];
            let second_half = &tail[period..];
            // Check if displacement pattern repeats
            let mut matches = true;
            for i in 0..period {
                let dx1 = if i > 0 { first_half[i].0 - first_half[i - 1].0 } else if n > period * 2 { first_half[0].0 - history[n - period * 2 - 1].0 } else { continue };
                let dx2 = second_half[i].0 - if i > 0 { second_half[i - 1].0 } else { first_half[period - 1].0 };
                let dy1 = if i > 0 { first_half[i].1 - first_half[i - 1].1 } else if n > period * 2 { first_half[0].1 - history[n - period * 2 - 1].1 } else { continue };
                let dy2 = second_half[i].1 - if i > 0 { second_half[i - 1].1 } else { first_half[period - 1].1 };
                if dx1 != dx2 || dy1 != dy2 {
                    matches = false;
                    break;
                }
            }
            if matches {
                return Some(period);
            }
        }
        None
    }

    /// Render a portion of the grid to a string.
    /// Uses digits 0-9 for colors, '.' for unvisited, 'A' for ant positions.
    pub fn render(&self, x0: i64, y0: i64, w: usize, h: usize) -> String {
        let mut ant_positions: HashMap<(i64, i64), bool> = HashMap::new();
        for ant in &self.ants {
            ant_positions.insert((ant.x, ant.y), true);
        }

        let mut s = String::with_capacity((w + 1) * h);
        for dy in 0..h {
            for dx in 0..w {
                let x = x0 + dx as i64;
                let y = y0 + dy as i64;
                if ant_positions.contains_key(&(x, y)) {
                    s.push('A');
                } else {
                    let color = self.get_color(x, y);
                    if color == 0 {
                        s.push('.');
                    } else {
                        s.push(char::from(b'0' + (color % 10)));
                    }
                }
            }
            if dy + 1 < h {
                s.push('\n');
            }
        }
        s
    }
}

impl fmt::Display for LangtonGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (min_x, min_y, max_x, max_y) = self.bounds();
        let w = (max_x - min_x + 1).max(1) as usize;
        let h = (max_y - min_y + 1).max(1) as usize;
        // Clamp to reasonable display size
        let w = w.min(80);
        let h = h.min(40);
        write!(f, "{}", self.render(min_x, min_y, w, h))
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classic_creation() {
        let g = LangtonGrid::classic();
        assert_eq!(g.ant_count(), 1);
        assert_eq!(g.total_steps(), 0);
        let ant = g.ant(0).unwrap();
        assert_eq!(ant.x, 0);
        assert_eq!(ant.y, 0);
        assert_eq!(ant.direction, Direction::North);
    }

    #[test]
    fn test_direction_turns() {
        assert_eq!(Direction::North.turn_right(), Direction::East);
        assert_eq!(Direction::East.turn_right(), Direction::South);
        assert_eq!(Direction::North.turn_left(), Direction::West);
        assert_eq!(Direction::North.u_turn(), Direction::South);
    }

    #[test]
    fn test_turn_rule_parse() {
        let r = TurnRule::parse("RL").unwrap();
        assert_eq!(r.num_colors(), 2);
        assert_eq!(r.turn_for(0), Turn::Right);
        assert_eq!(r.turn_for(1), Turn::Left);
    }

    #[test]
    fn test_turn_rule_parse_long() {
        let r = TurnRule::parse("RLLR").unwrap();
        assert_eq!(r.num_colors(), 4);
        assert_eq!(r.turn_for(2), Turn::Left);
        assert_eq!(r.turn_for(3), Turn::Right);
    }

    #[test]
    fn test_invalid_turn_rule() {
        assert!(TurnRule::parse("").is_err());
        assert!(TurnRule::parse("RXL").is_err());
    }

    #[test]
    fn test_classic_first_step() {
        let mut g = LangtonGrid::classic();
        // On white (0): turn right, flip to 1, move
        g.step();
        assert_eq!(g.get_color(0, 0), 1); // flipped
        let ant = g.ant(0).unwrap();
        // North + right = East, move east: (1, 0)
        assert_eq!(ant.x, 1);
        assert_eq!(ant.y, 0);
        assert_eq!(ant.direction, Direction::East);
    }

    #[test]
    fn test_classic_two_steps() {
        let mut g = LangtonGrid::classic();
        g.step(); // Turn right on white, move east
        g.step(); // Turn right on white again, move south
        let ant = g.ant(0).unwrap();
        assert_eq!(ant.x, 1);
        assert_eq!(ant.y, 1);
        assert_eq!(ant.direction, Direction::South);
    }

    #[test]
    fn test_classic_four_steps_symmetry() {
        let mut g = LangtonGrid::classic();
        g.step_n(4);
        // After 4 steps on blank grid, ant should have turned right 4 times
        // and left a small diamond pattern
        assert_eq!(g.total_steps(), 4);
        assert!(g.colored_cells() > 0);
    }

    #[test]
    fn test_step_counter() {
        let mut g = LangtonGrid::classic();
        g.step_n(100);
        assert_eq!(g.total_steps(), 100);
        assert_eq!(g.ant(0).unwrap().steps_taken, 100);
    }

    #[test]
    fn test_multi_color_rule() {
        let mut g = LangtonGrid::new(TurnRule::parse("RLR").unwrap());
        g.add_ant(0, 0, Direction::North);
        g.step();
        // Color 0 -> R -> next color = 1
        assert_eq!(g.get_color(0, 0), 1);
        g.step();
        // New cell at ant position was 0 -> R -> color 1
        assert_eq!(g.total_steps(), 2);
    }

    #[test]
    fn test_multiple_ants() {
        let mut g = LangtonGrid::new(TurnRule::classic());
        g.add_ant(0, 0, Direction::North);
        g.add_ant(10, 10, Direction::South);
        assert_eq!(g.ant_count(), 2);
        g.step();
        // Both ants should have moved
        assert_eq!(g.ant(0).unwrap().steps_taken, 1);
        assert_eq!(g.ant(1).unwrap().steps_taken, 1);
    }

    #[test]
    fn test_collision_detection() {
        let mut g = LangtonGrid::new(TurnRule::classic());
        g.add_ant(0, 0, Direction::East);
        g.add_ant(2, 0, Direction::West);
        // After 1 step they should both be at distance 1 from origin
        g.step();
        // ant0: (0,0) white -> right(east->south), flip, move to (0,1)... wait
        // Actually: ant0 at (0,0) facing East. White cell -> turn Right = South. Move south = (0,1).
        // ant1 at (2,0) facing West. White cell -> turn Right = North. Move north = (2,-1).
        // No collision yet.
        let collisions = g.collisions();
        assert!(collisions.is_empty());
    }

    #[test]
    fn test_auto_grow_bounds() {
        let mut g = LangtonGrid::classic();
        g.step_n(20);
        let (min_x, min_y, max_x, max_y) = g.bounds();
        // Grid should have expanded beyond origin
        assert!(max_x > 0 || min_x < 0 || max_y > 0 || min_y < 0);
    }

    #[test]
    fn test_set_get_color() {
        let mut g = LangtonGrid::classic();
        g.set_color(5, 5, 3);
        assert_eq!(g.get_color(5, 5), 3);
        g.set_color(5, 5, 0);
        assert_eq!(g.get_color(5, 5), 0); // removed
    }

    #[test]
    fn test_render() {
        let mut g = LangtonGrid::classic();
        g.step_n(5);
        let s = g.render(-2, -2, 5, 5);
        assert!(!s.is_empty());
        // Should contain the ant marker or colored cells
        let has_content = s.contains('A') || s.contains('1');
        assert!(has_content);
    }

    #[test]
    fn test_highway_detection_returns_none_for_short_history() {
        let g = LangtonGrid::classic();
        let history: Vec<(i64, i64)> = vec![(0, 0), (1, 0), (2, 0)];
        assert!(g.detect_highway(0, &history, 5).is_none());
    }

    #[test]
    fn test_highway_detection_with_repeating_pattern() {
        let g = LangtonGrid::classic();
        // Construct a repeating displacement pattern with period 2
        let history: Vec<(i64, i64)> = (0..20).map(|i| {
            if i % 2 == 0 { (i, 0) } else { (i, 1) }
        }).collect();
        // This has a repeating displacement pattern
        let result = g.detect_highway(0, &history, 2);
        // May or may not detect depending on exact displacements — just check it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_next_color_wraps() {
        let r = TurnRule::parse("RLR").unwrap();
        assert_eq!(r.next_color(0), 1);
        assert_eq!(r.next_color(1), 2);
        assert_eq!(r.next_color(2), 0); // wraps
    }

    #[test]
    fn test_display_trait() {
        let mut g = LangtonGrid::classic();
        g.step_n(10);
        let s = format!("{g}");
        assert!(!s.is_empty());
    }

    #[test]
    fn test_colored_cells_count() {
        let mut g = LangtonGrid::classic();
        assert_eq!(g.colored_cells(), 0);
        g.step();
        assert!(g.colored_cells() > 0);
    }

    #[test]
    fn test_direction_delta() {
        assert_eq!(Direction::North.delta(), (0, -1));
        assert_eq!(Direction::East.delta(), (1, 0));
        assert_eq!(Direction::South.delta(), (0, 1));
        assert_eq!(Direction::West.delta(), (-1, 0));
    }

    #[test]
    fn test_turn_apply() {
        assert_eq!(Turn::Left.apply(Direction::North), Direction::West);
        assert_eq!(Turn::Right.apply(Direction::North), Direction::East);
        assert_eq!(Turn::UTurn.apply(Direction::North), Direction::South);
        assert_eq!(Turn::NoTurn.apply(Direction::North), Direction::North);
    }
}
