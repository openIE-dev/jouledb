//! Conway's Game of Life and variants — birth/survival rules, patterns, RLE, memoization.
//!
//! Replaces LifeViewer / Golly / p5.js GoL libraries. Implements standard B3/S23,
//! configurable birth/survival rule strings, pattern loading (glider, blinker,
//! pulsar, R-pentomino, Gosper glider gun), RLE pattern parser, population and
//! generation tracking, period detection, hashlife-like memoization, and bounding box.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GolError {
    ZeroDimension,
    OutOfBounds { x: usize, y: usize },
    InvalidRuleString(String),
    InvalidRle(String),
    PatternTooLarge { pw: usize, ph: usize, gw: usize, gh: usize },
}

impl fmt::Display for GolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "dimensions must be non-zero"),
            Self::OutOfBounds { x, y } => write!(f, "({x},{y}) out of bounds"),
            Self::InvalidRuleString(s) => write!(f, "invalid rule string: {s}"),
            Self::InvalidRle(s) => write!(f, "invalid RLE: {s}"),
            Self::PatternTooLarge { pw, ph, gw, gh } => {
                write!(f, "pattern {pw}x{ph} too large for grid {gw}x{gh}")
            }
        }
    }
}

impl std::error::Error for GolError {}

// ── Birth/Survival Rule ────────────────────────────────────────

/// A birth/survival rule (e.g. B3/S23 for standard GoL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BsRule {
    /// Neighbor counts that cause birth.
    pub birth: [bool; 9],
    /// Neighbor counts that cause survival.
    pub survival: [bool; 9],
}

impl BsRule {
    /// Standard Conway's Game of Life: B3/S23.
    pub fn life() -> Self {
        let mut birth = [false; 9];
        let mut survival = [false; 9];
        birth[3] = true;
        survival[2] = true;
        survival[3] = true;
        Self { birth, survival }
    }

    /// HighLife: B36/S23.
    pub fn highlife() -> Self {
        let mut birth = [false; 9];
        let mut survival = [false; 9];
        birth[3] = true;
        birth[6] = true;
        survival[2] = true;
        survival[3] = true;
        Self { birth, survival }
    }

    /// Seeds: B2/S.
    pub fn seeds() -> Self {
        let mut birth = [false; 9];
        birth[2] = true;
        Self { birth, survival: [false; 9] }
    }

    /// Day & Night: B3678/S34678.
    pub fn day_night() -> Self {
        let mut birth = [false; 9];
        let mut survival = [false; 9];
        for &b in &[3, 6, 7, 8] { birth[b] = true; }
        for &s in &[3, 4, 6, 7, 8] { survival[s] = true; }
        Self { birth, survival }
    }

    /// Parse from a rule string like "B3/S23", "B36/S23", etc.
    pub fn parse(s: &str) -> Result<Self, GolError> {
        let upper = s.to_uppercase();
        let parts: Vec<&str> = upper.split('/').collect();
        if parts.len() != 2 {
            return Err(GolError::InvalidRuleString(s.to_string()));
        }

        let (b_part, s_part) = if parts[0].starts_with('B') && parts[1].starts_with('S') {
            (&parts[0][1..], &parts[1][1..])
        } else if parts[0].starts_with('S') && parts[1].starts_with('B') {
            (&parts[1][1..], &parts[0][1..])
        } else {
            return Err(GolError::InvalidRuleString(s.to_string()));
        };

        let mut birth = [false; 9];
        let mut survival = [false; 9];

        for ch in b_part.chars() {
            let d = ch.to_digit(10).ok_or_else(|| GolError::InvalidRuleString(s.to_string()))? as usize;
            if d > 8 {
                return Err(GolError::InvalidRuleString(s.to_string()));
            }
            birth[d] = true;
        }
        for ch in s_part.chars() {
            let d = ch.to_digit(10).ok_or_else(|| GolError::InvalidRuleString(s.to_string()))? as usize;
            if d > 8 {
                return Err(GolError::InvalidRuleString(s.to_string()));
            }
            survival[d] = true;
        }

        Ok(Self { birth, survival })
    }

    /// Convert to rule string.
    pub fn to_rule_string(&self) -> String {
        let mut s = String::from("B");
        for (i, &b) in self.birth.iter().enumerate() {
            if b { s.push_str(&i.to_string()); }
        }
        s.push_str("/S");
        for (i, &sv) in self.survival.iter().enumerate() {
            if sv { s.push_str(&i.to_string()); }
        }
        s
    }
}

// ── RLE Parser ─────────────────────────────────────────────────

/// A parsed RLE pattern.
#[derive(Debug, Clone)]
pub struct RlePattern {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<(usize, usize)>,
    pub rule: Option<String>,
}

/// Parse an RLE (Run Length Encoded) pattern string.
/// Format: header line "x = N, y = M[, rule = ...]" followed by encoded data.
/// 'b' = dead, 'o' = alive, '$' = end of row, '!' = end of pattern.
/// Optional run count prefix: "3o" = "ooo".
pub fn parse_rle(input: &str) -> Result<RlePattern, GolError> {
    let mut width = 0;
    let mut height = 0;
    let mut rule = None;
    let mut data_lines = Vec::new();
    let mut header_found = false;

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue; // comment
        }
        if !header_found && trimmed.contains("x") && trimmed.contains("y") {
            // Parse header
            for part in trimmed.split(',') {
                let part = part.trim();
                if let Some(val) = part.strip_prefix("x").or_else(|| part.strip_prefix("X")) {
                    let val = val.trim_start_matches(|c: char| c == ' ' || c == '=').trim();
                    width = val.parse::<usize>().map_err(|_| GolError::InvalidRle("bad width".into()))?;
                } else if let Some(val) = part.strip_prefix("y").or_else(|| part.strip_prefix("Y")) {
                    let val = val.trim_start_matches(|c: char| c == ' ' || c == '=').trim();
                    height = val.parse::<usize>().map_err(|_| GolError::InvalidRle("bad height".into()))?;
                } else if let Some(val) = part.strip_prefix("rule").or_else(|| part.strip_prefix("Rule")) {
                    let val = val.trim_start_matches(|c: char| c == ' ' || c == '=').trim();
                    rule = Some(val.to_string());
                }
            }
            header_found = true;
            continue;
        }
        if header_found {
            data_lines.push(trimmed.to_string());
        }
    }

    if width == 0 || height == 0 {
        return Err(GolError::InvalidRle("missing dimensions".into()));
    }

    let data: String = data_lines.join("");
    let mut cells = Vec::new();
    let mut x = 0usize;
    let mut y = 0usize;
    let mut run_count = 0usize;

    for ch in data.chars() {
        if ch == '!' {
            break;
        }
        if ch.is_ascii_digit() {
            run_count = run_count * 10 + (ch as usize - '0' as usize);
            continue;
        }
        let count = if run_count == 0 { 1 } else { run_count };
        run_count = 0;

        match ch {
            'b' | 'B' => {
                x += count;
            }
            'o' | 'O' => {
                for _ in 0..count {
                    cells.push((x, y));
                    x += 1;
                }
            }
            '$' => {
                y += count;
                x = 0;
            }
            _ => {} // ignore whitespace etc.
        }
    }

    Ok(RlePattern { width, height, cells, rule })
}

// ── Known Patterns ─────────────────────────────────────────────

/// Returns cell positions (x, y) for a glider (period-4 spaceship).
pub fn pattern_glider() -> Vec<(usize, usize)> {
    vec![(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
}

/// Blinker (period-2 oscillator).
pub fn pattern_blinker() -> Vec<(usize, usize)> {
    vec![(0, 1), (1, 1), (2, 1)]
}

/// Block (still life).
pub fn pattern_block() -> Vec<(usize, usize)> {
    vec![(0, 0), (1, 0), (0, 1), (1, 1)]
}

/// Pulsar (period-3 oscillator).
pub fn pattern_pulsar() -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    // Pulsar is symmetric; define one quadrant and mirror
    let quad = [
        (2, 0), (3, 0), (4, 0),
        (0, 2), (0, 3), (0, 4),
        (5, 2), (5, 3), (5, 4),
        (2, 5), (3, 5), (4, 5),
    ];
    let cx = 6;
    let cy = 6;
    for &(qx, qy) in &quad {
        // Four quadrants via mirroring
        cells.push((cx + qx, cy + qy));
        cells.push((cx - qx - 1 + 6, cy + qy));
        cells.push((cx + qx, cy - qy - 1 + 6));
        cells.push((cx - qx - 1 + 6, cy - qy - 1 + 6));
    }
    cells.sort();
    cells.dedup();
    cells
}

/// R-pentomino (methuselah — stabilizes after 1103 generations).
pub fn pattern_r_pentomino() -> Vec<(usize, usize)> {
    vec![(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)]
}

/// Gosper glider gun (period-30 gun).
pub fn pattern_gosper_glider_gun() -> Vec<(usize, usize)> {
    vec![
        (24, 0),
        (22, 1), (24, 1),
        (12, 2), (13, 2), (20, 2), (21, 2), (34, 2), (35, 2),
        (11, 3), (15, 3), (20, 3), (21, 3), (34, 3), (35, 3),
        (0, 4), (1, 4), (10, 4), (16, 4), (20, 4), (21, 4),
        (0, 5), (1, 5), (10, 5), (14, 5), (16, 5), (17, 5), (22, 5), (24, 5),
        (10, 6), (16, 6), (24, 6),
        (11, 7), (15, 7),
        (12, 8), (13, 8),
    ]
}

// ── Game of Life Grid ──────────────────────────────────────────

/// A Game of Life grid with configurable birth/survival rules.
#[derive(Debug, Clone)]
pub struct LifeGrid {
    width: usize,
    height: usize,
    cells: Vec<bool>,
    buffer: Vec<bool>,
    rule: BsRule,
    generation: u64,
    population_history: Vec<usize>,
    /// Memoization cache: maps hashed sub-grid patterns to their evolved state.
    memo: HashMap<u64, Vec<bool>>,
    memo_hits: u64,
}

impl LifeGrid {
    /// Create a new empty grid with standard B3/S23 rules.
    pub fn new(width: usize, height: usize) -> Result<Self, GolError> {
        Self::with_rule(width, height, BsRule::life())
    }

    /// Create with a custom rule.
    pub fn with_rule(width: usize, height: usize, rule: BsRule) -> Result<Self, GolError> {
        if width == 0 || height == 0 {
            return Err(GolError::ZeroDimension);
        }
        let size = width * height;
        Ok(Self {
            width,
            height,
            cells: vec![false; size],
            buffer: vec![false; size],
            rule,
            generation: 0,
            population_history: vec![0],
            memo: HashMap::new(),
            memo_hits: 0,
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn generation(&self) -> u64 { self.generation }

    /// Number of live cells.
    pub fn population(&self) -> usize {
        self.cells.iter().filter(|&&c| c).count()
    }

    pub fn population_history(&self) -> &[usize] {
        &self.population_history
    }

    pub fn memo_hits(&self) -> u64 { self.memo_hits }

    /// Get cell state.
    pub fn get(&self, x: usize, y: usize) -> Result<bool, GolError> {
        if x >= self.width || y >= self.height {
            return Err(GolError::OutOfBounds { x, y });
        }
        Ok(self.cells[y * self.width + x])
    }

    /// Set cell state.
    pub fn set(&mut self, x: usize, y: usize, alive: bool) -> Result<(), GolError> {
        if x >= self.width || y >= self.height {
            return Err(GolError::OutOfBounds { x, y });
        }
        self.cells[y * self.width + x] = alive;
        Ok(())
    }

    /// Set alive from a list of (x, y) positions.
    pub fn set_alive(&mut self, positions: &[(usize, usize)]) -> Result<(), GolError> {
        for &(x, y) in positions {
            self.set(x, y, true)?;
        }
        Ok(())
    }

    /// Load a pattern at offset.
    pub fn load_pattern(&mut self, ox: usize, oy: usize, pattern: &[(usize, usize)]) -> Result<(), GolError> {
        for &(px, py) in pattern {
            let x = ox + px;
            let y = oy + py;
            if x >= self.width || y >= self.height {
                return Err(GolError::OutOfBounds { x, y });
            }
            self.cells[y * self.width + x] = true;
        }
        Ok(())
    }

    /// Count live neighbors (Moore neighborhood, wrapping).
    fn count_neighbors(&self, x: usize, y: usize) -> usize {
        let mut count = 0;
        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 { continue; }
                let nx = ((x as i32 + dx).rem_euclid(self.width as i32)) as usize;
                let ny = ((y as i32 + dy).rem_euclid(self.height as i32)) as usize;
                if self.cells[ny * self.width + nx] {
                    count += 1;
                }
            }
        }
        count
    }

    /// Advance one generation.
    pub fn step(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let alive = self.cells[y * self.width + x];
                let n = self.count_neighbors(x, y);
                self.buffer[y * self.width + x] = if alive {
                    self.rule.survival[n]
                } else {
                    self.rule.birth[n]
                };
            }
        }
        std::mem::swap(&mut self.cells, &mut self.buffer);
        self.generation += 1;
        self.population_history.push(self.population());
    }

    /// Advance multiple generations.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Simple FNV-1a hash of the cell state.
    fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &c in &self.cells {
            h ^= c as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Memoized step: cache the grid state and reuse if seen before.
    pub fn step_memo(&mut self) {
        let hash = self.state_hash();
        if let Some(cached) = self.memo.get(&hash) {
            self.cells = cached.clone();
            self.memo_hits += 1;
        } else {
            let before = self.cells.clone();
            self.step();
            self.memo.insert(hash, self.cells.clone());
            // Undo the generation increment from step() — we manage it here
            // Actually step() already incremented, which is fine
            // We just inserted the mapping old_hash -> new_state
            let _ = before; // consumed
            return;
        }
        self.generation += 1;
        self.population_history.push(self.population());
    }

    /// Bounding box of live cells: (min_x, min_y, max_x, max_y).
    pub fn bounding_box(&self) -> Option<(usize, usize, usize, usize)> {
        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                if self.cells[y * self.width + x] {
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

    /// Detect the period of the current pattern by running up to `max_steps`
    /// and checking if the state repeats.
    pub fn detect_period(&mut self, max_steps: u64) -> Option<u64> {
        let initial = self.cells.clone();
        let initial_gen = self.generation;

        for i in 1..=max_steps {
            self.step();
            if self.cells == initial {
                // Restore state
                self.cells = initial;
                self.generation = initial_gen;
                // Fix population history length
                while self.population_history.len() as u64 > initial_gen + 1 {
                    self.population_history.pop();
                }
                return Some(i);
            }
        }

        // Restore state
        self.cells = initial;
        self.generation = initial_gen;
        while self.population_history.len() as u64 > initial_gen + 1 {
            self.population_history.pop();
        }
        None
    }

    /// Detect if this is a still life (period 1).
    pub fn is_still_life(&mut self) -> bool {
        self.detect_period(1) == Some(1)
    }

    /// Detect if this is an oscillator (period > 1).
    pub fn is_oscillator(&mut self, max_period: u64) -> Option<u64> {
        match self.detect_period(max_period) {
            Some(p) if p > 1 => Some(p),
            _ => None,
        }
    }

    /// Render the grid to a string.
    pub fn to_display_string(&self) -> String {
        let mut s = String::with_capacity((self.width + 1) * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                s.push(if self.cells[y * self.width + x] { '#' } else { '.' });
            }
            if y + 1 < self.height {
                s.push('\n');
            }
        }
        s
    }

    /// List of live cell positions.
    pub fn live_cells(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if self.cells[y * self.width + x] {
                    out.push((x, y));
                }
            }
        }
        out
    }

    /// Clear grid.
    pub fn clear(&mut self) {
        self.cells.fill(false);
        self.generation = 0;
        self.population_history.clear();
        self.population_history.push(0);
    }
}

impl fmt::Display for LifeGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_grid() {
        let g = LifeGrid::new(10, 10).unwrap();
        assert_eq!(g.width(), 10);
        assert_eq!(g.height(), 10);
        assert_eq!(g.population(), 0);
        assert_eq!(g.generation(), 0);
    }

    #[test]
    fn test_zero_dimension() {
        assert!(LifeGrid::new(0, 10).is_err());
    }

    #[test]
    fn test_set_get() {
        let mut g = LifeGrid::new(5, 5).unwrap();
        g.set(2, 3, true).unwrap();
        assert!(g.get(2, 3).unwrap());
        assert!(!g.get(0, 0).unwrap());
    }

    #[test]
    fn test_out_of_bounds() {
        let g = LifeGrid::new(5, 5).unwrap();
        assert!(g.get(5, 0).is_err());
    }

    #[test]
    fn test_blinker_period_2() {
        let mut g = LifeGrid::new(5, 5).unwrap();
        g.load_pattern(1, 2, &pattern_blinker()).unwrap();
        let period = g.detect_period(10).unwrap();
        assert_eq!(period, 2);
    }

    #[test]
    fn test_block_still_life() {
        let mut g = LifeGrid::new(6, 6).unwrap();
        g.load_pattern(2, 2, &pattern_block()).unwrap();
        assert!(g.is_still_life());
    }

    #[test]
    fn test_blinker_is_oscillator() {
        let mut g = LifeGrid::new(5, 5).unwrap();
        g.load_pattern(1, 2, &pattern_blinker()).unwrap();
        assert_eq!(g.is_oscillator(10), Some(2));
    }

    #[test]
    fn test_glider_moves() {
        let mut g = LifeGrid::new(20, 20).unwrap();
        g.load_pattern(1, 1, &pattern_glider()).unwrap();
        let pop0 = g.population();
        g.step_n(4); // One full cycle
        assert_eq!(g.population(), pop0); // Same population
        assert_eq!(g.generation(), 4);
    }

    #[test]
    fn test_r_pentomino_grows() {
        let mut g = LifeGrid::new(50, 50).unwrap();
        g.load_pattern(24, 24, &pattern_r_pentomino()).unwrap();
        assert_eq!(g.population(), 5);
        g.step_n(10);
        assert!(g.population() > 5);
    }

    #[test]
    fn test_gosper_gun_produces_gliders() {
        let mut g = LifeGrid::new(40, 12).unwrap();
        g.load_pattern(1, 1, &pattern_gosper_glider_gun()).unwrap();
        let pop0 = g.population();
        g.step_n(30);
        // After 30 steps the gun should have produced a glider
        assert!(g.population() > pop0);
    }

    #[test]
    fn test_parse_rule_b3s23() {
        let r = BsRule::parse("B3/S23").unwrap();
        assert!(r.birth[3]);
        assert!(!r.birth[2]);
        assert!(r.survival[2]);
        assert!(r.survival[3]);
        assert!(!r.survival[1]);
    }

    #[test]
    fn test_parse_rule_highlife() {
        let r = BsRule::parse("B36/S23").unwrap();
        assert!(r.birth[3]);
        assert!(r.birth[6]);
    }

    #[test]
    fn test_invalid_rule_string() {
        assert!(BsRule::parse("invalid").is_err());
        assert!(BsRule::parse("B/S/X").is_err());
    }

    #[test]
    fn test_rule_to_string() {
        let r = BsRule::life();
        assert_eq!(r.to_rule_string(), "B3/S23");
    }

    #[test]
    fn test_rle_parser_glider() {
        let rle = "x = 3, y = 3\nbo$2bo$3o!";
        let pat = parse_rle(rle).unwrap();
        assert_eq!(pat.width, 3);
        assert_eq!(pat.height, 3);
        assert_eq!(pat.cells.len(), 5);
        // Verify the glider shape
        assert!(pat.cells.contains(&(1, 0)));
        assert!(pat.cells.contains(&(2, 1)));
        assert!(pat.cells.contains(&(0, 2)));
        assert!(pat.cells.contains(&(1, 2)));
        assert!(pat.cells.contains(&(2, 2)));
    }

    #[test]
    fn test_rle_parser_blinker() {
        let rle = "x = 3, y = 1\n3o!";
        let pat = parse_rle(rle).unwrap();
        assert_eq!(pat.cells.len(), 3);
    }

    #[test]
    fn test_rle_with_comments() {
        let rle = "#C A glider\n#O Author\nx = 3, y = 3, rule = B3/S23\nbo$2bo$3o!";
        let pat = parse_rle(rle).unwrap();
        assert_eq!(pat.cells.len(), 5);
        assert_eq!(pat.rule.as_deref(), Some("B3/S23"));
    }

    #[test]
    fn test_rle_invalid() {
        assert!(parse_rle("no header here\n3o!").is_err());
    }

    #[test]
    fn test_bounding_box() {
        let mut g = LifeGrid::new(10, 10).unwrap();
        g.set(3, 2, true).unwrap();
        g.set(7, 8, true).unwrap();
        let bb = g.bounding_box().unwrap();
        assert_eq!(bb, (3, 2, 7, 8));
    }

    #[test]
    fn test_bounding_box_empty() {
        let g = LifeGrid::new(5, 5).unwrap();
        assert!(g.bounding_box().is_none());
    }

    #[test]
    fn test_population_history() {
        let mut g = LifeGrid::new(5, 5).unwrap();
        g.load_pattern(1, 2, &pattern_blinker()).unwrap();
        g.step();
        g.step();
        assert!(g.population_history().len() >= 3);
    }

    #[test]
    fn test_memo_step() {
        let mut g = LifeGrid::new(5, 5).unwrap();
        g.load_pattern(1, 2, &pattern_blinker()).unwrap();
        g.step_memo(); // gen 1
        g.step_memo(); // gen 2 — blinker is back to original, hash is same as gen 0
        g.step_memo(); // gen 3 — should be a memo hit
        assert_eq!(g.generation(), 3);
        assert!(g.memo_hits() > 0);
    }

    #[test]
    fn test_live_cells() {
        let mut g = LifeGrid::new(5, 5).unwrap();
        g.set(1, 1, true).unwrap();
        g.set(3, 3, true).unwrap();
        let live = g.live_cells();
        assert_eq!(live.len(), 2);
        assert!(live.contains(&(1, 1)));
        assert!(live.contains(&(3, 3)));
    }

    #[test]
    fn test_clear() {
        let mut g = LifeGrid::new(5, 5).unwrap();
        g.set(1, 1, true).unwrap();
        g.step();
        g.clear();
        assert_eq!(g.population(), 0);
        assert_eq!(g.generation(), 0);
    }

    #[test]
    fn test_highlife_rule() {
        let mut g = LifeGrid::with_rule(10, 10, BsRule::highlife()).unwrap();
        // HighLife B36/S23: 6 neighbors also cause birth
        g.set(3, 3, true).unwrap();
        g.step();
        // Just verify it runs without panic
        assert_eq!(g.generation(), 1);
    }

    #[test]
    fn test_display() {
        let mut g = LifeGrid::new(3, 3).unwrap();
        g.set(1, 1, true).unwrap();
        let s = format!("{g}");
        assert!(s.contains('#'));
        assert!(s.contains('.'));
    }
}
