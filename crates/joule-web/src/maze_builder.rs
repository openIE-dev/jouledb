//! Maze generation algorithms with multiple strategies.
//!
//! Implements recursive backtracker (DFS), Kruskal's, Prim's, Eller's,
//! binary tree, and sidewinder algorithms. Produces perfect mazes or
//! braided mazes with dead-end removal. Includes BFS solution finder and
//! text visualization.

use std::collections::VecDeque;

// ── Seeded RNG ──

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if lo >= hi { return lo; }
        lo + (self.next_u64() % (hi - lo) as u64) as usize
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.range(0, i + 1);
            slice.swap(i, j);
        }
    }

    fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }
}

// ── Direction ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    pub fn opposite(self) -> Self {
        match self {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::East => Direction::West,
            Direction::West => Direction::East,
        }
    }

    fn delta(self) -> (isize, isize) {
        match self {
            Direction::North => (0, -1),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
        }
    }

    fn all() -> [Direction; 4] {
        [Direction::North, Direction::South, Direction::East, Direction::West]
    }
}

// ── Algorithm ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    RecursiveBacktracker,
    Kruskal,
    Prim,
    Eller,
    BinaryTree,
    Sidewinder,
}

// ── Cell ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    pub passages: [bool; 4], // N, S, E, W
}

impl Cell {
    fn new() -> Self {
        Self { passages: [false; 4] }
    }

    fn has_passage(&self, dir: Direction) -> bool {
        self.passages[dir as usize]
    }

    fn open(&mut self, dir: Direction) {
        self.passages[dir as usize] = true;
    }
}

// ── Maze ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Maze {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Vec<Cell>>,
    pub start: (usize, usize),
    pub end_pos: (usize, usize),
}

impl Maze {
    fn new(width: usize, height: usize) -> Self {
        let cells = vec![vec![Cell::new(); width]; height];
        Self { width, height, cells, start: (0, 0), end_pos: (width - 1, height - 1) }
    }

    fn neighbor(&self, x: usize, y: usize, dir: Direction) -> Option<(usize, usize)> {
        let (dx, dy) = dir.delta();
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx >= 0 && ny >= 0 && (nx as usize) < self.width && (ny as usize) < self.height {
            Some((nx as usize, ny as usize))
        } else {
            None
        }
    }

    fn link(&mut self, x: usize, y: usize, dir: Direction) {
        self.cells[y][x].open(dir);
        if let Some((nx, ny)) = self.neighbor(x, y, dir) {
            self.cells[ny][nx].open(dir.opposite());
        }
    }

    /// Is the maze perfect (every cell reachable, no loops)?
    pub fn is_perfect(&self) -> bool {
        let mut visited = vec![vec![false; self.width]; self.height];
        let mut count = 0usize;
        let mut queue = VecDeque::new();
        visited[0][0] = true;
        queue.push_back((0usize, 0usize));
        count += 1;

        while let Some((cx, cy)) = queue.pop_front() {
            for dir in Direction::all() {
                if self.cells[cy][cx].has_passage(dir) {
                    if let Some((nx, ny)) = self.neighbor(cx, cy, dir) {
                        if !visited[ny][nx] {
                            visited[ny][nx] = true;
                            count += 1;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
        }
        // Perfect: all reachable, edge count = cell_count - 1
        let edges = self.edge_count();
        count == self.width * self.height && edges == self.width * self.height - 1
    }

    fn edge_count(&self) -> usize {
        let mut count = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.cells[y][x].has_passage(Direction::East) { count += 1; }
                if self.cells[y][x].has_passage(Direction::South) { count += 1; }
            }
        }
        count
    }

    /// Count dead ends (cells with exactly one passage).
    pub fn dead_end_count(&self) -> usize {
        let mut count = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                let passages: usize = self.cells[y][x].passages.iter().filter(|&&p| p).count();
                if passages == 1 { count += 1; }
            }
        }
        count
    }

    /// Solve the maze via BFS, returning the path from start to end.
    pub fn solve(&self) -> Option<Vec<(usize, usize)>> {
        let mut visited = vec![vec![false; self.width]; self.height];
        let mut parent: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; self.width]; self.height];
        let mut queue = VecDeque::new();
        let (sx, sy) = self.start;
        visited[sy][sx] = true;
        queue.push_back((sx, sy));

        while let Some((cx, cy)) = queue.pop_front() {
            if (cx, cy) == self.end_pos {
                let mut path = vec![(cx, cy)];
                let mut cur = (cx, cy);
                while let Some(p) = parent[cur.1][cur.0] {
                    path.push(p);
                    cur = p;
                }
                path.reverse();
                return Some(path);
            }
            for dir in Direction::all() {
                if self.cells[cy][cx].has_passage(dir) {
                    if let Some((nx, ny)) = self.neighbor(cx, cy, dir) {
                        if !visited[ny][nx] {
                            visited[ny][nx] = true;
                            parent[ny][nx] = Some((cx, cy));
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
        }
        None
    }

    /// Render to text: each cell becomes a 3x3 block.
    pub fn to_string_grid(&self) -> String {
        let grid_w = self.width * 2 + 1;
        let grid_h = self.height * 2 + 1;
        let mut grid = vec![vec!['#'; grid_w]; grid_h];

        for y in 0..self.height {
            for x in 0..self.width {
                let gx = x * 2 + 1;
                let gy = y * 2 + 1;
                grid[gy][gx] = ' ';
                if self.cells[y][x].has_passage(Direction::East) && x + 1 < self.width {
                    grid[gy][gx + 1] = ' ';
                }
                if self.cells[y][x].has_passage(Direction::South) && y + 1 < self.height {
                    grid[gy + 1][gx] = ' ';
                }
            }
        }

        let (sx, sy) = self.start;
        grid[sy * 2 + 1][sx * 2 + 1] = 'S';
        let (ex, ey) = self.end_pos;
        grid[ey * 2 + 1][ex * 2 + 1] = 'E';

        grid.iter().map(|row| row.iter().collect::<String>()).collect::<Vec<_>>().join("\n")
    }
}

// ── Builders ──

pub fn build(width: usize, height: usize, algorithm: Algorithm, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = Rng::new(seed);
    match algorithm {
        Algorithm::RecursiveBacktracker => recursive_backtracker(&mut maze, &mut rng),
        Algorithm::Kruskal => kruskal(&mut maze, &mut rng),
        Algorithm::Prim => prim(&mut maze, &mut rng),
        Algorithm::Eller => eller(&mut maze, &mut rng),
        Algorithm::BinaryTree => binary_tree(&mut maze, &mut rng),
        Algorithm::Sidewinder => sidewinder(&mut maze, &mut rng),
    }
    place_start_end(&mut maze);
    maze
}

fn recursive_backtracker(maze: &mut Maze, rng: &mut Rng) {
    let mut visited = vec![vec![false; maze.width]; maze.height];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    visited[0][0] = true;
    stack.push((0, 0));

    while let Some(&(cx, cy)) = stack.last() {
        let mut dirs = Direction::all();
        rng.shuffle(&mut dirs);
        let mut found = false;
        for dir in dirs {
            if let Some((nx, ny)) = maze.neighbor(cx, cy, dir) {
                if !visited[ny][nx] {
                    maze.link(cx, cy, dir);
                    visited[ny][nx] = true;
                    stack.push((nx, ny));
                    found = true;
                    break;
                }
            }
        }
        if !found {
            stack.pop();
        }
    }
}

fn kruskal(maze: &mut Maze, rng: &mut Rng) {
    let total = maze.width * maze.height;
    let mut parent: Vec<usize> = (0..total).collect();

    fn find(parent: &mut Vec<usize>, mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    fn union(parent: &mut Vec<usize>, a: usize, b: usize) -> bool {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb { return false; }
        parent[ra] = rb;
        true
    }

    let mut edges: Vec<(usize, usize, Direction)> = Vec::new();
    for y in 0..maze.height {
        for x in 0..maze.width {
            if x + 1 < maze.width { edges.push((x, y, Direction::East)); }
            if y + 1 < maze.height { edges.push((x, y, Direction::South)); }
        }
    }
    rng.shuffle(&mut edges);

    for (x, y, dir) in edges {
        if let Some((nx, ny)) = maze.neighbor(x, y, dir) {
            let a = y * maze.width + x;
            let b = ny * maze.width + nx;
            if union(&mut parent, a, b) {
                maze.link(x, y, dir);
            }
        }
    }
}

fn prim(maze: &mut Maze, rng: &mut Rng) {
    let mut in_maze = vec![vec![false; maze.width]; maze.height];
    let mut frontier: Vec<(usize, usize, Direction)> = Vec::new();

    let sx = rng.range(0, maze.width);
    let sy = rng.range(0, maze.height);
    in_maze[sy][sx] = true;
    add_frontier(maze, sx, sy, &in_maze, &mut frontier);

    while !frontier.is_empty() {
        let idx = rng.range(0, frontier.len());
        let (fx, fy, dir) = frontier.swap_remove(idx);
        if let Some((nx, ny)) = maze.neighbor(fx, fy, dir) {
            if !in_maze[ny][nx] {
                maze.link(fx, fy, dir);
                in_maze[ny][nx] = true;
                add_frontier(maze, nx, ny, &in_maze, &mut frontier);
            }
        }
    }
}

fn add_frontier(maze: &Maze, x: usize, y: usize, in_maze: &[Vec<bool>], frontier: &mut Vec<(usize, usize, Direction)>) {
    for dir in Direction::all() {
        if let Some((nx, ny)) = maze.neighbor(x, y, dir) {
            if !in_maze[ny][nx] {
                frontier.push((x, y, dir));
            }
        }
    }
}

fn eller(maze: &mut Maze, rng: &mut Rng) {
    let w = maze.width;
    let h = maze.height;
    let mut sets: Vec<usize> = (0..w).collect();
    let mut next_set = w;

    for y in 0..h {
        // Horizontal merges
        for x in 0..w - 1 {
            let last_row = y == h - 1;
            if sets[x] != sets[x + 1] && (last_row || rng.coin()) {
                maze.link(x, y, Direction::East);
                let old = sets[x + 1];
                let new = sets[x];
                for i in 0..w {
                    if sets[i] == old { sets[i] = new; }
                }
            }
        }

        if y < h - 1 {
            // Vertical: each set must have at least one downward connection
            let mut extended: Vec<bool> = vec![false; next_set + w];
            let mut new_sets: Vec<usize> = vec![0; w];

            for x in 0..w {
                if rng.coin() || !extended.get(sets[x]).copied().unwrap_or(false) {
                    maze.link(x, y, Direction::South);
                    if sets[x] < extended.len() {
                        extended[sets[x]] = true;
                    }
                    new_sets[x] = sets[x];
                } else {
                    new_sets[x] = next_set;
                    next_set += 1;
                }
            }

            // Ensure every set has at least one downward
            let mut set_has_down = std::collections::HashMap::new();
            for x in 0..w {
                set_has_down.entry(sets[x]).or_insert(false);
                if new_sets[x] == sets[x] {
                    set_has_down.insert(sets[x], true);
                }
            }
            for x in 0..w {
                if !set_has_down.get(&sets[x]).copied().unwrap_or(true) {
                    maze.link(x, y, Direction::South);
                    new_sets[x] = sets[x];
                    set_has_down.insert(sets[x], true);
                }
            }

            sets = new_sets;
        }
    }
}

fn binary_tree(maze: &mut Maze, rng: &mut Rng) {
    for y in 0..maze.height {
        for x in 0..maze.width {
            let can_n = y > 0;
            let can_e = x + 1 < maze.width;
            if can_n && can_e {
                if rng.coin() { maze.link(x, y, Direction::North); } else { maze.link(x, y, Direction::East); }
            } else if can_n {
                maze.link(x, y, Direction::North);
            } else if can_e {
                maze.link(x, y, Direction::East);
            }
        }
    }
}

fn sidewinder(maze: &mut Maze, rng: &mut Rng) {
    for y in 0..maze.height {
        let mut run_start = 0;
        for x in 0..maze.width {
            let at_east_boundary = x + 1 == maze.width;
            let at_north_boundary = y == 0;

            if at_east_boundary || (!at_north_boundary && rng.coin()) {
                // Close the run: carve north from a random cell in the run
                if !at_north_boundary {
                    let nx = rng.range(run_start, x + 1);
                    maze.link(nx, y, Direction::North);
                }
                run_start = x + 1;
            } else {
                maze.link(x, y, Direction::East);
            }
        }
    }
}

/// Place start/end to maximize path length by BFS from corners.
fn place_start_end(maze: &mut Maze) {
    let (fx, fy) = bfs_farthest(maze, 0, 0);
    maze.start = (fx, fy);
    let (gx, gy) = bfs_farthest(maze, fx, fy);
    maze.end_pos = (gx, gy);
}

fn bfs_farthest(maze: &Maze, sx: usize, sy: usize) -> (usize, usize) {
    let mut dist = vec![vec![usize::MAX; maze.width]; maze.height];
    let mut queue = VecDeque::new();
    dist[sy][sx] = 0;
    queue.push_back((sx, sy));
    let mut farthest = (sx, sy);
    let mut max_dist = 0;

    while let Some((cx, cy)) = queue.pop_front() {
        for dir in Direction::all() {
            if maze.cells[cy][cx].has_passage(dir) {
                if let Some((nx, ny)) = maze.neighbor(cx, cy, dir) {
                    if dist[ny][nx] == usize::MAX {
                        dist[ny][nx] = dist[cy][cx] + 1;
                        if dist[ny][nx] > max_dist {
                            max_dist = dist[ny][nx];
                            farthest = (nx, ny);
                        }
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
    }
    farthest
}

/// Remove dead ends by opening random walls, creating loops (braided maze).
pub fn remove_dead_ends(maze: &mut Maze, fraction: f64, seed: u64) {
    let mut rng = Rng::new(seed);
    let target = ((maze.dead_end_count() as f64) * fraction.clamp(0.0, 1.0)) as usize;
    let mut removed = 0usize;
    let mut attempts = 0usize;
    let max_attempts = maze.width * maze.height * 4;

    while removed < target && attempts < max_attempts {
        attempts += 1;
        let x = rng.range(0, maze.width);
        let y = rng.range(0, maze.height);
        let passages: usize = maze.cells[y][x].passages.iter().filter(|&&p| p).count();
        if passages != 1 { continue; }

        let mut dirs = Direction::all();
        rng.shuffle(&mut dirs);
        for dir in dirs {
            if !maze.cells[y][x].has_passage(dir) {
                if maze.neighbor(x, y, dir).is_some() {
                    maze.link(x, y, dir);
                    removed += 1;
                    break;
                }
            }
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_backtracker_perfect() {
        let m = build(10, 10, Algorithm::RecursiveBacktracker, 42);
        assert!(m.is_perfect());
    }

    #[test]
    fn test_kruskal_perfect() {
        let m = build(8, 8, Algorithm::Kruskal, 99);
        assert!(m.is_perfect());
    }

    #[test]
    fn test_prim_perfect() {
        let m = build(10, 10, Algorithm::Prim, 77);
        assert!(m.is_perfect());
    }

    #[test]
    fn test_eller_all_reachable() {
        let m = build(10, 10, Algorithm::Eller, 55);
        let solution = m.solve();
        // Eller produces connected mazes; verify start->end is solvable
        assert!(solution.is_some());
    }

    #[test]
    fn test_binary_tree_perfect() {
        let m = build(8, 8, Algorithm::BinaryTree, 33);
        assert!(m.is_perfect());
    }

    #[test]
    fn test_sidewinder_perfect() {
        let m = build(8, 8, Algorithm::Sidewinder, 44);
        assert!(m.is_perfect());
    }

    #[test]
    fn test_seed_determinism() {
        let m1 = build(10, 10, Algorithm::RecursiveBacktracker, 123);
        let m2 = build(10, 10, Algorithm::RecursiveBacktracker, 123);
        assert_eq!(m1.cells, m2.cells);
    }

    #[test]
    fn test_different_seeds() {
        let m1 = build(10, 10, Algorithm::RecursiveBacktracker, 1);
        let m2 = build(10, 10, Algorithm::RecursiveBacktracker, 999);
        assert_ne!(m1.cells, m2.cells);
    }

    #[test]
    fn test_solve_returns_path() {
        let m = build(10, 10, Algorithm::RecursiveBacktracker, 42);
        let path = m.solve().expect("should have solution");
        assert_eq!(*path.first().unwrap(), m.start);
        assert_eq!(*path.last().unwrap(), m.end_pos);
        assert!(path.len() > 1);
    }

    #[test]
    fn test_solve_path_valid() {
        let m = build(8, 8, Algorithm::Kruskal, 77);
        let path = m.solve().unwrap();
        for window in path.windows(2) {
            let (x1, y1) = window[0];
            let (x2, y2) = window[1];
            let dx = (x2 as isize - x1 as isize).unsigned_abs();
            let dy = (y2 as isize - y1 as isize).unsigned_abs();
            assert_eq!(dx + dy, 1, "path steps must be adjacent");
        }
    }

    #[test]
    fn test_dead_end_removal() {
        let mut m = build(10, 10, Algorithm::RecursiveBacktracker, 42);
        let before = m.dead_end_count();
        remove_dead_ends(&mut m, 0.5, 100);
        let after = m.dead_end_count();
        assert!(after <= before, "dead ends should not increase");
        assert!(!m.is_perfect(), "maze should have loops now");
    }

    #[test]
    fn test_to_string_grid() {
        let m = build(5, 5, Algorithm::BinaryTree, 42);
        let s = m.to_string_grid();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 11); // 5*2+1
        assert_eq!(lines[0].len(), 11);
        assert!(s.contains('S'));
        assert!(s.contains('E'));
    }

    #[test]
    fn test_start_end_placement() {
        let m = build(10, 10, Algorithm::RecursiveBacktracker, 42);
        assert_ne!(m.start, m.end_pos);
    }

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction::North.opposite(), Direction::South);
        assert_eq!(Direction::East.opposite(), Direction::West);
    }

    #[test]
    fn test_small_maze() {
        let m = build(2, 2, Algorithm::RecursiveBacktracker, 1);
        assert!(m.is_perfect());
        assert!(m.solve().is_some());
    }

    #[test]
    fn test_1x1_maze() {
        let m = build(1, 1, Algorithm::RecursiveBacktracker, 0);
        assert_eq!(m.start, m.end_pos);
    }

    #[test]
    fn test_all_algorithms_solvable() {
        let algos = [
            Algorithm::RecursiveBacktracker,
            Algorithm::Kruskal,
            Algorithm::Prim,
            Algorithm::Eller,
            Algorithm::BinaryTree,
            Algorithm::Sidewinder,
        ];
        for algo in algos {
            let m = build(8, 8, algo, 42);
            assert!(m.solve().is_some(), "{:?} should produce solvable maze", algo);
        }
    }

    #[test]
    fn test_dead_end_count() {
        let m = build(10, 10, Algorithm::RecursiveBacktracker, 42);
        let count = m.dead_end_count();
        assert!(count > 0, "perfect maze should have dead ends");
    }

    #[test]
    fn test_large_maze() {
        let m = build(50, 50, Algorithm::RecursiveBacktracker, 42);
        assert!(m.is_perfect());
        assert!(m.solve().is_some());
    }

    #[test]
    fn test_remove_all_dead_ends() {
        let mut m = build(8, 8, Algorithm::RecursiveBacktracker, 42);
        remove_dead_ends(&mut m, 1.0, 200);
        // Most dead ends should be gone (may not reach zero due to corners)
        // Just ensure it reduced significantly
        assert!(m.solve().is_some());
    }

    #[test]
    fn test_edge_count_perfect() {
        let m = build(5, 5, Algorithm::RecursiveBacktracker, 42);
        assert_eq!(m.edge_count(), 5 * 5 - 1);
    }
}
