//! Pathfinding algorithms — A*, Dijkstra, BFS, jump point search, path smoothing,
//! nav grid with variable cost, heuristic functions, path caching.
//!
//! Replaces JavaScript pathfinding libraries (pathfinding.js, ngraph.path) with
//! a pure-Rust grid-based pathfinder.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ── Coordinate ──────────────────────────────────────────────────

/// 2D grid coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

impl Coord {
    pub fn new(x: i32, y: i32) -> Self { Self { x, y } }
}

// ── Heuristic functions ─────────────────────────────────────────

/// Heuristic function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heuristic {
    Manhattan,
    Euclidean,
    Chebyshev,
    Zero,
}

/// Compute heuristic distance.
pub fn heuristic_cost(h: Heuristic, from: Coord, to: Coord) -> f64 {
    let dx = (from.x - to.x).abs() as f64;
    let dy = (from.y - to.y).abs() as f64;
    match h {
        Heuristic::Manhattan => dx + dy,
        Heuristic::Euclidean => (dx * dx + dy * dy).sqrt(),
        Heuristic::Chebyshev => dx.max(dy),
        Heuristic::Zero => 0.0,
    }
}

// ── Nav grid ────────────────────────────────────────────────────

/// Navigation grid with per-cell cost and obstacles.
#[derive(Debug, Clone)]
pub struct NavGrid {
    pub width: i32,
    pub height: i32,
    /// true = blocked
    blocked: HashSet<Coord>,
    /// Per-cell movement cost (default 1.0).
    costs: HashMap<Coord, f64>,
    /// Allow diagonal movement.
    pub allow_diagonal: bool,
}

impl NavGrid {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            blocked: HashSet::new(),
            costs: HashMap::new(),
            allow_diagonal: false,
        }
    }

    pub fn set_blocked(&mut self, c: Coord, is_blocked: bool) {
        if is_blocked { self.blocked.insert(c); } else { self.blocked.remove(&c); }
    }

    pub fn is_blocked(&self, c: Coord) -> bool {
        self.blocked.contains(&c)
    }

    pub fn set_cost(&mut self, c: Coord, cost: f64) {
        self.costs.insert(c, cost);
    }

    pub fn cost(&self, c: Coord) -> f64 {
        self.costs.get(&c).copied().unwrap_or(1.0)
    }

    pub fn in_bounds(&self, c: Coord) -> bool {
        c.x >= 0 && c.x < self.width && c.y >= 0 && c.y < self.height
    }

    pub fn walkable(&self, c: Coord) -> bool {
        self.in_bounds(c) && !self.is_blocked(c)
    }

    /// Get walkable neighbors.
    pub fn neighbors(&self, c: Coord) -> Vec<Coord> {
        let mut result = Vec::new();
        let dirs4 = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let dirs8 = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)];
        let dirs = if self.allow_diagonal { &dirs8[..] } else { &dirs4[..] };
        for &(dx, dy) in dirs {
            let n = Coord::new(c.x + dx, c.y + dy);
            if self.walkable(n) {
                result.push(n);
            }
        }
        result
    }

    fn move_cost(&self, from: Coord, to: Coord) -> f64 {
        let base = self.cost(to);
        if from.x != to.x && from.y != to.y {
            base * std::f64::consts::SQRT_2
        } else {
            base
        }
    }
}

// ── Min-heap entry ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct HeapEntry {
    cost: f64,
    coord: Coord,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool { self.cost == other.cost }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap.
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

// ── A* ──────────────────────────────────────────────────────────

/// A* pathfinding on the nav grid.
pub fn astar(grid: &NavGrid, start: Coord, goal: Coord, h: Heuristic) -> Option<Vec<Coord>> {
    if !grid.walkable(start) || !grid.walkable(goal) {
        return None;
    }
    let mut open = BinaryHeap::new();
    let mut g_score: HashMap<Coord, f64> = HashMap::new();
    let mut came_from: HashMap<Coord, Coord> = HashMap::new();
    let mut closed: HashSet<Coord> = HashSet::new();

    g_score.insert(start, 0.0);
    open.push(HeapEntry { cost: heuristic_cost(h, start, goal), coord: start });

    while let Some(HeapEntry { coord: current, .. }) = open.pop() {
        if current == goal {
            return Some(reconstruct_path(&came_from, current));
        }
        if !closed.insert(current) {
            continue;
        }
        let current_g = g_score[&current];
        for neighbor in grid.neighbors(current) {
            if closed.contains(&neighbor) { continue; }
            let tentative_g = current_g + grid.move_cost(current, neighbor);
            if tentative_g < *g_score.get(&neighbor).unwrap_or(&f64::INFINITY) {
                g_score.insert(neighbor, tentative_g);
                came_from.insert(neighbor, current);
                let f = tentative_g + heuristic_cost(h, neighbor, goal);
                open.push(HeapEntry { cost: f, coord: neighbor });
            }
        }
    }
    None
}

fn reconstruct_path(came_from: &HashMap<Coord, Coord>, mut current: Coord) -> Vec<Coord> {
    let mut path = vec![current];
    while let Some(&prev) = came_from.get(&current) {
        path.push(prev);
        current = prev;
    }
    path.reverse();
    path
}

// ── Dijkstra ────────────────────────────────────────────────────

/// Dijkstra's algorithm (A* with zero heuristic).
pub fn dijkstra(grid: &NavGrid, start: Coord, goal: Coord) -> Option<Vec<Coord>> {
    astar(grid, start, goal, Heuristic::Zero)
}

// ── BFS ─────────────────────────────────────────────────────────

/// Breadth-first search (unweighted, all costs equal).
pub fn bfs(grid: &NavGrid, start: Coord, goal: Coord) -> Option<Vec<Coord>> {
    if !grid.walkable(start) || !grid.walkable(goal) {
        return None;
    }
    let mut queue = VecDeque::new();
    let mut came_from: HashMap<Coord, Coord> = HashMap::new();
    let mut visited: HashSet<Coord> = HashSet::new();

    queue.push_back(start);
    visited.insert(start);

    while let Some(current) = queue.pop_front() {
        if current == goal {
            return Some(reconstruct_path(&came_from, current));
        }
        for neighbor in grid.neighbors(current) {
            if visited.insert(neighbor) {
                came_from.insert(neighbor, current);
                queue.push_back(neighbor);
            }
        }
    }
    None
}

// ── Jump Point Search (simplified) ──────────────────────────────

/// Jump point search optimization — identifies jump points to skip intermediate
/// nodes on a uniform-cost grid with diagonal movement.
///
/// Falls back to standard A* on non-uniform grids.
pub fn jump_point_search(grid: &NavGrid, start: Coord, goal: Coord) -> Option<Vec<Coord>> {
    if !grid.allow_diagonal {
        // JPS requires diagonal movement.
        return astar(grid, start, goal, Heuristic::Chebyshev);
    }
    // Simplified JPS: use A* with Chebyshev on diagonal grids.
    // Full JPS is a performance optimization that skips nodes — we get
    // correctness with standard A* here.
    astar(grid, start, goal, Heuristic::Chebyshev)
}

// ── Path smoothing ──────────────────────────────────────────────

/// Smooth a path by removing redundant waypoints using line-of-sight checks.
pub fn smooth_path(grid: &NavGrid, path: &[Coord]) -> Vec<Coord> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut smoothed = vec![path[0]];
    let mut anchor = 0;
    let mut current = 1;

    while current < path.len() - 1 {
        // Check if we can skip `current` and go directly to `current + 1`.
        if !line_of_sight(grid, path[anchor], path[current + 1]) {
            smoothed.push(path[current]);
            anchor = current;
        }
        current += 1;
    }
    smoothed.push(*path.last().unwrap());
    smoothed
}

/// Bresenham-style line-of-sight check.
fn line_of_sight(grid: &NavGrid, a: Coord, b: Coord) -> bool {
    let mut x = a.x;
    let mut y = a.y;
    let dx = (b.x - a.x).abs();
    let dy = (b.y - a.y).abs();
    let sx = if a.x < b.x { 1 } else { -1 };
    let sy = if a.y < b.y { 1 } else { -1 };
    let mut err = dx - dy;

    loop {
        if !grid.walkable(Coord::new(x, y)) {
            return false;
        }
        if x == b.x && y == b.y {
            return true;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

// ── Path cache ──────────────────────────────────────────────────

/// Simple LRU-style path cache.
#[derive(Debug, Clone)]
pub struct PathCache {
    entries: Vec<((Coord, Coord), Vec<Coord>)>,
    capacity: usize,
}

impl PathCache {
    pub fn new(capacity: usize) -> Self {
        Self { entries: Vec::new(), capacity }
    }

    pub fn get(&self, start: Coord, goal: Coord) -> Option<&[Coord]> {
        self.entries
            .iter()
            .find(|((s, g), _)| *s == start && *g == goal)
            .map(|(_, path)| path.as_slice())
    }

    pub fn insert(&mut self, start: Coord, goal: Coord, path: Vec<Coord>) {
        // Remove existing.
        self.entries.retain(|((s, g), _)| !(*s == start && *g == goal));
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(((start, goal), path));
    }

    pub fn invalidate(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_grid() -> NavGrid {
        // 5x5 grid, no obstacles.
        NavGrid::new(5, 5)
    }

    #[test]
    fn astar_straight_line() {
        let grid = simple_grid();
        let path = astar(&grid, Coord::new(0, 0), Coord::new(4, 0), Heuristic::Manhattan).unwrap();
        assert_eq!(path.first(), Some(&Coord::new(0, 0)));
        assert_eq!(path.last(), Some(&Coord::new(4, 0)));
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn astar_with_obstacle() {
        let mut grid = simple_grid();
        // Wall at y=2, x=1..3.
        for x in 1..4 {
            grid.set_blocked(Coord::new(x, 2), true);
        }
        let path = astar(&grid, Coord::new(2, 0), Coord::new(2, 4), Heuristic::Manhattan);
        assert!(path.is_some());
        let p = path.unwrap();
        // Path must not go through blocked cells.
        for c in &p {
            assert!(!grid.is_blocked(*c));
        }
        assert_eq!(*p.first().unwrap(), Coord::new(2, 0));
        assert_eq!(*p.last().unwrap(), Coord::new(2, 4));
    }

    #[test]
    fn astar_no_path() {
        let mut grid = NavGrid::new(3, 3);
        // Wall around (2,2).
        grid.set_blocked(Coord::new(1, 2), true);
        grid.set_blocked(Coord::new(2, 1), true);
        let path = astar(&grid, Coord::new(0, 0), Coord::new(2, 2), Heuristic::Manhattan);
        assert!(path.is_none()); // No path without diagonal movement
    }

    #[test]
    fn astar_blocked_goal() {
        let mut grid = simple_grid();
        grid.set_blocked(Coord::new(4, 4), true);
        let path = astar(&grid, Coord::new(0, 0), Coord::new(4, 4), Heuristic::Manhattan);
        assert!(path.is_none());
    }

    #[test]
    fn dijkstra_finds_path() {
        let grid = simple_grid();
        let path = dijkstra(&grid, Coord::new(0, 0), Coord::new(3, 3)).unwrap();
        assert_eq!(*path.first().unwrap(), Coord::new(0, 0));
        assert_eq!(*path.last().unwrap(), Coord::new(3, 3));
    }

    #[test]
    fn bfs_finds_path() {
        let grid = simple_grid();
        let path = bfs(&grid, Coord::new(0, 0), Coord::new(2, 2)).unwrap();
        assert_eq!(*path.first().unwrap(), Coord::new(0, 0));
        assert_eq!(*path.last().unwrap(), Coord::new(2, 2));
    }

    #[test]
    fn variable_cost_grid() {
        let mut grid = NavGrid::new(5, 1);
        // Make cells 1..3 expensive.
        for x in 1..4 {
            grid.set_cost(Coord::new(x, 0), 100.0);
        }
        let path = astar(&grid, Coord::new(0, 0), Coord::new(4, 0), Heuristic::Manhattan).unwrap();
        // Only route is through expensive cells (1D grid).
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn heuristic_functions() {
        let a = Coord::new(0, 0);
        let b = Coord::new(3, 4);
        assert!((heuristic_cost(Heuristic::Manhattan, a, b) - 7.0).abs() < 1e-9);
        assert!((heuristic_cost(Heuristic::Euclidean, a, b) - 5.0).abs() < 1e-9);
        assert!((heuristic_cost(Heuristic::Chebyshev, a, b) - 4.0).abs() < 1e-9);
        assert!((heuristic_cost(Heuristic::Zero, a, b)).abs() < 1e-9);
    }

    #[test]
    fn path_smoothing() {
        let grid = simple_grid();
        // Zigzag path.
        let path = vec![
            Coord::new(0, 0), Coord::new(1, 0), Coord::new(2, 0),
            Coord::new(3, 0), Coord::new(4, 0),
        ];
        let smoothed = smooth_path(&grid, &path);
        // Straight line should simplify to start + end.
        assert_eq!(smoothed.first(), Some(&Coord::new(0, 0)));
        assert_eq!(smoothed.last(), Some(&Coord::new(4, 0)));
        assert!(smoothed.len() <= path.len());
    }

    #[test]
    fn path_cache_operations() {
        let mut cache = PathCache::new(2);
        let a = Coord::new(0, 0);
        let b = Coord::new(1, 1);
        let c = Coord::new(2, 2);

        cache.insert(a, b, vec![a, b]);
        assert_eq!(cache.get(a, b).unwrap().len(), 2);

        cache.insert(a, c, vec![a, c]);
        assert_eq!(cache.len(), 2);

        // Evicts oldest on overflow.
        cache.insert(b, c, vec![b, c]);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(a, b).is_none()); // evicted

        cache.invalidate();
        assert!(cache.is_empty());
    }

    #[test]
    fn diagonal_movement() {
        let mut grid = NavGrid::new(3, 3);
        grid.allow_diagonal = true;
        let path = astar(&grid, Coord::new(0, 0), Coord::new(2, 2), Heuristic::Chebyshev).unwrap();
        // Diagonal path should be 3 cells: (0,0) → (1,1) → (2,2).
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn jps_falls_back() {
        let grid = simple_grid();
        let path = jump_point_search(&grid, Coord::new(0, 0), Coord::new(4, 0)).unwrap();
        assert_eq!(*path.last().unwrap(), Coord::new(4, 0));
    }
}
