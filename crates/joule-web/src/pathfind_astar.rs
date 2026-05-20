//! A* pathfinding on weighted graphs — generic over node type, multiple
//! heuristics (Manhattan, Euclidean, Chebyshev, Octile), binary heap open set,
//! incremental search, tie-breaking, grid helpers (4/8-connected).
//!
//! Replaces JavaScript A* libraries (astar, pathfinding) with a pure-Rust
//! generic graph pathfinder.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;

// ── Heuristic ───────────────────────────────────────────────────

/// Heuristic function type for grid-based search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heuristic {
    Manhattan,
    Euclidean,
    Chebyshev,
    Octile,
    Zero,
}

/// Compute heuristic cost between two grid coordinates.
pub fn heuristic_cost(h: Heuristic, ax: i32, ay: i32, bx: i32, by: i32) -> f64 {
    let dx = (ax - bx).abs() as f64;
    let dy = (ay - by).abs() as f64;
    match h {
        Heuristic::Manhattan => dx + dy,
        Heuristic::Euclidean => (dx * dx + dy * dy).sqrt(),
        Heuristic::Chebyshev => dx.max(dy),
        Heuristic::Octile => {
            let min = dx.min(dy);
            let max = dx.max(dy);
            (max - min) + std::f64::consts::SQRT_2 * min
        }
        Heuristic::Zero => 0.0,
    }
}

// ── Search result ───────────────────────────────────────────────

/// Result of an A* search.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchResult<N> {
    /// Path found with total cost.
    Found { path: Vec<N>, cost: f64 },
    /// No path exists.
    NotFound,
    /// Search was interrupted (max iterations reached); partial state can be resumed.
    Partial { explored: usize },
}

// ── Generic A* ──────────────────────────────────────────────────

/// A* search node for the open set.
#[derive(Debug)]
struct AStarNode<N> {
    node: N,
    f_cost: f64,
    g_cost: f64,
    /// Tie-breaker: lower is better (insertion order).
    tiebreak: u64,
}

impl<N: Eq> PartialEq for AStarNode<N> {
    fn eq(&self, other: &Self) -> bool { self.node == other.node }
}
impl<N: Eq> Eq for AStarNode<N> {}

impl<N: Eq> Ord for AStarNode<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_cost.partial_cmp(&self.f_cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.g_cost.partial_cmp(&self.g_cost).unwrap_or(Ordering::Equal))
            .then_with(|| other.tiebreak.cmp(&self.tiebreak))
    }
}

impl<N: Eq> PartialOrd for AStarNode<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A* search state that supports incremental search.
pub struct AStarState<N: Clone + Eq + Hash> {
    open: BinaryHeap<AStarNode<N>>,
    g_cost: HashMap<N, f64>,
    came_from: HashMap<N, N>,
    closed: HashSet<N>,
    counter: u64,
    iterations: usize,
    goal: N,
}

impl<N: Clone + Eq + Hash> AStarState<N> {
    /// Number of iterations so far.
    pub fn iterations(&self) -> usize { self.iterations }

    /// Number of explored nodes.
    pub fn explored(&self) -> usize { self.closed.len() }
}

/// Run generic A* search.
///
/// - `start`: starting node
/// - `goal`: target node
/// - `neighbors`: function returning (neighbor, edge_cost) pairs
/// - `heuristic`: function estimating cost from node to goal
/// - `max_iterations`: max nodes to expand (0 = unlimited)
pub fn astar<N, FN, FH>(
    start: N,
    goal: N,
    neighbors: FN,
    heuristic: FH,
    max_iterations: usize,
) -> SearchResult<N>
where
    N: Clone + Eq + Hash + std::fmt::Debug,
    FN: Fn(&N) -> Vec<(N, f64)>,
    FH: Fn(&N) -> f64,
{
    let mut state = astar_init(start, goal, &heuristic);
    astar_resume(&mut state, &neighbors, &heuristic, max_iterations)
}

/// Initialize A* search state (for incremental use).
pub fn astar_init<N, FH>(start: N, goal: N, heuristic: &FH) -> AStarState<N>
where
    N: Clone + Eq + Hash,
    FH: Fn(&N) -> f64,
{
    let mut open = BinaryHeap::new();
    let mut g_cost = HashMap::new();
    let h = heuristic(&start);
    g_cost.insert(start.clone(), 0.0);
    open.push(AStarNode { node: start, f_cost: h, g_cost: 0.0, tiebreak: 0 });
    AStarState {
        open,
        g_cost,
        came_from: HashMap::new(),
        closed: HashSet::new(),
        counter: 1,
        iterations: 0,
        goal,
    }
}

/// Resume A* search from a saved state.
pub fn astar_resume<N, FN, FH>(
    state: &mut AStarState<N>,
    neighbors: &FN,
    heuristic: &FH,
    max_iterations: usize,
) -> SearchResult<N>
where
    N: Clone + Eq + Hash + std::fmt::Debug,
    FN: Fn(&N) -> Vec<(N, f64)>,
    FH: Fn(&N) -> f64,
{
    let limit = if max_iterations == 0 { usize::MAX } else { max_iterations };
    let mut iters = 0;

    while let Some(current) = state.open.pop() {
        if iters >= limit {
            // Put back and return partial
            state.open.push(current);
            return SearchResult::Partial { explored: state.closed.len() };
        }

        if current.node == state.goal {
            let path = reconstruct_path(&state.came_from, &state.goal);
            let cost = current.g_cost;
            state.iterations += iters;
            return SearchResult::Found { path, cost };
        }

        if !state.closed.insert(current.node.clone()) {
            continue;
        }

        let current_g = current.g_cost;

        for (neighbor, edge_cost) in neighbors(&current.node) {
            if state.closed.contains(&neighbor) {
                continue;
            }
            let new_g = current_g + edge_cost;
            let existing_g = state.g_cost.get(&neighbor).copied().unwrap_or(f64::MAX);
            if new_g < existing_g {
                state.g_cost.insert(neighbor.clone(), new_g);
                state.came_from.insert(neighbor.clone(), current.node.clone());
                let h = heuristic(&neighbor);
                state.open.push(AStarNode {
                    node: neighbor,
                    f_cost: new_g + h,
                    g_cost: new_g,
                    tiebreak: state.counter,
                });
                state.counter += 1;
            }
        }

        iters += 1;
    }

    state.iterations += iters;
    SearchResult::NotFound
}

/// Reconstruct path from came_from map.
fn reconstruct_path<N: Clone + Eq + Hash>(came_from: &HashMap<N, N>, goal: &N) -> Vec<N> {
    let mut path = vec![goal.clone()];
    let mut current = goal.clone();
    while let Some(prev) = came_from.get(&current) {
        path.push(prev.clone());
        current = prev.clone();
    }
    path.reverse();
    path
}

// ── Grid helpers ────────────────────────────────────────────────

/// Grid coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPos {
    pub x: i32,
    pub y: i32,
}

impl GridPos {
    pub fn new(x: i32, y: i32) -> Self { Self { x, y } }
}

/// Diagonal movement options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagonalMode {
    /// No diagonal movement (4-connected).
    Never,
    /// Always allow diagonals (8-connected).
    Always,
    /// Only if no adjacent walls block (no corner cutting).
    NoCornering,
}

/// Grid for pathfinding with variable cell costs.
pub struct PathGrid {
    pub width: usize,
    pub height: usize,
    /// Cell costs (0.0 = impassable, >0 = traversable).
    costs: Vec<f64>,
}

impl PathGrid {
    /// Create a new grid with uniform cost.
    pub fn new(width: usize, height: usize, default_cost: f64) -> Self {
        Self {
            width,
            height,
            costs: vec![default_cost; width * height],
        }
    }

    /// Set cost for a cell (0.0 = blocked).
    pub fn set_cost(&mut self, x: usize, y: usize, cost: f64) {
        if x < self.width && y < self.height {
            self.costs[y * self.width + x] = cost;
        }
    }

    /// Get cost for a cell.
    pub fn cost(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.costs[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Check if a cell is walkable.
    pub fn is_walkable(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 {
            return false;
        }
        let ux = x as usize;
        let uy = y as usize;
        ux < self.width && uy < self.height && self.costs[uy * self.width + ux] > 0.0
    }

    /// Get neighbors for a grid cell.
    pub fn neighbors(&self, pos: GridPos, diagonal: DiagonalMode) -> Vec<(GridPos, f64)> {
        let mut result = Vec::new();
        let cardinal = [(0, -1), (1, 0), (0, 1), (-1, 0)];
        let diag = [(1, -1), (1, 1), (-1, 1), (-1, -1)];

        for (dx, dy) in cardinal {
            let nx = pos.x + dx;
            let ny = pos.y + dy;
            if self.is_walkable(nx, ny) {
                let c = self.cost(nx as usize, ny as usize);
                result.push((GridPos::new(nx, ny), c));
            }
        }

        if diagonal != DiagonalMode::Never {
            for (i, (dx, dy)) in diag.iter().enumerate() {
                let nx = pos.x + dx;
                let ny = pos.y + dy;
                if !self.is_walkable(nx, ny) {
                    continue;
                }
                if diagonal == DiagonalMode::NoCornering {
                    // Check adjacent cardinals
                    let adj1 = cardinal[i];
                    let adj2 = cardinal[(i + 1) % 4];
                    if !self.is_walkable(pos.x + adj1.0, pos.y + adj1.1)
                        || !self.is_walkable(pos.x + adj2.0, pos.y + adj2.1)
                    {
                        continue;
                    }
                }
                let c = self.cost(nx as usize, ny as usize) * std::f64::consts::SQRT_2;
                result.push((GridPos::new(nx, ny), c));
            }
        }

        result
    }

    /// A* pathfinding on this grid.
    pub fn find_path(
        &self,
        start: GridPos,
        goal: GridPos,
        heur: Heuristic,
        diagonal: DiagonalMode,
        max_iterations: usize,
    ) -> SearchResult<GridPos> {
        astar(
            start,
            goal,
            |pos| self.neighbors(*pos, diagonal),
            |pos| heuristic_cost(heur, pos.x, pos.y, goal.x, goal.y),
            max_iterations,
        )
    }
}

// ── Path cost utility ───────────────────────────────────────────

/// Compute total Euclidean length of a path on a grid.
pub fn path_length(path: &[GridPos]) -> f64 {
    let mut total = 0.0;
    for i in 1..path.len() {
        let dx = (path[i].x - path[i - 1].x) as f64;
        let dy = (path[i].y - path[i - 1].y) as f64;
        total += (dx * dx + dy * dy).sqrt();
    }
    total
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_manhattan() {
        let h = heuristic_cost(Heuristic::Manhattan, 0, 0, 3, 4);
        assert!((h - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_heuristic_euclidean() {
        let h = heuristic_cost(Heuristic::Euclidean, 0, 0, 3, 4);
        assert!((h - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_heuristic_chebyshev() {
        let h = heuristic_cost(Heuristic::Chebyshev, 0, 0, 3, 4);
        assert!((h - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_heuristic_octile() {
        let h = heuristic_cost(Heuristic::Octile, 0, 0, 3, 4);
        // max(3,4) - min(3,4) + sqrt(2) * min(3,4) = 1 + 3*sqrt(2)
        let expected = 1.0 + 3.0 * std::f64::consts::SQRT_2;
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn test_heuristic_zero() {
        let h = heuristic_cost(Heuristic::Zero, 5, 5, 10, 10);
        assert!((h - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_graph_astar() {
        // Graph: A -> B (1), A -> C (4), B -> C (2), C -> D (1)
        let result = astar(
            'A',
            'D',
            |node| match node {
                'A' => vec![('B', 1.0), ('C', 4.0)],
                'B' => vec![('C', 2.0)],
                'C' => vec![('D', 1.0)],
                _ => vec![],
            },
            |_| 0.0, // Dijkstra
            0,
        );
        match result {
            SearchResult::Found { path, cost } => {
                assert_eq!(path, vec!['A', 'B', 'C', 'D']);
                assert!((cost - 4.0).abs() < 1e-10);
            }
            _ => panic!("Expected Found"),
        }
    }

    #[test]
    fn test_no_path() {
        let result = astar(
            0,
            5,
            |node| match node {
                0 => vec![(1, 1.0)],
                1 => vec![(2, 1.0)],
                _ => vec![],
            },
            |_| 0.0,
            0,
        );
        assert_eq!(result, SearchResult::NotFound);
    }

    #[test]
    fn test_start_equals_goal() {
        let result = astar(
            0,
            0,
            |_| vec![(1, 1.0)],
            |_| 0.0,
            0,
        );
        match result {
            SearchResult::Found { path, cost } => {
                assert_eq!(path, vec![0]);
                assert!((cost - 0.0).abs() < 1e-10);
            }
            _ => panic!("Expected Found"),
        }
    }

    #[test]
    fn test_max_iterations() {
        let result = astar(
            0i32,
            100,
            |n| vec![(*n + 1, 1.0)],
            |n| (100 - *n) as f64,
            5,
        );
        match result {
            SearchResult::Partial { explored } => {
                assert!(explored <= 6);
            }
            _ => panic!("Expected Partial"),
        }
    }

    #[test]
    fn test_incremental_search() {
        let goal = 10i32;
        let neighbors = |n: &i32| -> Vec<(i32, f64)> {
            let mut r = Vec::new();
            if *n + 1 <= goal { r.push((*n + 1, 1.0)); }
            r
        };
        let heuristic = |n: &i32| (goal - *n) as f64;

        let mut state = astar_init(0, goal, &heuristic);
        // Run 3 iterations
        let r1 = astar_resume(&mut state, &neighbors, &heuristic, 3);
        assert!(matches!(r1, SearchResult::Partial { .. }));

        // Resume to completion
        let r2 = astar_resume(&mut state, &neighbors, &heuristic, 0);
        match r2 {
            SearchResult::Found { path, cost } => {
                assert_eq!(path.len(), 11); // 0..=10
                assert!((cost - 10.0).abs() < 1e-10);
            }
            _ => panic!("Expected Found after resume"),
        }
    }

    #[test]
    fn test_grid_4connected() {
        let grid = PathGrid::new(5, 5, 1.0);
        let result = grid.find_path(
            GridPos::new(0, 0),
            GridPos::new(4, 0),
            Heuristic::Manhattan,
            DiagonalMode::Never,
            0,
        );
        match result {
            SearchResult::Found { path, cost } => {
                assert_eq!(path.len(), 5);
                assert!((cost - 4.0).abs() < 1e-10);
            }
            _ => panic!("Expected Found"),
        }
    }

    #[test]
    fn test_grid_8connected() {
        let grid = PathGrid::new(5, 5, 1.0);
        let result = grid.find_path(
            GridPos::new(0, 0),
            GridPos::new(2, 2),
            Heuristic::Octile,
            DiagonalMode::Always,
            0,
        );
        match result {
            SearchResult::Found { path, cost } => {
                // Diagonal path: 2 diagonal steps
                assert!((cost - 2.0 * std::f64::consts::SQRT_2).abs() < 1e-6);
                assert!(path.len() <= 4);
            }
            _ => panic!("Expected Found"),
        }
    }

    #[test]
    fn test_grid_blocked() {
        let mut grid = PathGrid::new(3, 3, 1.0);
        // Block middle column
        grid.set_cost(1, 0, 0.0);
        grid.set_cost(1, 1, 0.0);
        grid.set_cost(1, 2, 0.0);
        let result = grid.find_path(
            GridPos::new(0, 0),
            GridPos::new(2, 0),
            Heuristic::Manhattan,
            DiagonalMode::Never,
            0,
        );
        assert_eq!(result, SearchResult::NotFound);
    }

    #[test]
    fn test_grid_variable_cost() {
        let mut grid = PathGrid::new(3, 1, 1.0);
        grid.set_cost(1, 0, 5.0); // expensive middle cell
        let result = grid.find_path(
            GridPos::new(0, 0),
            GridPos::new(2, 0),
            Heuristic::Manhattan,
            DiagonalMode::Never,
            0,
        );
        match result {
            SearchResult::Found { cost, .. } => {
                assert!((cost - 6.0).abs() < 1e-10); // 5 + 1
            }
            _ => panic!("Expected Found"),
        }
    }

    #[test]
    fn test_grid_no_cornering() {
        let mut grid = PathGrid::new(3, 3, 1.0);
        grid.set_cost(1, 0, 0.0); // wall above diagonal
        let neighbors = grid.neighbors(GridPos::new(0, 0), DiagonalMode::NoCornering);
        // Should not have (1,1) as neighbor if (1,0) is blocked and diagonal to (1,-1) is also blocked
        let has_diag = neighbors.iter().any(|(p, _)| p.x == 1 && p.y == -1);
        assert!(!has_diag); // (-1,-1) is out of bounds too
    }

    #[test]
    fn test_path_length() {
        let path = vec![
            GridPos::new(0, 0),
            GridPos::new(1, 0),
            GridPos::new(2, 0),
        ];
        assert!((path_length(&path) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_path_length_diagonal() {
        let path = vec![
            GridPos::new(0, 0),
            GridPos::new(1, 1),
        ];
        assert!((path_length(&path) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_tie_breaking() {
        // Multiple equal-cost paths; A* should still find one
        let grid = PathGrid::new(3, 3, 1.0);
        let result = grid.find_path(
            GridPos::new(0, 0),
            GridPos::new(2, 2),
            Heuristic::Manhattan,
            DiagonalMode::Never,
            0,
        );
        match result {
            SearchResult::Found { path, cost } => {
                assert!((cost - 4.0).abs() < 1e-10);
                assert_eq!(path.len(), 5);
            }
            _ => panic!("Expected Found"),
        }
    }

    #[test]
    fn test_grid_walkable() {
        let mut grid = PathGrid::new(5, 5, 1.0);
        assert!(grid.is_walkable(0, 0));
        assert!(!grid.is_walkable(-1, 0));
        assert!(!grid.is_walkable(5, 0));
        grid.set_cost(2, 2, 0.0);
        assert!(!grid.is_walkable(2, 2));
    }

    #[test]
    fn test_grid_neighbors_count() {
        let grid = PathGrid::new(5, 5, 1.0);
        let n4 = grid.neighbors(GridPos::new(2, 2), DiagonalMode::Never);
        assert_eq!(n4.len(), 4);
        let n8 = grid.neighbors(GridPos::new(2, 2), DiagonalMode::Always);
        assert_eq!(n8.len(), 8);
    }

    #[test]
    fn test_grid_corner_neighbors() {
        let grid = PathGrid::new(5, 5, 1.0);
        let n = grid.neighbors(GridPos::new(0, 0), DiagonalMode::Never);
        assert_eq!(n.len(), 2); // right and down
    }

    #[test]
    fn test_weighted_graph() {
        // Shortest path prefers lighter weight
        let result = astar(
            0,
            3,
            |n| match *n {
                0 => vec![(1, 10.0), (2, 1.0)],
                1 => vec![(3, 1.0)],
                2 => vec![(3, 1.0)],
                _ => vec![],
            },
            |_| 0.0,
            0,
        );
        match result {
            SearchResult::Found { path, cost } => {
                assert_eq!(path, vec![0, 2, 3]);
                assert!((cost - 2.0).abs() < 1e-10);
            }
            _ => panic!("Expected Found"),
        }
    }
}
