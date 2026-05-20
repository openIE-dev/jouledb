//! Corridor/hallway generation between rooms on a tile grid.
//!
//! Provides A*-based pathfinding, multiple corridor styles (straight,
//! wide, organic drunk-walk), L-shaped and Z-shaped connectors, door
//! placement at room entries, corridor straightening, dead-end pruning,
//! and intersection widening.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Ordering;

// ── Seeded RNG ──

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }
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
    fn coin(&mut self) -> bool { self.next_u64() & 1 == 0 }
    fn chance(&mut self, pct: u64) -> bool { self.next_u64() % 100 < pct }
}

// ── Tile ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Wall,
    Floor,
    Corridor,
    Door,
    Empty,
}

// ── CorridorStyle ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorridorStyle {
    Straight,
    Wide,
    Organic,
    LShaped,
    ZShaped,
}

// ── Grid ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Vec<Tile>>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, tiles: vec![vec![Tile::Empty; width]; height] }
    }

    pub fn get(&self, x: usize, y: usize) -> Tile {
        if x < self.width && y < self.height { self.tiles[y][x] } else { Tile::Wall }
    }

    pub fn set(&mut self, x: usize, y: usize, tile: Tile) {
        if x < self.width && y < self.height { self.tiles[y][x] = tile; }
    }

    pub fn count(&self, tile: Tile) -> usize {
        self.tiles.iter().flat_map(|r| r.iter()).filter(|&&t| t == tile).count()
    }

    pub fn to_string_grid(&self) -> String {
        self.tiles.iter().map(|row| {
            row.iter().map(|t| match t {
                Tile::Wall => '#',
                Tile::Floor => '.',
                Tile::Corridor => ',',
                Tile::Door => 'D',
                Tile::Empty => ' ',
            }).collect::<String>()
        }).collect::<Vec<_>>().join("\n")
    }
}

// ── Endpoint ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Endpoint {
    pub x: usize,
    pub y: usize,
}

impl Endpoint {
    pub fn new(x: usize, y: usize) -> Self { Self { x, y } }
}

// ── CorridorResult ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorridorResult {
    pub path: Vec<(usize, usize)>,
    pub doors: Vec<(usize, usize)>,
    pub style: CorridorStyle,
}

// ── A* pathfinder ──

#[derive(Eq, PartialEq)]
struct ANode {
    f: usize,
    g: usize,
    x: usize,
    y: usize,
}
impl Ord for ANode {
    fn cmp(&self, other: &Self) -> Ordering { other.f.cmp(&self.f) }
}
impl PartialOrd for ANode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

fn astar(grid: &Grid, start: Endpoint, goal: Endpoint) -> Vec<(usize, usize)> {
    let heuristic = |x: usize, y: usize| -> usize {
        let dx = if x > goal.x { x - goal.x } else { goal.x - x };
        let dy = if y > goal.y { y - goal.y } else { goal.y - y };
        dx + dy
    };

    let mut g_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut parent: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut heap = BinaryHeap::new();

    g_map.insert((start.x, start.y), 0);
    heap.push(ANode { f: heuristic(start.x, start.y), g: 0, x: start.x, y: start.y });

    while let Some(node) = heap.pop() {
        if node.x == goal.x && node.y == goal.y {
            return reconstruct(&parent, start, goal);
        }
        if node.g > *g_map.get(&(node.x, node.y)).unwrap_or(&usize::MAX) { continue; }

        for (dx, dy) in &[(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
            let nx = node.x as isize + dx;
            let ny = node.y as isize + dy;
            if nx < 0 || ny < 0 { continue; }
            let (ux, uy) = (nx as usize, ny as usize);
            if ux >= grid.width || uy >= grid.height { continue; }

            let step = match grid.tiles[uy][ux] {
                Tile::Floor | Tile::Corridor | Tile::Door => 1,
                Tile::Empty => 2,
                Tile::Wall => 8,
            };
            let ng = node.g + step;
            if ng < *g_map.get(&(ux, uy)).unwrap_or(&usize::MAX) {
                g_map.insert((ux, uy), ng);
                parent.insert((ux, uy), (node.x, node.y));
                heap.push(ANode { f: ng + heuristic(ux, uy), g: ng, x: ux, y: uy });
            }
        }
    }
    // Fallback: direct L-path
    l_path(start, goal, true)
}

fn reconstruct(
    parent: &HashMap<(usize, usize), (usize, usize)>,
    start: Endpoint,
    goal: Endpoint,
) -> Vec<(usize, usize)> {
    let mut path = vec![(goal.x, goal.y)];
    let mut cur = (goal.x, goal.y);
    while cur != (start.x, start.y) {
        match parent.get(&cur) {
            Some(&p) => { path.push(p); cur = p; }
            None => break,
        }
    }
    path.reverse();
    path
}

fn l_path(start: Endpoint, goal: Endpoint, horizontal_first: bool) -> Vec<(usize, usize)> {
    let mut path = Vec::new();
    let (mut cx, mut cy) = (start.x, start.y);
    if horizontal_first {
        while cx != goal.x { path.push((cx, cy)); if cx < goal.x { cx += 1; } else { cx -= 1; } }
        while cy != goal.y { path.push((cx, cy)); if cy < goal.y { cy += 1; } else { cy -= 1; } }
    } else {
        while cy != goal.y { path.push((cx, cy)); if cy < goal.y { cy += 1; } else { cy -= 1; } }
        while cx != goal.x { path.push((cx, cy)); if cx < goal.x { cx += 1; } else { cx -= 1; } }
    }
    path.push((goal.x, goal.y));
    path
}

fn z_path(start: Endpoint, goal: Endpoint) -> Vec<(usize, usize)> {
    let mid_y = if start.y < goal.y {
        start.y + (goal.y - start.y) / 2
    } else {
        goal.y + (start.y - goal.y) / 2
    };
    let mut path = Vec::new();
    let (mut cx, mut cy) = (start.x, start.y);
    // Vertical to mid
    while cy != mid_y { path.push((cx, cy)); if cy < mid_y { cy += 1; } else { cy -= 1; } }
    // Horizontal
    while cx != goal.x { path.push((cx, cy)); if cx < goal.x { cx += 1; } else { cx -= 1; } }
    // Vertical to goal
    while cy != goal.y { path.push((cx, cy)); if cy < goal.y { cy += 1; } else { cy -= 1; } }
    path.push((goal.x, goal.y));
    path
}

fn organic_path(start: Endpoint, goal: Endpoint, rng: &mut Rng) -> Vec<(usize, usize)> {
    let mut path = Vec::new();
    let (mut cx, mut cy) = (start.x as isize, start.y as isize);
    let (gx, gy) = (goal.x as isize, goal.y as isize);

    while cx != gx || cy != gy {
        path.push((cx as usize, cy as usize));
        // 70% chance to move toward goal, 30% random orthogonal
        if rng.chance(70) {
            let dx = (gx - cx).signum();
            let dy = (gy - cy).signum();
            if dx != 0 && (dy == 0 || rng.coin()) {
                cx += dx;
            } else if dy != 0 {
                cy += dy;
            }
        } else {
            match rng.range(0, 4) {
                0 => cx += 1,
                1 => { if cx > 0 { cx -= 1; } }
                2 => cy += 1,
                _ => { if cy > 0 { cy -= 1; } }
            }
        }
        // Prevent going too far off course
        let dist_now = (cx - gx).unsigned_abs() + (cy - gy).unsigned_abs();
        let orig = (start.x as isize - gx).unsigned_abs() + (start.y as isize - gy).unsigned_abs();
        if dist_now > orig * 2 {
            // Snap back toward goal
            let dx = (gx - cx).signum();
            let dy = (gy - cy).signum();
            if dx != 0 { cx += dx; } else { cy += dy; }
        }
        if path.len() > (orig as usize + 1) * 4 { break; } // safety
    }
    path.push((goal.x, goal.y));
    path
}

// ── Public API ──

/// Carve a corridor on the grid between two endpoints.
pub fn link_corridor(
    grid: &mut Grid,
    start: Endpoint,
    goal: Endpoint,
    style: CorridorStyle,
    seed: u64,
) -> CorridorResult {
    let mut rng = Rng::new(seed);

    let raw_path = match style {
        CorridorStyle::Straight | CorridorStyle::Wide => astar(grid, start, goal),
        CorridorStyle::LShaped => l_path(start, goal, rng.coin()),
        CorridorStyle::ZShaped => z_path(start, goal),
        CorridorStyle::Organic => organic_path(start, goal, &mut rng),
    };

    let path = straighten_path(&raw_path);
    let mut doors = Vec::new();

    for &(px, py) in &path {
        if px < grid.width && py < grid.height {
            let current = grid.tiles[py][px];
            if current == Tile::Floor { continue; } // Don't overwrite rooms

            let at_room_edge = is_adjacent_to(grid, px, py, Tile::Floor);
            if at_room_edge && current != Tile::Door {
                grid.set(px, py, Tile::Door);
                doors.push((px, py));
            } else if current != Tile::Door {
                grid.set(px, py, Tile::Corridor);
            }
        }
    }

    // Wide corridors: expand path by 1 tile perpendicular
    if style == CorridorStyle::Wide {
        widen_corridor(grid, &path);
    }

    CorridorResult { path, doors, style }
}

fn is_adjacent_to(grid: &Grid, x: usize, y: usize, tile: Tile) -> bool {
    for (dx, dy) in &[(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx >= 0 && ny >= 0 {
            let (ux, uy) = (nx as usize, ny as usize);
            if ux < grid.width && uy < grid.height && grid.tiles[uy][ux] == tile {
                return true;
            }
        }
    }
    false
}

fn widen_corridor(grid: &mut Grid, path: &[(usize, usize)]) {
    for window in path.windows(2) {
        let (x1, y1) = window[0];
        let (x2, y2) = window[1];
        // Perpendicular expansion
        if x1 != x2 {
            // Horizontal movement: expand vertically
            for &(px, py) in &[window[0], window[1]] {
                if py + 1 < grid.height && grid.tiles[py + 1][px] != Tile::Floor {
                    grid.set(px, py + 1, Tile::Corridor);
                }
            }
        } else if y1 != y2 {
            // Vertical movement: expand horizontally
            for &(px, py) in &[window[0], window[1]] {
                if px + 1 < grid.width && grid.tiles[py][px + 1] != Tile::Floor {
                    grid.set(px + 1, py, Tile::Corridor);
                }
            }
        }
    }
}

/// Remove unnecessary turns from a path (straightening pass).
pub fn straighten_path(path: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if path.len() <= 2 { return path.to_vec(); }
    let mut result = vec![path[0]];
    for i in 1..path.len() - 1 {
        let (px, py) = path[i - 1];
        let (cx, cy) = path[i];
        let (nx, ny) = path[i + 1];
        // Keep the point if direction changes
        let d1x = cx as isize - px as isize;
        let d1y = cy as isize - py as isize;
        let d2x = nx as isize - cx as isize;
        let d2y = ny as isize - cy as isize;
        if d1x != d2x || d1y != d2y {
            result.push(path[i]);
        }
    }
    result.push(*path.last().unwrap());

    // Re-expand to full path
    let mut full = Vec::new();
    for window in result.windows(2) {
        let (x1, y1) = window[0];
        let (x2, y2) = window[1];
        let mut cx = x1 as isize;
        let mut cy = y1 as isize;
        let gx = x2 as isize;
        let gy = y2 as isize;
        while cx != gx || cy != gy {
            full.push((cx as usize, cy as usize));
            if cx != gx { cx += (gx - cx).signum(); }
            else { cy += (gy - cy).signum(); }
        }
    }
    if let Some(&last) = path.last() {
        if full.last().copied() != Some(last) { full.push(last); }
    }
    full
}

/// Widen tiles at corridor intersections.
pub fn widen_intersections(grid: &mut Grid) {
    let snapshot: Vec<Vec<Tile>> = grid.tiles.clone();
    for y in 1..grid.height.saturating_sub(1) {
        for x in 1..grid.width.saturating_sub(1) {
            if snapshot[y][x] != Tile::Corridor { continue; }
            let adj_corridors = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .filter(|&&(dx, dy)| {
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    nx < grid.width && ny < grid.height && snapshot[ny][nx] == Tile::Corridor
                })
                .count();
            if adj_corridors >= 3 {
                // Widen: set diagonals to corridor
                for (dx, dy) in &[(1isize, 1isize), (1, -1), (-1, 1), (-1, -1)] {
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    if nx < grid.width && ny < grid.height && snapshot[ny][nx] == Tile::Empty {
                        grid.set(nx, ny, Tile::Corridor);
                    }
                }
            }
        }
    }
}

/// Remove dead-end corridor tiles (tiles with only one corridor neighbor).
pub fn prune_dead_ends(grid: &mut Grid) {
    let mut changed = true;
    while changed {
        changed = false;
        for y in 0..grid.height {
            for x in 0..grid.width {
                if grid.tiles[y][x] != Tile::Corridor { continue; }
                let adj = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .filter(|&&(dx, dy)| {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx >= 0 && ny >= 0 {
                            let (ux, uy) = (nx as usize, ny as usize);
                            ux < grid.width && uy < grid.height
                                && matches!(grid.tiles[uy][ux], Tile::Corridor | Tile::Floor | Tile::Door)
                        } else { false }
                    })
                    .count();
                if adj <= 1 {
                    grid.set(x, y, Tile::Empty);
                    changed = true;
                }
            }
        }
    }
}

/// Place doors at every corridor tile adjacent to a floor tile.
pub fn place_doors(grid: &mut Grid) -> Vec<(usize, usize)> {
    let mut doors = Vec::new();
    let snapshot: Vec<Vec<Tile>> = grid.tiles.clone();
    for y in 0..grid.height {
        for x in 0..grid.width {
            if snapshot[y][x] != Tile::Corridor { continue; }
            if is_adjacent_to_snapshot(&snapshot, x, y, Tile::Floor, grid.width, grid.height) {
                grid.set(x, y, Tile::Door);
                doors.push((x, y));
            }
        }
    }
    doors
}

fn is_adjacent_to_snapshot(tiles: &[Vec<Tile>], x: usize, y: usize, tile: Tile, w: usize, h: usize) -> bool {
    for (dx, dy) in &[(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx >= 0 && ny >= 0 {
            let (ux, uy) = (nx as usize, ny as usize);
            if ux < w && uy < h && tiles[uy][ux] == tile { return true; }
        }
    }
    false
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_with_rooms() -> Grid {
        let mut grid = Grid::new(30, 20);
        // Room A at (2,2)-(7,7)
        for y in 2..8 { for x in 2..8 { grid.set(x, y, Tile::Floor); } }
        // Room B at (20,12)-(27,17)
        for y in 12..18 { for x in 20..28 { grid.set(x, y, Tile::Floor); } }
        grid
    }

    #[test]
    fn test_astar_basic() {
        let mut grid = make_grid_with_rooms();
        let result = link_corridor(
            &mut grid,
            Endpoint::new(5, 5),
            Endpoint::new(23, 14),
            CorridorStyle::Straight,
            42,
        );
        assert!(!result.path.is_empty());
        assert_eq!(*result.path.first().unwrap(), (5, 5));
        assert_eq!(*result.path.last().unwrap(), (23, 14));
    }

    #[test]
    fn test_l_shaped_corridor() {
        let mut grid = Grid::new(20, 20);
        let result = link_corridor(
            &mut grid,
            Endpoint::new(2, 2),
            Endpoint::new(15, 15),
            CorridorStyle::LShaped,
            42,
        );
        assert!(!result.path.is_empty());
        assert!(grid.count(Tile::Corridor) > 0);
    }

    #[test]
    fn test_z_shaped_corridor() {
        let mut grid = Grid::new(20, 20);
        let result = link_corridor(
            &mut grid,
            Endpoint::new(3, 3),
            Endpoint::new(16, 16),
            CorridorStyle::ZShaped,
            42,
        );
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_organic_corridor() {
        let mut grid = Grid::new(30, 30);
        let result = link_corridor(
            &mut grid,
            Endpoint::new(5, 5),
            Endpoint::new(25, 25),
            CorridorStyle::Organic,
            42,
        );
        assert!(!result.path.is_empty());
        assert_eq!(*result.path.last().unwrap(), (25, 25));
    }

    #[test]
    fn test_wide_corridor() {
        let mut grid = Grid::new(20, 20);
        let result = link_corridor(
            &mut grid,
            Endpoint::new(2, 10),
            Endpoint::new(17, 10),
            CorridorStyle::Wide,
            42,
        );
        assert!(!result.path.is_empty());
        // Wide corridors produce extra tiles
        assert!(grid.count(Tile::Corridor) > result.path.len() / 2);
    }

    #[test]
    fn test_door_placement() {
        let mut grid = make_grid_with_rooms();
        let result = link_corridor(
            &mut grid,
            Endpoint::new(5, 5),
            Endpoint::new(23, 14),
            CorridorStyle::Straight,
            42,
        );
        assert!(!result.doors.is_empty(), "doors should be placed at room entries");
        for &(dx, dy) in &result.doors {
            assert_eq!(grid.tiles[dy][dx], Tile::Door);
        }
    }

    #[test]
    fn test_straighten_path() {
        let path = vec![(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)];
        let straightened = straighten_path(&path);
        assert_eq!(*straightened.first().unwrap(), (0, 0));
        assert_eq!(*straightened.last().unwrap(), (2, 2));
    }

    #[test]
    fn test_straighten_empty() {
        let path: Vec<(usize, usize)> = Vec::new();
        let result = straighten_path(&path);
        assert!(result.is_empty());
    }

    #[test]
    fn test_prune_dead_ends() {
        let mut grid = Grid::new(10, 10);
        // Create a corridor with a dead end
        for x in 2..8 { grid.set(x, 5, Tile::Corridor); }
        grid.set(7, 4, Tile::Corridor); // dead end
        grid.set(7, 3, Tile::Corridor); // dead end extension
        prune_dead_ends(&mut grid);
        // The dead end branch should be removed, but main corridor partially pruned from ends too
        // Just verify dead end was reduced
        let corridor_count = grid.count(Tile::Corridor);
        assert!(corridor_count < 8); // started with 8, some pruned
    }

    #[test]
    fn test_widen_intersections() {
        let mut grid = Grid::new(10, 10);
        grid.set(5, 5, Tile::Corridor);
        grid.set(5, 4, Tile::Corridor);
        grid.set(5, 6, Tile::Corridor);
        grid.set(4, 5, Tile::Corridor);
        grid.set(6, 5, Tile::Corridor);
        let before = grid.count(Tile::Corridor);
        widen_intersections(&mut grid);
        let after = grid.count(Tile::Corridor);
        assert!(after > before, "intersection should be widened");
    }

    #[test]
    fn test_place_doors_function() {
        let mut grid = Grid::new(15, 15);
        for y in 2..6 { for x in 2..6 { grid.set(x, y, Tile::Floor); } }
        for x in 6..10 { grid.set(x, 3, Tile::Corridor); }
        let doors = place_doors(&mut grid);
        assert!(!doors.is_empty());
    }

    #[test]
    fn test_grid_to_string() {
        let mut grid = Grid::new(5, 5);
        grid.set(2, 2, Tile::Corridor);
        let s = grid.to_string_grid();
        assert!(s.contains(','));
    }

    #[test]
    fn test_grid_count() {
        let mut grid = Grid::new(10, 10);
        for x in 0..10 { grid.set(x, 5, Tile::Corridor); }
        assert_eq!(grid.count(Tile::Corridor), 10);
        assert_eq!(grid.count(Tile::Empty), 90);
    }

    #[test]
    fn test_seed_determinism() {
        let mut g1 = Grid::new(20, 20);
        let mut g2 = Grid::new(20, 20);
        link_corridor(&mut g1, Endpoint::new(2, 2), Endpoint::new(17, 17), CorridorStyle::Organic, 42);
        link_corridor(&mut g2, Endpoint::new(2, 2), Endpoint::new(17, 17), CorridorStyle::Organic, 42);
        assert_eq!(g1.tiles, g2.tiles);
    }

    #[test]
    fn test_corridor_does_not_overwrite_floor() {
        let mut grid = make_grid_with_rooms();
        link_corridor(
            &mut grid,
            Endpoint::new(5, 5),
            Endpoint::new(23, 14),
            CorridorStyle::Straight,
            42,
        );
        // Room A tiles should still be floor
        for y in 3..7 { for x in 3..7 { assert_eq!(grid.tiles[y][x], Tile::Floor); } }
    }

    #[test]
    fn test_same_point_corridor() {
        let mut grid = Grid::new(10, 10);
        let result = link_corridor(
            &mut grid,
            Endpoint::new(5, 5),
            Endpoint::new(5, 5),
            CorridorStyle::Straight,
            42,
        );
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_adjacent_points() {
        let mut grid = Grid::new(10, 10);
        let result = link_corridor(
            &mut grid,
            Endpoint::new(5, 5),
            Endpoint::new(6, 5),
            CorridorStyle::LShaped,
            42,
        );
        assert!(result.path.len() <= 3);
    }

    #[test]
    fn test_multiple_corridors() {
        let mut grid = Grid::new(30, 30);
        for y in 2..6 { for x in 2..6 { grid.set(x, y, Tile::Floor); } }
        for y in 2..6 { for x in 20..26 { grid.set(x, y, Tile::Floor); } }
        for y in 20..26 { for x in 2..6 { grid.set(x, y, Tile::Floor); } }

        link_corridor(&mut grid, Endpoint::new(4, 4), Endpoint::new(22, 4), CorridorStyle::Straight, 1);
        link_corridor(&mut grid, Endpoint::new(4, 4), Endpoint::new(4, 22), CorridorStyle::Straight, 2);
        assert!(grid.count(Tile::Corridor) > 0 || grid.count(Tile::Door) > 0);
    }

    #[test]
    fn test_endpoint_new() {
        let e = Endpoint::new(10, 20);
        assert_eq!(e.x, 10);
        assert_eq!(e.y, 20);
    }

    #[test]
    fn test_corridor_result_style() {
        let mut grid = Grid::new(20, 20);
        let result = link_corridor(
            &mut grid,
            Endpoint::new(2, 2),
            Endpoint::new(15, 15),
            CorridorStyle::ZShaped,
            42,
        );
        assert_eq!(result.style, CorridorStyle::ZShaped);
    }
}
