//! Room placement for dungeon/level generation.
//!
//! Places non-overlapping rooms in a 2D space using random rejection,
//! grid-based jitter, or Poisson disc spacing. Supports rectangle,
//! L-shape, cross, and circular room shapes. Connects rooms via
//! Delaunay triangulation → MST + extra edges, then routes hallways
//! with A* on the tile grid.

use std::collections::{BinaryHeap, HashMap, HashSet};
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

    fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        lo + t * (hi - lo)
    }
}

// ── RoomShape ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomShape {
    Rectangle,
    LShape,
    Cross,
    Circular,
}

// ── PlacementStrategy ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStrategy {
    RandomRejection,
    GridJitter,
    PoissonDisc,
}

// ── PlacedRoom ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedRoom {
    pub id: usize,
    pub shape: RoomShape,
    pub cells: Vec<(usize, usize)>,
    pub center: (usize, usize),
    pub bounding_x: usize,
    pub bounding_y: usize,
    pub bounding_w: usize,
    pub bounding_h: usize,
}

impl PlacedRoom {
    fn bounding_box(cells: &[(usize, usize)]) -> (usize, usize, usize, usize) {
        let min_x = cells.iter().map(|c| c.0).min().unwrap_or(0);
        let min_y = cells.iter().map(|c| c.1).min().unwrap_or(0);
        let max_x = cells.iter().map(|c| c.0).max().unwrap_or(0);
        let max_y = cells.iter().map(|c| c.1).max().unwrap_or(0);
        (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    }
}

// ── Connection ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connection {
    pub room_a: usize,
    pub room_b: usize,
    pub hallway: Vec<(usize, usize)>,
}

// ── Tile ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Empty,
    Floor,
    Hallway,
    Wall,
}

// ── LevelConfig ──

#[derive(Debug, Clone)]
pub struct LevelConfig {
    pub width: usize,
    pub height: usize,
    pub room_count: usize,
    pub min_room_size: usize,
    pub max_room_size: usize,
    pub shape: RoomShape,
    pub strategy: PlacementStrategy,
    pub extra_connections: f64,
    pub padding: usize,
    pub seed: u64,
}

impl Default for LevelConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 60,
            room_count: 10,
            min_room_size: 4,
            max_room_size: 10,
            shape: RoomShape::Rectangle,
            strategy: PlacementStrategy::RandomRejection,
            extra_connections: 0.2,
            padding: 1,
            seed: 42,
        }
    }
}

// ── Level ──

#[derive(Debug, Clone)]
pub struct Level {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Vec<Tile>>,
    pub rooms: Vec<PlacedRoom>,
    pub connections: Vec<Connection>,
}

impl Level {
    pub fn tile_at(&self, x: usize, y: usize) -> Tile {
        if y < self.height && x < self.width { self.tiles[y][x] } else { Tile::Empty }
    }

    pub fn to_string_grid(&self) -> String {
        self.tiles.iter().map(|row| {
            row.iter().map(|t| match t {
                Tile::Empty => ' ',
                Tile::Floor => '.',
                Tile::Hallway => ',',
                Tile::Wall => '#',
            }).collect::<String>()
        }).collect::<Vec<_>>().join("\n")
    }
}

// ── Generation ──

pub fn generate_level(config: &LevelConfig) -> Level {
    let mut rng = Rng::new(config.seed);
    let mut rooms = place_rooms(config, &mut rng);
    let edges = connect_rooms(&rooms, config, &mut rng);

    let mut tiles = vec![vec![Tile::Empty; config.width]; config.height];

    // Carve rooms
    for room in &rooms {
        for &(x, y) in &room.cells {
            if y < config.height && x < config.width {
                tiles[y][x] = Tile::Floor;
            }
        }
    }

    // Route hallways
    let mut connections = Vec::new();
    for (a, b) in &edges {
        let hallway = route_hallway(&tiles, &rooms[*a], &rooms[*b], config);
        for &(hx, hy) in &hallway {
            if tiles[hy][hx] == Tile::Empty {
                tiles[hy][hx] = Tile::Hallway;
            }
        }
        connections.push(Connection { room_a: *a, room_b: *b, hallway });
    }

    // Add walls around floor/hallway tiles
    let snapshot: Vec<Vec<Tile>> = tiles.clone();
    for y in 0..config.height {
        for x in 0..config.width {
            if snapshot[y][x] == Tile::Empty {
                let adj = [(0isize, 1isize), (0, -1), (1, 0), (-1, 0),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)];
                let near_open = adj.iter().any(|&(dx, dy)| {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx >= 0 && ny >= 0 {
                        let ux = nx as usize;
                        let uy = ny as usize;
                        uy < config.height && ux < config.width
                            && matches!(snapshot[uy][ux], Tile::Floor | Tile::Hallway)
                    } else { false }
                });
                if near_open { tiles[y][x] = Tile::Wall; }
            }
        }
    }

    Level { width: config.width, height: config.height, tiles, rooms, connections }
}

fn place_rooms(config: &LevelConfig, rng: &mut Rng) -> Vec<PlacedRoom> {
    match config.strategy {
        PlacementStrategy::RandomRejection => place_random(config, rng),
        PlacementStrategy::GridJitter => place_grid_jitter(config, rng),
        PlacementStrategy::PoissonDisc => place_poisson(config, rng),
    }
}

fn make_room_cells(
    shape: RoomShape, ox: usize, oy: usize, w: usize, h: usize,
) -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    match shape {
        RoomShape::Rectangle => {
            for dy in 0..h { for dx in 0..w { cells.push((ox + dx, oy + dy)); } }
        }
        RoomShape::LShape => {
            let hw = w / 2;
            let hh = h / 2;
            for dy in 0..h { for dx in 0..hw.max(1) { cells.push((ox + dx, oy + dy)); } }
            for dy in 0..hh.max(1) { for dx in hw..w { cells.push((ox + dx, oy + dy)); } }
        }
        RoomShape::Cross => {
            let third_w = (w / 3).max(1);
            let third_h = (h / 3).max(1);
            // Horizontal bar
            for dy in third_h..(h - third_h).max(third_h + 1) {
                for dx in 0..w { cells.push((ox + dx, oy + dy)); }
            }
            // Vertical bar
            for dy in 0..h {
                for dx in third_w..(w - third_w).max(third_w + 1) {
                    cells.push((ox + dx, oy + dy));
                }
            }
        }
        RoomShape::Circular => {
            let cx = w as f64 / 2.0;
            let cy = h as f64 / 2.0;
            let r = cx.min(cy);
            for dy in 0..h {
                for dx in 0..w {
                    let dist = ((dx as f64 + 0.5 - cx).powi(2) + (dy as f64 + 0.5 - cy).powi(2)).sqrt();
                    if dist <= r { cells.push((ox + dx, oy + dy)); }
                }
            }
        }
    }
    cells.sort();
    cells.dedup();
    cells
}

fn place_random(config: &LevelConfig, rng: &mut Rng) -> Vec<PlacedRoom> {
    let mut rooms = Vec::new();
    let mut occupied: HashSet<(usize, usize)> = HashSet::new();
    let max_attempts = config.room_count * 100;
    let mut attempts = 0;

    while rooms.len() < config.room_count && attempts < max_attempts {
        attempts += 1;
        let w = rng.range(config.min_room_size, config.max_room_size + 1);
        let h = rng.range(config.min_room_size, config.max_room_size + 1);
        if w + config.padding * 2 >= config.width || h + config.padding * 2 >= config.height { continue; }
        let ox = rng.range(config.padding, config.width - w - config.padding);
        let oy = rng.range(config.padding, config.height - h - config.padding);

        let cells = make_room_cells(config.shape, ox, oy, w, h);
        let padded: Vec<(usize, usize)> = cells.iter().flat_map(|&(cx, cy)| {
            let mut expanded = Vec::new();
            let p = config.padding;
            for dy in 0..=p * 2 {
                for dx in 0..=p * 2 {
                    let nx = (cx + dx).saturating_sub(p);
                    let ny = (cy + dy).saturating_sub(p);
                    expanded.push((nx, ny));
                }
            }
            expanded
        }).collect();

        if padded.iter().any(|c| occupied.contains(c)) { continue; }

        for &c in &padded { occupied.insert(c); }
        let (bx, by, bw, bh) = PlacedRoom::bounding_box(&cells);
        let center = (bx + bw / 2, by + bh / 2);
        rooms.push(PlacedRoom {
            id: rooms.len(),
            shape: config.shape,
            cells,
            center,
            bounding_x: bx, bounding_y: by, bounding_w: bw, bounding_h: bh,
        });
    }
    rooms
}

fn place_grid_jitter(config: &LevelConfig, rng: &mut Rng) -> Vec<PlacedRoom> {
    let cols = ((config.room_count as f64).sqrt().ceil()) as usize;
    let rows = (config.room_count + cols - 1) / cols;
    let cell_w = config.width / cols.max(1);
    let cell_h = config.height / rows.max(1);
    let mut rooms = Vec::new();

    for gy in 0..rows {
        for gx in 0..cols {
            if rooms.len() >= config.room_count { break; }
            let w = rng.range(config.min_room_size, config.max_room_size.min(cell_w.saturating_sub(4)) + 1);
            let h = rng.range(config.min_room_size, config.max_room_size.min(cell_h.saturating_sub(4)) + 1);
            let jx = rng.range(0, (cell_w.saturating_sub(w)).max(1));
            let jy = rng.range(0, (cell_h.saturating_sub(h)).max(1));
            let ox = (gx * cell_w + jx).min(config.width.saturating_sub(w + 1));
            let oy = (gy * cell_h + jy).min(config.height.saturating_sub(h + 1));
            let cells = make_room_cells(config.shape, ox, oy, w, h);
            let (bx, by, bw, bh) = PlacedRoom::bounding_box(&cells);
            rooms.push(PlacedRoom {
                id: rooms.len(),
                shape: config.shape,
                cells,
                center: (bx + bw / 2, by + bh / 2),
                bounding_x: bx, bounding_y: by, bounding_w: bw, bounding_h: bh,
            });
        }
    }
    rooms
}

fn place_poisson(config: &LevelConfig, rng: &mut Rng) -> Vec<PlacedRoom> {
    // Simplified Poisson disc: maintain min distance between centers
    let min_dist = (config.max_room_size + config.padding * 2) as f64;
    let mut centers: Vec<(f64, f64)> = Vec::new();
    let mut rooms = Vec::new();
    let max_attempts = config.room_count * 200;
    let mut attempts = 0;

    while rooms.len() < config.room_count && attempts < max_attempts {
        attempts += 1;
        let cx = rng.range_f64(config.padding as f64 + 2.0, config.width as f64 - config.padding as f64 - 2.0);
        let cy = rng.range_f64(config.padding as f64 + 2.0, config.height as f64 - config.padding as f64 - 2.0);

        let too_close = centers.iter().any(|&(ox, oy)| {
            ((cx - ox).powi(2) + (cy - oy).powi(2)).sqrt() < min_dist
        });
        if too_close { continue; }

        let w = rng.range(config.min_room_size, config.max_room_size + 1);
        let h = rng.range(config.min_room_size, config.max_room_size + 1);
        let ox = (cx as usize).saturating_sub(w / 2).min(config.width.saturating_sub(w + 1));
        let oy = (cy as usize).saturating_sub(h / 2).min(config.height.saturating_sub(h + 1));
        let cells = make_room_cells(config.shape, ox, oy, w, h);
        let (bx, by, bw, bh) = PlacedRoom::bounding_box(&cells);
        centers.push((cx, cy));
        rooms.push(PlacedRoom {
            id: rooms.len(),
            shape: config.shape,
            cells,
            center: (bx + bw / 2, by + bh / 2),
            bounding_x: bx, bounding_y: by, bounding_w: bw, bounding_h: bh,
        });
    }
    rooms
}

// ── Connection graph (simplified Delaunay → MST + extras) ──

fn connect_rooms(rooms: &[PlacedRoom], config: &LevelConfig, rng: &mut Rng) -> Vec<(usize, usize)> {
    if rooms.len() < 2 { return Vec::new(); }

    // All-pairs edges sorted by distance
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..rooms.len() {
        for j in (i + 1)..rooms.len() {
            let (ax, ay) = rooms[i].center;
            let (bx, by) = rooms[j].center;
            let dist = (((bx as f64 - ax as f64).powi(2)) + ((by as f64 - ay as f64).powi(2))).sqrt();
            edges.push((dist, i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    // Kruskal MST
    let mut parent: Vec<usize> = (0..rooms.len()).collect();
    fn find(p: &mut Vec<usize>, mut i: usize) -> usize {
        while p[i] != i { p[i] = p[p[i]]; i = p[i]; }
        i
    }

    let mut mst: Vec<(usize, usize)> = Vec::new();
    let mut extra: Vec<(usize, usize)> = Vec::new();

    for &(_, a, b) in &edges {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent[ra] = rb;
            mst.push((a, b));
        } else {
            extra.push((a, b));
        }
    }

    // Add some extra edges for loops
    let extra_count = ((extra.len() as f64) * config.extra_connections).round() as usize;
    let mut result = mst;
    for i in 0..extra_count.min(extra.len()) {
        let idx = rng.range(i, extra.len());
        extra.swap(i, idx);
        result.push(extra[i]);
    }
    result
}

// ── A* hallway routing ──

#[derive(Debug, Clone, Eq, PartialEq)]
struct AStarNode {
    cost: usize,
    heuristic: usize,
    x: usize,
    y: usize,
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.cost + other.heuristic).cmp(&(self.cost + self.heuristic))
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn route_hallway(tiles: &[Vec<Tile>], a: &PlacedRoom, b: &PlacedRoom, config: &LevelConfig) -> Vec<(usize, usize)> {
    let (sx, sy) = a.center;
    let (gx, gy) = b.center;
    let h = config.height;
    let w = config.width;

    let heuristic = |x: usize, y: usize| -> usize {
        let dx = if x > gx { x - gx } else { gx - x };
        let dy = if y > gy { y - gy } else { gy - y };
        dx + dy
    };

    let mut dist: HashMap<(usize, usize), usize> = HashMap::new();
    let mut parent: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut heap = BinaryHeap::new();

    dist.insert((sx, sy), 0);
    heap.push(AStarNode { cost: 0, heuristic: heuristic(sx, sy), x: sx, y: sy });

    while let Some(node) = heap.pop() {
        if node.x == gx && node.y == gy {
            let mut path = Vec::new();
            let mut cur = (gx, gy);
            while cur != (sx, sy) {
                path.push(cur);
                cur = parent[&cur];
            }
            path.push((sx, sy));
            path.reverse();
            return path;
        }

        if node.cost > *dist.get(&(node.x, node.y)).unwrap_or(&usize::MAX) { continue; }

        for (dx, dy) in &[(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
            let nx = node.x as isize + dx;
            let ny = node.y as isize + dy;
            if nx < 0 || ny < 0 { continue; }
            let (ux, uy) = (nx as usize, ny as usize);
            if ux >= w || uy >= h { continue; }

            // Cost: floor/hallway is cheap, empty is moderate, wall is expensive
            let step_cost = match tiles[uy][ux] {
                Tile::Floor | Tile::Hallway => 1,
                Tile::Empty => 2,
                Tile::Wall => 5,
            };
            let new_cost = node.cost + step_cost;
            if new_cost < *dist.get(&(ux, uy)).unwrap_or(&usize::MAX) {
                dist.insert((ux, uy), new_cost);
                parent.insert((ux, uy), (node.x, node.y));
                heap.push(AStarNode { cost: new_cost, heuristic: heuristic(ux, uy), x: ux, y: uy });
            }
        }
    }

    // Fallback: straight line
    let mut path = Vec::new();
    let (mut cx, mut cy) = (sx, sy);
    while cx != gx { path.push((cx, cy)); if cx < gx { cx += 1; } else { cx -= 1; } }
    while cy != gy { path.push((cx, cy)); if cy < gy { cy += 1; } else { cy -= 1; } }
    path.push((gx, gy));
    path
}

// ── Public helpers ──

/// Check if any two rooms' bounding boxes overlap.
pub fn rooms_overlap(rooms: &[PlacedRoom]) -> bool {
    for i in 0..rooms.len() {
        for j in (i + 1)..rooms.len() {
            let a_cells: HashSet<(usize, usize)> = rooms[i].cells.iter().copied().collect();
            if rooms[j].cells.iter().any(|c| a_cells.contains(c)) {
                return true;
            }
        }
    }
    false
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn default_level() -> Level {
        generate_level(&LevelConfig::default())
    }

    #[test]
    fn test_basic_generation() {
        let level = default_level();
        assert!(!level.rooms.is_empty());
        assert!(!level.connections.is_empty());
    }

    #[test]
    fn test_seed_determinism() {
        let a = generate_level(&LevelConfig { seed: 123, ..Default::default() });
        let b = generate_level(&LevelConfig { seed: 123, ..Default::default() });
        assert_eq!(a.rooms.len(), b.rooms.len());
        assert_eq!(a.tiles, b.tiles);
    }

    #[test]
    fn test_different_seeds() {
        let a = generate_level(&LevelConfig { seed: 1, ..Default::default() });
        let b = generate_level(&LevelConfig { seed: 999, ..Default::default() });
        assert_ne!(a.tiles, b.tiles);
    }

    #[test]
    fn test_rooms_within_bounds() {
        let level = default_level();
        for room in &level.rooms {
            for &(x, y) in &room.cells {
                assert!(x < level.width && y < level.height);
            }
        }
    }

    #[test]
    fn test_no_room_overlap_random() {
        let level = generate_level(&LevelConfig {
            strategy: PlacementStrategy::RandomRejection,
            seed: 42,
            ..Default::default()
        });
        assert!(!rooms_overlap(&level.rooms));
    }

    #[test]
    fn test_grid_jitter_strategy() {
        let level = generate_level(&LevelConfig {
            strategy: PlacementStrategy::GridJitter,
            seed: 77,
            ..Default::default()
        });
        assert!(!level.rooms.is_empty());
    }

    #[test]
    fn test_poisson_disc_strategy() {
        let level = generate_level(&LevelConfig {
            strategy: PlacementStrategy::PoissonDisc,
            seed: 55,
            ..Default::default()
        });
        assert!(!level.rooms.is_empty());
    }

    #[test]
    fn test_rectangle_shape() {
        let cells = make_room_cells(RoomShape::Rectangle, 0, 0, 5, 4);
        assert_eq!(cells.len(), 20);
    }

    #[test]
    fn test_l_shape() {
        let cells = make_room_cells(RoomShape::LShape, 0, 0, 6, 6);
        assert!(cells.len() < 36); // L-shape has fewer cells than full rect
        assert!(!cells.is_empty());
    }

    #[test]
    fn test_cross_shape() {
        let cells = make_room_cells(RoomShape::Cross, 0, 0, 9, 9);
        assert!(cells.len() < 81);
        assert!(!cells.is_empty());
    }

    #[test]
    fn test_circular_shape() {
        let cells = make_room_cells(RoomShape::Circular, 0, 0, 8, 8);
        assert!(cells.len() < 64);
        assert!(!cells.is_empty());
    }

    #[test]
    fn test_connections_form_spanning_tree() {
        let level = default_level();
        if level.rooms.len() > 1 {
            // MST connects all rooms => rooms.len()-1 connections minimum
            assert!(level.connections.len() >= level.rooms.len() - 1);
        }
    }

    #[test]
    fn test_hallways_exist() {
        let level = default_level();
        let hallway_count = level.tiles.iter().flat_map(|r| r.iter()).filter(|&&t| t == Tile::Hallway).count();
        if level.rooms.len() > 1 {
            assert!(hallway_count > 0);
        }
    }

    #[test]
    fn test_walls_surround_rooms() {
        let level = default_level();
        let wall_count = level.tiles.iter().flat_map(|r| r.iter()).filter(|&&t| t == Tile::Wall).count();
        assert!(wall_count > 0);
    }

    #[test]
    fn test_to_string_grid() {
        let level = generate_level(&LevelConfig {
            width: 40, height: 30, room_count: 5, seed: 42,
            ..Default::default()
        });
        let s = level.to_string_grid();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 30);
    }

    #[test]
    fn test_bounding_box() {
        let cells = vec![(2, 3), (5, 7), (3, 4)];
        let (bx, by, bw, bh) = PlacedRoom::bounding_box(&cells);
        assert_eq!((bx, by, bw, bh), (2, 3, 4, 5));
    }

    #[test]
    fn test_unique_room_ids() {
        let level = default_level();
        let ids: Vec<usize> = level.rooms.iter().map(|r| r.id).collect();
        let unique: HashSet<usize> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_l_shape_placement() {
        let level = generate_level(&LevelConfig {
            shape: RoomShape::LShape,
            seed: 42,
            ..Default::default()
        });
        assert!(!level.rooms.is_empty());
    }

    #[test]
    fn test_cross_placement() {
        let level = generate_level(&LevelConfig {
            shape: RoomShape::Cross,
            seed: 42,
            ..Default::default()
        });
        assert!(!level.rooms.is_empty());
    }

    #[test]
    fn test_circular_placement() {
        let level = generate_level(&LevelConfig {
            shape: RoomShape::Circular,
            seed: 42,
            ..Default::default()
        });
        assert!(!level.rooms.is_empty());
    }

    #[test]
    fn test_many_seeds() {
        for seed in 0..10u64 {
            let level = generate_level(&LevelConfig { seed, ..Default::default() });
            assert!(!level.rooms.is_empty(), "seed {} produced no rooms", seed);
        }
    }

    #[test]
    fn test_extra_connections() {
        let level = generate_level(&LevelConfig {
            extra_connections: 1.0,
            seed: 42,
            ..Default::default()
        });
        if level.rooms.len() > 2 {
            assert!(level.connections.len() > level.rooms.len() - 1);
        }
    }

    #[test]
    fn test_tile_at_out_of_bounds() {
        let level = default_level();
        assert_eq!(level.tile_at(9999, 9999), Tile::Empty);
    }
}
