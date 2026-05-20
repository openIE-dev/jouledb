// Procedural Terrain Generation — Cave system generation
// Cellular automata, 3D random walk, stalactite/stalagmite placement,
// room detection, tunnel connection (A*), multi-layer caves

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Reverse;
use std::fmt;

/// Cell state in a cave grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaveCell {
    Wall,
    Air,
}

/// A detected room in the cave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaveRoom {
    pub id: u32,
    pub cells: Vec<(usize, usize)>,
    pub center: (usize, usize),
}

/// Feature that can be placed in a cave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaveFeature {
    Stalactite,
    Stalagmite,
    EntryPoint,
    ExitPoint,
}

/// A placed feature with its position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedFeature {
    pub feature: CaveFeature,
    pub x: usize,
    pub y: usize,
}

/// Configuration for cave generation.
#[derive(Debug, Clone, PartialEq)]
pub struct CaveConfig {
    pub width: usize,
    pub height: usize,
    pub fill_probability: f64,
    pub automata_iterations: u32,
    pub birth_limit: u32,
    pub death_limit: u32,
    pub seed: u64,
    pub min_room_size: usize,
}

impl Default for CaveConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            fill_probability: 0.45,
            automata_iterations: 5,
            birth_limit: 5,
            death_limit: 4,
            seed: 42,
            min_room_size: 10,
        }
    }
}

/// A 2D cave grid.
#[derive(Clone)]
pub struct CaveGrid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<CaveCell>,
}

impl fmt::Debug for CaveGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaveGrid")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl CaveGrid {
    pub fn new(width: usize, height: usize, default: CaveCell) -> Self {
        Self {
            width,
            height,
            cells: vec![default; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> CaveCell {
        if x < self.width && y < self.height {
            self.cells[y * self.width + x]
        } else {
            CaveCell::Wall
        }
    }

    pub fn set(&mut self, x: usize, y: usize, cell: CaveCell) {
        if x < self.width && y < self.height {
            self.cells[y * self.width + x] = cell;
        }
    }

    /// Count wall neighbors in the Moore neighborhood (8-connected).
    pub fn count_wall_neighbors(&self, x: usize, y: usize) -> u32 {
        let mut count = 0u32;
        for dy in [-1i64, 0, 1] {
            for dx in [-1i64, 0, 1] {
                if dx == 0 && dy == 0 { continue; }
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx < 0 || nx >= self.width as i64 || ny < 0 || ny >= self.height as i64 {
                    count += 1; // Out of bounds counts as wall
                } else if self.cells[ny as usize * self.width + nx as usize] == CaveCell::Wall {
                    count += 1;
                }
            }
        }
        count
    }

    /// Count air cells in the grid.
    pub fn air_count(&self) -> usize {
        self.cells.iter().filter(|c| **c == CaveCell::Air).count()
    }

    /// Count wall cells in the grid.
    pub fn wall_count(&self) -> usize {
        self.cells.iter().filter(|c| **c == CaveCell::Wall).count()
    }
}

/// Cave generator.
pub struct CaveGenerator {
    config: CaveConfig,
}

impl fmt::Debug for CaveGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaveGenerator").field("config", &self.config).finish()
    }
}

impl CaveGenerator {
    pub fn new(config: CaveConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &CaveConfig {
        &self.config
    }

    fn next_rng(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 32) as f64 / (u32::MAX as f64)
    }

    fn next_rng_int(state: &mut u64, max: usize) -> usize {
        if max == 0 { return 0; }
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % max
    }

    /// Step 1: Random fill based on fill probability.
    pub fn random_fill(&self) -> CaveGrid {
        let mut grid = CaveGrid::new(self.config.width, self.config.height, CaveCell::Wall);
        let mut rng = self.config.seed;

        for y in 0..self.config.height {
            for x in 0..self.config.width {
                // Keep borders as walls
                if x == 0 || x == self.config.width - 1
                    || y == 0 || y == self.config.height - 1
                {
                    continue;
                }
                if Self::next_rng(&mut rng) > self.config.fill_probability {
                    grid.set(x, y, CaveCell::Air);
                }
            }
        }
        grid
    }

    /// Step 2: Apply cellular automata (B5678/S45678 variant).
    pub fn automata_step(&self, grid: &CaveGrid) -> CaveGrid {
        let mut new_grid = CaveGrid::new(grid.width, grid.height, CaveCell::Wall);

        for y in 1..(grid.height - 1) {
            for x in 1..(grid.width - 1) {
                let walls = grid.count_wall_neighbors(x, y);
                let cell = match grid.get(x, y) {
                    CaveCell::Wall => {
                        // Birth: wall becomes air if few enough wall neighbors
                        if walls < self.config.birth_limit {
                            CaveCell::Air
                        } else {
                            CaveCell::Wall
                        }
                    }
                    CaveCell::Air => {
                        // Death: air becomes wall if enough wall neighbors
                        if walls > self.config.death_limit {
                            CaveCell::Wall
                        } else {
                            CaveCell::Air
                        }
                    }
                };
                new_grid.set(x, y, cell);
            }
        }
        new_grid
    }

    /// Generate a cave using cellular automata.
    pub fn generate(&self) -> CaveGrid {
        let mut grid = self.random_fill();
        for _ in 0..self.config.automata_iterations {
            grid = self.automata_step(&grid);
        }
        grid
    }

    /// Detect connected rooms using flood fill.
    pub fn detect_rooms(&self, grid: &CaveGrid) -> Vec<CaveRoom> {
        let mut visited = vec![false; grid.width * grid.height];
        let mut rooms = Vec::new();
        let mut room_id = 0u32;

        for y in 0..grid.height {
            for x in 0..grid.width {
                let idx = y * grid.width + x;
                if visited[idx] || grid.cells[idx] != CaveCell::Air {
                    continue;
                }

                // Flood fill
                let mut cells = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back((x, y));
                visited[idx] = true;

                while let Some((cx, cy)) = queue.pop_front() {
                    cells.push((cx, cy));

                    for (dx, dy) in [(0i64, 1), (0, -1), (1, 0), (-1, 0)] {
                        let nx = cx as i64 + dx;
                        let ny = cy as i64 + dy;
                        if nx >= 0 && nx < grid.width as i64
                            && ny >= 0 && ny < grid.height as i64
                        {
                            let ni = ny as usize * grid.width + nx as usize;
                            if !visited[ni] && grid.cells[ni] == CaveCell::Air {
                                visited[ni] = true;
                                queue.push_back((nx as usize, ny as usize));
                            }
                        }
                    }
                }

                if cells.len() >= self.config.min_room_size {
                    let (sum_x, sum_y) = cells.iter().fold((0usize, 0usize), |(sx, sy), (cx, cy)| {
                        (sx + cx, sy + cy)
                    });
                    let center = (sum_x / cells.len(), sum_y / cells.len());
                    rooms.push(CaveRoom { id: room_id, cells, center });
                    room_id += 1;
                }
            }
        }
        rooms
    }

    /// Connect two rooms by carving a tunnel (A* pathfinding).
    pub fn connect_rooms(grid: &mut CaveGrid, from: (usize, usize), to: (usize, usize)) {
        let w = grid.width;
        let h = grid.height;

        #[derive(Eq, PartialEq)]
        struct Node {
            cost: u64,
            pos: (usize, usize),
        }

        impl Ord for Node {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.cost.cmp(&self.cost)
            }
        }
        impl PartialOrd for Node {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let heuristic = |a: (usize, usize), b: (usize, usize)| -> u64 {
            let dx = if a.0 > b.0 { a.0 - b.0 } else { b.0 - a.0 };
            let dy = if a.1 > b.1 { a.1 - b.1 } else { b.1 - a.1 };
            (dx + dy) as u64
        };

        let mut open = BinaryHeap::new();
        let mut g_score: HashMap<(usize, usize), u64> = HashMap::new();
        let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();

        g_score.insert(from, 0);
        open.push(Node { cost: heuristic(from, to), pos: from });

        while let Some(Node { pos, .. }) = open.pop() {
            if pos == to {
                // Reconstruct path and carve
                let mut current = to;
                while current != from {
                    // Carve a 1-cell-wide tunnel
                    grid.set(current.0, current.1, CaveCell::Air);
                    if let Some(&prev) = came_from.get(&current) {
                        current = prev;
                    } else {
                        break;
                    }
                }
                grid.set(from.0, from.1, CaveCell::Air);
                return;
            }

            let current_g = g_score.get(&pos).copied().unwrap_or(u64::MAX);

            for (dx, dy) in [(0i64, 1), (0, -1), (1, 0), (-1, 0)] {
                let nx = pos.0 as i64 + dx;
                let ny = pos.1 as i64 + dy;
                if nx < 0 || nx >= w as i64 || ny < 0 || ny >= h as i64 {
                    continue;
                }
                let next = (nx as usize, ny as usize);
                // Walls cost more to traverse (need to carve)
                let move_cost = if grid.get(next.0, next.1) == CaveCell::Wall { 5 } else { 1 };
                let tentative_g = current_g.saturating_add(move_cost);

                if tentative_g < g_score.get(&next).copied().unwrap_or(u64::MAX) {
                    g_score.insert(next, tentative_g);
                    came_from.insert(next, pos);
                    open.push(Node {
                        cost: tentative_g + heuristic(next, to),
                        pos: next,
                    });
                }
            }
        }
    }

    /// Connect all rooms by carving tunnels between their centers.
    pub fn connect_all_rooms(grid: &mut CaveGrid, rooms: &[CaveRoom]) {
        if rooms.len() < 2 { return; }
        for i in 0..(rooms.len() - 1) {
            Self::connect_rooms(grid, rooms[i].center, rooms[i + 1].center);
        }
    }

    /// Place stalactites at ceiling cells (air above wall below).
    pub fn place_features(&self, grid: &CaveGrid) -> Vec<PlacedFeature> {
        let mut features = Vec::new();
        let mut rng = self.config.seed.wrapping_add(9999);

        for y in 1..(grid.height - 1) {
            for x in 1..(grid.width - 1) {
                if grid.get(x, y) != CaveCell::Air { continue; }

                // Stalactite: wall above, air at current
                if grid.get(x, y - 1) == CaveCell::Wall {
                    if Self::next_rng(&mut rng) < 0.15 {
                        features.push(PlacedFeature {
                            feature: CaveFeature::Stalactite,
                            x, y,
                        });
                    }
                }

                // Stalagmite: wall below, air at current
                if grid.get(x, y + 1) == CaveCell::Wall {
                    if Self::next_rng(&mut rng) < 0.15 {
                        features.push(PlacedFeature {
                            feature: CaveFeature::Stalagmite,
                            x, y,
                        });
                    }
                }
            }
        }
        features
    }

    /// Place entry and exit points on the edge of air regions.
    pub fn place_entry_exit(&self, grid: &CaveGrid, rooms: &[CaveRoom]) -> Vec<PlacedFeature> {
        let mut features = Vec::new();
        if rooms.is_empty() { return features; }

        // Entry: leftmost air cell in the first room
        let first_room = &rooms[0];
        if let Some(&(ex, ey)) = first_room.cells.iter()
            .min_by_key(|&&(cx, _)| cx)
        {
            features.push(PlacedFeature { feature: CaveFeature::EntryPoint, x: ex, y: ey });
        }

        // Exit: rightmost air cell in the last room
        let last_room = &rooms[rooms.len() - 1];
        if let Some(&(ex, ey)) = last_room.cells.iter()
            .max_by_key(|&&(cx, _)| cx)
        {
            features.push(PlacedFeature { feature: CaveFeature::ExitPoint, x: ex, y: ey });
        }

        features
    }

    /// Generate cave layers at different depths with varying fill probability.
    pub fn generate_layers(&self, num_layers: u32) -> Vec<CaveGrid> {
        let mut layers = Vec::new();
        for layer in 0..num_layers {
            let depth_factor = layer as f64 / num_layers.max(1) as f64;
            let fill_prob = self.config.fill_probability + depth_factor * 0.15;
            let layer_config = CaveConfig {
                fill_probability: fill_prob.min(0.7),
                seed: self.config.seed.wrapping_add(layer as u64 * 1000),
                ..self.config.clone()
            };
            let layer_gen = CaveGenerator::new(layer_config);
            layers.push(layer_gen.generate());
        }
        layers
    }

    /// 3D cave carving using random walk with branching.
    pub fn random_walk_3d(
        width: usize,
        height: usize,
        depth: usize,
        seed: u64,
        walk_length: u32,
        branch_prob: f64,
    ) -> Vec<Vec<Vec<CaveCell>>> {
        let mut volume = vec![vec![vec![CaveCell::Wall; width]; height]; depth];
        let mut rng = seed;

        let mut walks: Vec<(usize, usize, usize, u32)> = vec![
            (width / 2, height / 2, depth / 2, walk_length),
        ];

        while let Some((mut x, mut y, mut z, remaining)) = walks.pop() {
            for _ in 0..remaining {
                if x > 0 && x < width - 1 && y > 0 && y < height - 1 && z > 0 && z < depth - 1 {
                    volume[z][y][x] = CaveCell::Air;
                }

                let dir = Self::next_rng_int(&mut rng, 6);
                match dir {
                    0 if x > 1 => x -= 1,
                    1 if x < width - 2 => x += 1,
                    2 if y > 1 => y -= 1,
                    3 if y < height - 2 => y += 1,
                    4 if z > 1 => z -= 1,
                    5 if z < depth - 2 => z += 1,
                    _ => {}
                }

                // Branching
                if Self::next_rng(&mut rng) < branch_prob && remaining > 10 {
                    walks.push((x, y, z, remaining / 3));
                }
            }
        }
        volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cave_grid_new() {
        let g = CaveGrid::new(16, 16, CaveCell::Wall);
        assert_eq!(g.width, 16);
        assert_eq!(g.height, 16);
        assert_eq!(g.wall_count(), 256);
        assert_eq!(g.air_count(), 0);
    }

    #[test]
    fn test_cave_grid_get_set() {
        let mut g = CaveGrid::new(8, 8, CaveCell::Wall);
        g.set(3, 4, CaveCell::Air);
        assert_eq!(g.get(3, 4), CaveCell::Air);
        assert_eq!(g.get(0, 0), CaveCell::Wall);
    }

    #[test]
    fn test_cave_grid_out_of_bounds() {
        let g = CaveGrid::new(8, 8, CaveCell::Air);
        assert_eq!(g.get(100, 100), CaveCell::Wall);
    }

    #[test]
    fn test_count_wall_neighbors() {
        let g = CaveGrid::new(8, 8, CaveCell::Wall);
        assert_eq!(g.count_wall_neighbors(4, 4), 8);

        let g2 = CaveGrid::new(8, 8, CaveCell::Air);
        assert_eq!(g2.count_wall_neighbors(4, 4), 0);
    }

    #[test]
    fn test_random_fill_has_both() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let grid = cave_gen.random_fill();
        assert!(grid.air_count() > 0);
        assert!(grid.wall_count() > 0);
    }

    #[test]
    fn test_random_fill_borders_are_walls() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let grid = cave_gen.random_fill();
        for x in 0..grid.width {
            assert_eq!(grid.get(x, 0), CaveCell::Wall);
            assert_eq!(grid.get(x, grid.height - 1), CaveCell::Wall);
        }
        for y in 0..grid.height {
            assert_eq!(grid.get(0, y), CaveCell::Wall);
            assert_eq!(grid.get(grid.width - 1, y), CaveCell::Wall);
        }
    }

    #[test]
    fn test_automata_step_does_not_crash() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let grid = cave_gen.random_fill();
        let stepped = cave_gen.automata_step(&grid);
        assert_eq!(stepped.width, grid.width);
    }

    #[test]
    fn test_generate_produces_caves() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let grid = cave_gen.generate();
        assert!(grid.air_count() > 0);
    }

    #[test]
    fn test_generate_deterministic() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let g1 = cave_gen.generate();
        let g2 = cave_gen.generate();
        assert_eq!(g1.cells, g2.cells);
    }

    #[test]
    fn test_detect_rooms() {
        let cave_gen = CaveGenerator::new(CaveConfig { min_room_size: 5, ..CaveConfig::default() });
        let grid = cave_gen.generate();
        let rooms = cave_gen.detect_rooms(&grid);
        for room in &rooms {
            assert!(room.cells.len() >= 5);
        }
    }

    #[test]
    fn test_room_center_within_bounds() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let grid = cave_gen.generate();
        let rooms = cave_gen.detect_rooms(&grid);
        for room in &rooms {
            assert!(room.center.0 < grid.width);
            assert!(room.center.1 < grid.height);
        }
    }

    #[test]
    fn test_connect_rooms() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let mut grid = cave_gen.generate();
        let rooms = cave_gen.detect_rooms(&grid);
        if rooms.len() >= 2 {
            CaveGenerator::connect_rooms(&mut grid, rooms[0].center, rooms[1].center);
            // Path should exist (all air) from room0.center to room1.center
            assert_eq!(grid.get(rooms[0].center.0, rooms[0].center.1), CaveCell::Air);
        }
    }

    #[test]
    fn test_connect_all_rooms() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let mut grid = cave_gen.generate();
        let rooms = cave_gen.detect_rooms(&grid);
        CaveGenerator::connect_all_rooms(&mut grid, &rooms);
        // Should not crash
    }

    #[test]
    fn test_place_features() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let grid = cave_gen.generate();
        let features = cave_gen.place_features(&grid);
        for f in &features {
            assert!(f.x < grid.width && f.y < grid.height);
            assert!(f.feature == CaveFeature::Stalactite || f.feature == CaveFeature::Stalagmite);
        }
    }

    #[test]
    fn test_place_entry_exit() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let grid = cave_gen.generate();
        let rooms = cave_gen.detect_rooms(&grid);
        let entries = cave_gen.place_entry_exit(&grid, &rooms);
        if !rooms.is_empty() {
            assert!(entries.len() >= 1);
        }
    }

    #[test]
    fn test_generate_layers() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let layers = cave_gen.generate_layers(3);
        assert_eq!(layers.len(), 3);
        for layer in &layers {
            assert_eq!(layer.width, cave_gen.config().width);
        }
    }

    #[test]
    fn test_layers_differ() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let layers = cave_gen.generate_layers(2);
        // Layers should differ (different seeds)
        assert_ne!(layers[0].cells, layers[1].cells);
    }

    #[test]
    fn test_random_walk_3d() {
        let volume = CaveGenerator::random_walk_3d(16, 16, 16, 42, 200, 0.1);
        assert_eq!(volume.len(), 16);
        let mut air = 0;
        for z in &volume {
            for y in z {
                for c in y {
                    if *c == CaveCell::Air { air += 1; }
                }
            }
        }
        assert!(air > 0);
    }

    #[test]
    fn test_random_walk_deterministic() {
        let v1 = CaveGenerator::random_walk_3d(8, 8, 8, 42, 100, 0.1);
        let v2 = CaveGenerator::random_walk_3d(8, 8, 8, 42, 100, 0.1);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_cave_config_default() {
        let cfg = CaveConfig::default();
        assert_eq!(cfg.width, 64);
        assert_eq!(cfg.height, 64);
        assert_eq!(cfg.automata_iterations, 5);
    }

    #[test]
    fn test_debug_format() {
        let cave_gen = CaveGenerator::new(CaveConfig::default());
        let s = format!("{:?}", cave_gen);
        assert!(s.contains("CaveGenerator"));
    }

    #[test]
    fn test_cave_room_partial_eq() {
        let r1 = CaveRoom { id: 0, cells: vec![(1, 2)], center: (1, 2) };
        let r2 = CaveRoom { id: 0, cells: vec![(1, 2)], center: (1, 2) };
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_placed_feature_eq() {
        let f1 = PlacedFeature { feature: CaveFeature::Stalactite, x: 3, y: 4 };
        let f2 = PlacedFeature { feature: CaveFeature::Stalactite, x: 3, y: 4 };
        assert_eq!(f1, f2);
    }
}
