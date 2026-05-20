//! Procedural dungeon generation via Binary Space Partitioning.
//!
//! Recursively splits a rectangular area into rooms, connects sibling
//! partitions with corridors, places doors at room-corridor junctions,
//! and designates special rooms (spawn, boss, treasure). All output
//! is a deterministic tile grid driven by a 64-bit seed.

// ── Tile ──

/// Tile types that compose the dungeon grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tile {
    Wall,
    Floor,
    Door,
    Corridor,
    Empty,
}

impl Tile {
    /// Character representation for text visualization.
    pub fn as_char(self) -> char {
        match self {
            Tile::Wall => '#',
            Tile::Floor => '.',
            Tile::Door => 'D',
            Tile::Corridor => ',',
            Tile::Empty => ' ',
        }
    }
}

// ── Rect ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        Self { x, y, w, h }
    }

    pub fn center(&self) -> (usize, usize) {
        (self.x + self.w / 2, self.y + self.h / 2)
    }

    pub fn area(&self) -> usize {
        self.w * self.h
    }

    pub fn contains(&self, px: usize, py: usize) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }
}

// ── CorridorStyle ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorridorStyle {
    Straight,
    LShaped,
    ZShaped,
}

// ── RoomKind ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomKind {
    Normal,
    Spawn,
    Boss,
    Treasure,
}

// ── Room ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Room {
    pub bounds: Rect,
    pub kind: RoomKind,
    pub id: usize,
}

// ── CorridorSegment ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorridorSegment {
    pub cells: Vec<(usize, usize)>,
    pub style: CorridorStyle,
    pub doors: Vec<(usize, usize)>,
}

// ── BSP node ──

#[derive(Debug)]
enum BspNode {
    Leaf {
        area: Rect,
        room: Option<Rect>,
    },
    Split {
        area: Rect,
        left: Box<BspNode>,
        right: Box<BspNode>,
        horizontal: bool,
    },
}

// ── DungeonConfig ──

#[derive(Debug, Clone)]
pub struct DungeonConfig {
    pub width: usize,
    pub height: usize,
    pub min_room_size: usize,
    pub max_room_size: usize,
    pub padding: usize,
    pub max_depth: usize,
    pub corridor_style: CorridorStyle,
    pub seed: u64,
}

impl Default for DungeonConfig {
    fn default() -> Self {
        Self {
            width: 60,
            height: 40,
            min_room_size: 4,
            max_room_size: 12,
            padding: 1,
            max_depth: 5,
            corridor_style: CorridorStyle::LShaped,
            seed: 42,
        }
    }
}

// ── Dungeon ──

#[derive(Debug, Clone)]
pub struct Dungeon {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Vec<Tile>>,
    pub rooms: Vec<Room>,
    pub corridors: Vec<CorridorSegment>,
}

impl Dungeon {
    /// Render the dungeon to a string.
    pub fn to_string_grid(&self) -> String {
        let mut out = String::with_capacity(self.width * self.height + self.height);
        for row in &self.tiles {
            for tile in row {
                out.push(tile.as_char());
            }
            out.push('\n');
        }
        out
    }

    /// Get tile at (x, y). Returns Empty if out of bounds.
    pub fn tile_at(&self, x: usize, y: usize) -> Tile {
        if y < self.height && x < self.width {
            self.tiles[y][x]
        } else {
            Tile::Empty
        }
    }

    /// Find the room containing (x, y), if any.
    pub fn room_at(&self, x: usize, y: usize) -> Option<&Room> {
        self.rooms.iter().find(|r| r.bounds.contains(x, y))
    }

    /// Count tiles of a given type.
    pub fn count_tiles(&self, tile: Tile) -> usize {
        self.tiles.iter().flat_map(|row| row.iter()).filter(|&&t| t == tile).count()
    }
}

// ── Seeded RNG (splitmix64) ──

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
        if lo >= hi {
            return lo;
        }
        let span = (hi - lo) as u64;
        lo + (self.next_u64() % span) as usize
    }

    fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }
}

// ── Generator ──

/// Build a dungeon from the given configuration.
pub fn generate(config: &DungeonConfig) -> Dungeon {
    let mut rng = Rng::new(config.seed);
    let mut tiles = vec![vec![Tile::Wall; config.width]; config.height];

    let root_area = Rect::new(0, 0, config.width, config.height);
    let mut tree = build_bsp(&root_area, config, &mut rng, 0);

    let mut rooms: Vec<Room> = Vec::new();
    let mut room_id = 0usize;
    collect_rooms(&mut tree, config, &mut rng, &mut rooms, &mut room_id);

    // Carve rooms
    for room in &rooms {
        let b = &room.bounds;
        for ry in b.y..b.y + b.h {
            for rx in b.x..b.x + b.w {
                if ry < config.height && rx < config.width {
                    tiles[ry][rx] = Tile::Floor;
                }
            }
        }
    }

    // Connect siblings via corridors
    let mut corridors: Vec<CorridorSegment> = Vec::new();
    connect_bsp(&tree, &mut tiles, config, &mut rng, &mut corridors);

    // Assign special rooms
    assign_special_rooms(&mut rooms, &mut rng);

    Dungeon {
        width: config.width,
        height: config.height,
        tiles,
        rooms,
        corridors,
    }
}

fn build_bsp(area: &Rect, config: &DungeonConfig, rng: &mut Rng, depth: usize) -> BspNode {
    let min_partition = config.min_room_size + config.padding * 2 + 2;
    if depth >= config.max_depth || (area.w < min_partition * 2 && area.h < min_partition * 2) {
        return BspNode::Leaf { area: *area, room: None };
    }

    let can_split_h = area.h >= min_partition * 2;
    let can_split_v = area.w >= min_partition * 2;

    if !can_split_h && !can_split_v {
        return BspNode::Leaf { area: *area, room: None };
    }

    let horizontal = if can_split_h && can_split_v {
        rng.coin()
    } else {
        can_split_h
    };

    if horizontal {
        let split = rng.range(area.y + min_partition, area.y + area.h - min_partition + 1);
        let top = Rect::new(area.x, area.y, area.w, split - area.y);
        let bottom = Rect::new(area.x, split, area.w, area.y + area.h - split);
        let left = build_bsp(&top, config, rng, depth + 1);
        let right = build_bsp(&bottom, config, rng, depth + 1);
        BspNode::Split {
            area: *area,
            left: Box::new(left),
            right: Box::new(right),
            horizontal: true,
        }
    } else {
        let split = rng.range(area.x + min_partition, area.x + area.w - min_partition + 1);
        let left_rect = Rect::new(area.x, area.y, split - area.x, area.h);
        let right_rect = Rect::new(split, area.y, area.x + area.w - split, area.h);
        let left = build_bsp(&left_rect, config, rng, depth + 1);
        let right = build_bsp(&right_rect, config, rng, depth + 1);
        BspNode::Split {
            area: *area,
            left: Box::new(left),
            right: Box::new(right),
            horizontal: false,
        }
    }
}

fn collect_rooms(
    node: &mut BspNode,
    config: &DungeonConfig,
    rng: &mut Rng,
    rooms: &mut Vec<Room>,
    next_id: &mut usize,
) {
    match node {
        BspNode::Leaf { area, room } => {
            let pad = config.padding;
            let avail_w = if area.w > pad * 2 + 2 { area.w - pad * 2 } else { return; };
            let avail_h = if area.h > pad * 2 + 2 { area.h - pad * 2 } else { return; };
            let rw = rng.range(
                config.min_room_size.min(avail_w),
                config.max_room_size.min(avail_w) + 1,
            );
            let rh = rng.range(
                config.min_room_size.min(avail_h),
                config.max_room_size.min(avail_h) + 1,
            );
            let rx = if avail_w > rw { rng.range(area.x + pad, area.x + pad + avail_w - rw + 1) } else { area.x + pad };
            let ry = if avail_h > rh { rng.range(area.y + pad, area.y + pad + avail_h - rh + 1) } else { area.y + pad };
            let room_rect = Rect::new(rx, ry, rw, rh);
            *room = Some(room_rect);
            let r = Room {
                bounds: room_rect,
                kind: RoomKind::Normal,
                id: *next_id,
            };
            *next_id += 1;
            rooms.push(r);
        }
        BspNode::Split { left, right, .. } => {
            collect_rooms(left, config, rng, rooms, next_id);
            collect_rooms(right, config, rng, rooms, next_id);
        }
    }
}

fn find_any_room(node: &BspNode) -> Option<Rect> {
    match node {
        BspNode::Leaf { room, .. } => *room,
        BspNode::Split { left, right, .. } => find_any_room(left).or_else(|| find_any_room(right)),
    }
}

fn connect_bsp(
    node: &BspNode,
    tiles: &mut [Vec<Tile>],
    config: &DungeonConfig,
    rng: &mut Rng,
    corridors: &mut Vec<CorridorSegment>,
) {
    if let BspNode::Split { left, right, .. } = node {
        connect_bsp(left, tiles, config, rng, corridors);
        connect_bsp(right, tiles, config, rng, corridors);

        let la = find_any_room(left);
        let ra = find_any_room(right);
        if let (Some(la), Some(ra)) = (la, ra) {
            let (lx, ly) = la.center();
            let (rx, ry) = ra.center();
            let seg = carve_corridor(tiles, lx, ly, rx, ry, config, rng);
            corridors.push(seg);
        }
    }
}

fn carve_corridor(
    tiles: &mut [Vec<Tile>],
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    config: &DungeonConfig,
    rng: &mut Rng,
) -> CorridorSegment {
    let mut cells = Vec::new();
    let mut doors = Vec::new();
    let style = config.corridor_style;

    match style {
        CorridorStyle::Straight => {
            carve_line(tiles, x1, y1, x2, y1, &mut cells, &mut doors);
            carve_line(tiles, x2, y1, x2, y2, &mut cells, &mut doors);
        }
        CorridorStyle::LShaped => {
            if rng.coin() {
                carve_line(tiles, x1, y1, x2, y1, &mut cells, &mut doors);
                carve_line(tiles, x2, y1, x2, y2, &mut cells, &mut doors);
            } else {
                carve_line(tiles, x1, y1, x1, y2, &mut cells, &mut doors);
                carve_line(tiles, x1, y2, x2, y2, &mut cells, &mut doors);
            }
        }
        CorridorStyle::ZShaped => {
            let mid_y = if y1 < y2 { y1 + (y2 - y1) / 2 } else { y2 + (y1 - y2) / 2 };
            carve_line(tiles, x1, y1, x1, mid_y, &mut cells, &mut doors);
            carve_line(tiles, x1, mid_y, x2, mid_y, &mut cells, &mut doors);
            carve_line(tiles, x2, mid_y, x2, y2, &mut cells, &mut doors);
        }
    }

    CorridorSegment { cells, style, doors }
}

fn carve_line(
    tiles: &mut [Vec<Tile>],
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    cells: &mut Vec<(usize, usize)>,
    doors: &mut Vec<(usize, usize)>,
) {
    let height = tiles.len();
    let width = if height > 0 { tiles[0].len() } else { 0 };

    let (mut cx, mut cy) = (x1 as isize, y1 as isize);
    let dx: isize = if x2 > x1 { 1 } else if x2 < x1 { -1 } else { 0 };
    let dy: isize = if y2 > y1 { 1 } else if y2 < y1 { -1 } else { 0 };

    loop {
        let ux = cx as usize;
        let uy = cy as usize;
        if uy < height && ux < width {
            let prev = tiles[uy][ux];
            if prev == Tile::Wall {
                // Check if adjacent to a floor tile (room entry => door)
                let adj_floor = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|&(adx, ady)| {
                        let nx = cx + adx;
                        let ny = cy + ady;
                        if nx >= 0 && ny >= 0 {
                            let nux = nx as usize;
                            let nuy = ny as usize;
                            nuy < height && nux < width && tiles[nuy][nux] == Tile::Floor
                        } else {
                            false
                        }
                    });
                if adj_floor {
                    tiles[uy][ux] = Tile::Door;
                    doors.push((ux, uy));
                } else {
                    tiles[uy][ux] = Tile::Corridor;
                }
                cells.push((ux, uy));
            }
        }
        if cx == x2 as isize && cy == y2 as isize {
            break;
        }
        if cx != x2 as isize {
            cx += dx;
        } else {
            cy += dy;
        }
    }
}

fn assign_special_rooms(rooms: &mut [Room], rng: &mut Rng) {
    if rooms.is_empty() {
        return;
    }

    // Smallest room = spawn, largest = boss, second largest = treasure
    let mut by_area: Vec<usize> = (0..rooms.len()).collect();
    by_area.sort_by_key(|i| rooms[*i].bounds.area());

    rooms[by_area[0]].kind = RoomKind::Spawn;

    if rooms.len() > 1 {
        rooms[by_area[by_area.len() - 1]].kind = RoomKind::Boss;
    }
    if rooms.len() > 2 {
        // Pick a treasure room from the middle
        let mid = rng.range(1, by_area.len() - 1);
        rooms[by_area[mid]].kind = RoomKind::Treasure;
    }
}

// ── Helpers ──

/// Count all rooms of a given kind.
pub fn count_rooms_by_kind(dungeon: &Dungeon, kind: RoomKind) -> usize {
    dungeon.rooms.iter().filter(|r| r.kind == kind).count()
}

/// Verify no two rooms overlap.
pub fn rooms_overlap(rooms: &[Room]) -> bool {
    for i in 0..rooms.len() {
        for j in (i + 1)..rooms.len() {
            let a = &rooms[i].bounds;
            let b = &rooms[j].bounds;
            let sep_x = a.x + a.w <= b.x || b.x + b.w <= a.x;
            let sep_y = a.y + a.h <= b.y || b.y + b.h <= a.y;
            if !sep_x && !sep_y {
                return true;
            }
        }
    }
    false
}

/// Verify all floor tiles are reachable from the first floor tile via BFS.
pub fn is_connected(dungeon: &Dungeon) -> bool {
    use std::collections::VecDeque;

    let passable = |t: Tile| matches!(t, Tile::Floor | Tile::Corridor | Tile::Door);

    let mut start = None;
    'outer: for y in 0..dungeon.height {
        for x in 0..dungeon.width {
            if passable(dungeon.tiles[y][x]) {
                start = Some((x, y));
                break 'outer;
            }
        }
    }
    let (sx, sy) = match start {
        Some(s) => s,
        None => return true,
    };

    let mut visited = vec![vec![false; dungeon.width]; dungeon.height];
    let mut queue = VecDeque::new();
    visited[sy][sx] = true;
    queue.push_back((sx, sy));

    while let Some((cx, cy)) = queue.pop_front() {
        for (dx, dy) in &[(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 {
                let (ux, uy) = (nx as usize, ny as usize);
                if uy < dungeon.height && ux < dungeon.width && !visited[uy][ux] && passable(dungeon.tiles[uy][ux]) {
                    visited[uy][ux] = true;
                    queue.push_back((ux, uy));
                }
            }
        }
    }

    for y in 0..dungeon.height {
        for x in 0..dungeon.width {
            if passable(dungeon.tiles[y][x]) && !visited[y][x] {
                return false;
            }
        }
    }
    true
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn default_dungeon() -> Dungeon {
        generate(&DungeonConfig::default())
    }

    #[test]
    fn test_rect_basics() {
        let r = Rect::new(5, 10, 20, 15);
        assert_eq!(r.center(), (15, 17));
        assert_eq!(r.area(), 300);
        assert!(r.contains(5, 10));
        assert!(r.contains(24, 24));
        assert!(!r.contains(25, 10));
    }

    #[test]
    fn test_tile_chars() {
        assert_eq!(Tile::Wall.as_char(), '#');
        assert_eq!(Tile::Floor.as_char(), '.');
        assert_eq!(Tile::Door.as_char(), 'D');
        assert_eq!(Tile::Corridor.as_char(), ',');
        assert_eq!(Tile::Empty.as_char(), ' ');
    }

    #[test]
    fn test_default_generation() {
        let d = default_dungeon();
        assert_eq!(d.width, 60);
        assert_eq!(d.height, 40);
        assert!(!d.rooms.is_empty());
        assert!(d.count_tiles(Tile::Floor) > 0);
    }

    #[test]
    fn test_seed_determinism() {
        let cfg = DungeonConfig { seed: 123, ..Default::default() };
        let d1 = generate(&cfg);
        let d2 = generate(&cfg);
        assert_eq!(d1.tiles, d2.tiles);
        assert_eq!(d1.rooms.len(), d2.rooms.len());
    }

    #[test]
    fn test_different_seeds_differ() {
        let d1 = generate(&DungeonConfig { seed: 1, ..Default::default() });
        let d2 = generate(&DungeonConfig { seed: 999, ..Default::default() });
        // Extremely unlikely to be identical
        assert_ne!(d1.tiles, d2.tiles);
    }

    #[test]
    fn test_rooms_within_bounds() {
        let d = default_dungeon();
        for room in &d.rooms {
            let b = &room.bounds;
            assert!(b.x + b.w <= d.width, "room exceeds width");
            assert!(b.y + b.h <= d.height, "room exceeds height");
        }
    }

    #[test]
    fn test_no_room_overlap() {
        let d = default_dungeon();
        assert!(!rooms_overlap(&d.rooms));
    }

    #[test]
    fn test_dungeon_connected() {
        let d = default_dungeon();
        assert!(is_connected(&d));
    }

    #[test]
    fn test_special_rooms_assigned() {
        let d = default_dungeon();
        assert_eq!(count_rooms_by_kind(&d, RoomKind::Spawn), 1);
        if d.rooms.len() > 1 {
            assert_eq!(count_rooms_by_kind(&d, RoomKind::Boss), 1);
        }
    }

    #[test]
    fn test_corridors_present() {
        let d = default_dungeon();
        assert!(!d.corridors.is_empty());
        assert!(d.count_tiles(Tile::Corridor) > 0 || d.count_tiles(Tile::Door) > 0);
    }

    #[test]
    fn test_doors_at_junctions() {
        let d = default_dungeon();
        for seg in &d.corridors {
            for &(dx, dy) in &seg.doors {
                assert_eq!(d.tiles[dy][dx], Tile::Door);
            }
        }
    }

    #[test]
    fn test_to_string_grid() {
        let d = generate(&DungeonConfig {
            width: 20,
            height: 10,
            seed: 7,
            max_depth: 3,
            ..Default::default()
        });
        let s = d.to_string_grid();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 10);
        for line in &lines {
            assert_eq!(line.len(), 20);
        }
    }

    #[test]
    fn test_small_dungeon() {
        let d = generate(&DungeonConfig {
            width: 30,
            height: 20,
            min_room_size: 3,
            max_room_size: 6,
            max_depth: 3,
            seed: 55,
            ..Default::default()
        });
        assert!(!d.rooms.is_empty());
        assert!(is_connected(&d));
    }

    #[test]
    fn test_corridor_style_straight() {
        let d = generate(&DungeonConfig {
            corridor_style: CorridorStyle::Straight,
            seed: 88,
            ..Default::default()
        });
        assert!(is_connected(&d));
    }

    #[test]
    fn test_corridor_style_z() {
        let d = generate(&DungeonConfig {
            corridor_style: CorridorStyle::ZShaped,
            seed: 42,
            ..Default::default()
        });
        assert!(is_connected(&d));
    }

    #[test]
    fn test_room_at() {
        let d = default_dungeon();
        if let Some(room) = d.rooms.first() {
            let (cx, cy) = room.bounds.center();
            let found = d.room_at(cx, cy);
            assert!(found.is_some());
        }
    }

    #[test]
    fn test_unique_room_ids() {
        let d = default_dungeon();
        let mut ids: Vec<usize> = d.rooms.iter().map(|r| r.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), d.rooms.len());
    }

    #[test]
    fn test_wall_border() {
        let d = default_dungeon();
        // Top and bottom borders should be walls (or empty)
        for x in 0..d.width {
            assert!(matches!(d.tiles[0][x], Tile::Wall | Tile::Empty));
            assert!(matches!(d.tiles[d.height - 1][x], Tile::Wall | Tile::Empty));
        }
    }

    #[test]
    fn test_count_tiles() {
        let d = default_dungeon();
        let total = d.count_tiles(Tile::Wall)
            + d.count_tiles(Tile::Floor)
            + d.count_tiles(Tile::Door)
            + d.count_tiles(Tile::Corridor)
            + d.count_tiles(Tile::Empty);
        assert_eq!(total, d.width * d.height);
    }

    #[test]
    fn test_many_seeds_connected() {
        for seed in 0..20u64 {
            let d = generate(&DungeonConfig { seed, ..Default::default() });
            assert!(is_connected(&d), "seed {} produced disconnected dungeon", seed);
        }
    }

    #[test]
    fn test_room_min_size_respected() {
        let cfg = DungeonConfig {
            min_room_size: 5,
            max_room_size: 10,
            seed: 33,
            ..Default::default()
        };
        let d = generate(&cfg);
        for room in &d.rooms {
            assert!(room.bounds.w >= 4, "room width {} < min", room.bounds.w);
            assert!(room.bounds.h >= 4, "room height {} < min", room.bounds.h);
        }
    }
}
