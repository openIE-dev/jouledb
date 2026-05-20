//! Tile-based map engine: multi-layer tilemaps, chunk storage, coordinate
//! systems (rectangular and isometric), tile properties, and animation.
//!
//! Designed for 2D games with large scrolling worlds. Maps are divided into
//! fixed-size chunks so only visible regions need to reside in memory.

use std::collections::HashMap;

// ── Coordinate Systems ─────────────────────────────────────────

/// Which projection the tilemap uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordSystem {
    /// Standard rectangular grid (x right, y down).
    Rectangular,
    /// Diamond isometric (x right-down, y left-down).
    Isometric,
}

/// A floating-point world position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldPos {
    pub x: f64,
    pub y: f64,
}

/// An integer tile coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub col: i64,
    pub row: i64,
}

/// Convert world position → tile coordinate.
pub fn world_to_tile(pos: WorldPos, tile_w: f64, tile_h: f64, system: CoordSystem) -> TileCoord {
    match system {
        CoordSystem::Rectangular => TileCoord {
            col: (pos.x / tile_w).floor() as i64,
            row: (pos.y / tile_h).floor() as i64,
        },
        CoordSystem::Isometric => {
            // Inverse of iso→world: rotate 45° then scale
            let half_w = tile_w / 2.0;
            let half_h = tile_h / 2.0;
            let col = ((pos.x / half_w + pos.y / half_h) / 2.0).floor() as i64;
            let row = ((pos.y / half_h - pos.x / half_w) / 2.0).floor() as i64;
            TileCoord { col, row }
        }
    }
}

/// Convert tile coordinate → world position (returns the tile's top-left /
/// center depending on projection).
pub fn tile_to_world(coord: TileCoord, tile_w: f64, tile_h: f64, system: CoordSystem) -> WorldPos {
    match system {
        CoordSystem::Rectangular => WorldPos {
            x: coord.col as f64 * tile_w,
            y: coord.row as f64 * tile_h,
        },
        CoordSystem::Isometric => {
            let half_w = tile_w / 2.0;
            let half_h = tile_h / 2.0;
            WorldPos {
                x: (coord.col as f64 - coord.row as f64) * half_w,
                y: (coord.col as f64 + coord.row as f64) * half_h,
            }
        }
    }
}

// ── Tile Properties ────────────────────────────────────────────

/// Bit-flag properties for a tile ID in the tileset.
#[derive(Debug, Clone, PartialEq)]
pub struct TileProps {
    pub solid: bool,
    pub animated: bool,
    pub damage: f64,
    pub custom: HashMap<String, String>,
}

impl Default for TileProps {
    fn default() -> Self {
        Self { solid: false, animated: false, damage: 0.0, custom: HashMap::new() }
    }
}

// ── Tile Animation ─────────────────────────────────────────────

/// A single animation frame in a tile animation sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct TileAnimFrame {
    pub tile_id: u32,
    pub duration_ms: u64,
}

/// Animation data for an animated tile.
#[derive(Debug, Clone, PartialEq)]
pub struct TileAnimation {
    pub frames: Vec<TileAnimFrame>,
}

impl TileAnimation {
    /// Given elapsed ms, return the current display tile ID.
    pub fn current_tile(&self, elapsed_ms: u64) -> u32 {
        if self.frames.is_empty() {
            return 0;
        }
        let total: u64 = self.frames.iter().map(|f| f.duration_ms).sum();
        if total == 0 {
            return self.frames[0].tile_id;
        }
        let t = elapsed_ms % total;
        let mut acc = 0u64;
        for frame in &self.frames {
            acc += frame.duration_ms;
            if t < acc {
                return frame.tile_id;
            }
        }
        self.frames.last().unwrap().tile_id
    }
}

// ── Tileset ────────────────────────────────────────────────────

/// A tileset maps tile IDs to properties and animations.
#[derive(Debug, Clone)]
pub struct Tileset {
    pub name: String,
    pub tile_width: u32,
    pub tile_height: u32,
    pub properties: HashMap<u32, TileProps>,
    pub animations: HashMap<u32, TileAnimation>,
}

impl Tileset {
    pub fn new(name: &str, tile_w: u32, tile_h: u32) -> Self {
        Self {
            name: name.to_string(),
            tile_width: tile_w,
            tile_height: tile_h,
            properties: HashMap::new(),
            animations: HashMap::new(),
        }
    }

    pub fn set_props(&mut self, tile_id: u32, props: TileProps) {
        self.properties.insert(tile_id, props);
    }

    pub fn set_animation(&mut self, tile_id: u32, anim: TileAnimation) {
        self.animations.insert(tile_id, anim);
    }

    pub fn get_props(&self, tile_id: u32) -> Option<&TileProps> {
        self.properties.get(&tile_id)
    }

    pub fn is_solid(&self, tile_id: u32) -> bool {
        self.properties.get(&tile_id).map_or(false, |p| p.solid)
    }
}

// ── Chunk Storage ──────────────────────────────────────────────

/// Chunk coordinate — a chunk is a fixed-size block of tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub cx: i64,
    pub cy: i64,
}

/// A single chunk storing `size × size` tile IDs.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub size: u32,
    pub tiles: Vec<u32>,
}

impl Chunk {
    pub fn new(size: u32) -> Self {
        Self {
            size,
            tiles: vec![0; (size * size) as usize],
        }
    }

    fn index(&self, local_col: u32, local_row: u32) -> usize {
        (local_row * self.size + local_col) as usize
    }

    pub fn get(&self, local_col: u32, local_row: u32) -> u32 {
        if local_col >= self.size || local_row >= self.size {
            return 0;
        }
        self.tiles[self.index(local_col, local_row)]
    }

    pub fn set(&mut self, local_col: u32, local_row: u32, tile_id: u32) {
        if local_col < self.size && local_row < self.size {
            let idx = self.index(local_col, local_row);
            self.tiles[idx] = tile_id;
        }
    }
}

// ── Map Layer ──────────────────────────────────────────────────

/// A layer kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Ground,
    Objects,
    Overhead,
    Custom(u32),
}

/// One layer of the tilemap, stored as chunks.
#[derive(Debug, Clone)]
pub struct TileLayer {
    pub kind: LayerKind,
    pub visible: bool,
    pub opacity: f64,
    pub chunk_size: u32,
    chunks: HashMap<ChunkCoord, Chunk>,
}

impl TileLayer {
    pub fn new(kind: LayerKind, chunk_size: u32) -> Self {
        Self {
            kind,
            visible: true,
            opacity: 1.0,
            chunk_size,
            chunks: HashMap::new(),
        }
    }

    fn chunk_and_local(&self, col: i64, row: i64) -> (ChunkCoord, u32, u32) {
        let cs = self.chunk_size as i64;
        let cx = col.div_euclid(cs);
        let cy = row.div_euclid(cs);
        let lx = col.rem_euclid(cs) as u32;
        let ly = row.rem_euclid(cs) as u32;
        (ChunkCoord { cx, cy }, lx, ly)
    }

    pub fn get_tile(&self, col: i64, row: i64) -> u32 {
        let (cc, lx, ly) = self.chunk_and_local(col, row);
        self.chunks.get(&cc).map_or(0, |c| c.get(lx, ly))
    }

    pub fn set_tile(&mut self, col: i64, row: i64, tile_id: u32) {
        let (cc, lx, ly) = self.chunk_and_local(col, row);
        self.chunks.entry(cc).or_insert_with(|| Chunk::new(self.chunk_size)).set(lx, ly, tile_id);
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn has_chunk(&self, cx: i64, cy: i64) -> bool {
        self.chunks.contains_key(&ChunkCoord { cx, cy })
    }
}

// ── Tilemap ────────────────────────────────────────────────────

/// Top-level tilemap holding layers and a tileset reference.
#[derive(Debug, Clone)]
pub struct Tilemap {
    pub coord_system: CoordSystem,
    pub tileset: Tileset,
    layers: Vec<TileLayer>,
}

impl Tilemap {
    pub fn new(coord_system: CoordSystem, tileset: Tileset) -> Self {
        Self { coord_system, tileset, layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: TileLayer) -> usize {
        let idx = self.layers.len();
        self.layers.push(layer);
        idx
    }

    pub fn layer(&self, index: usize) -> Option<&TileLayer> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut TileLayer> {
        self.layers.get_mut(index)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Query tile at a world position across all visible layers (front to back).
    /// Returns vec of `(layer_index, tile_id)` for non-zero tiles.
    pub fn query_world(&self, pos: WorldPos) -> Vec<(usize, u32)> {
        let tw = self.tileset.tile_width as f64;
        let th = self.tileset.tile_height as f64;
        let coord = world_to_tile(pos, tw, th, self.coord_system);
        let mut result = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if !layer.visible {
                continue;
            }
            let tid = layer.get_tile(coord.col, coord.row);
            if tid != 0 {
                result.push((i, tid));
            }
        }
        result
    }

    /// Check if a tile coordinate is solid on any visible layer.
    pub fn is_solid(&self, col: i64, row: i64) -> bool {
        for layer in &self.layers {
            if !layer.visible {
                continue;
            }
            let tid = layer.get_tile(col, row);
            if tid != 0 && self.tileset.is_solid(tid) {
                return true;
            }
        }
        false
    }

    /// Get the animated tile ID for a given tile at elapsed time.
    pub fn animated_tile(&self, tile_id: u32, elapsed_ms: u64) -> u32 {
        self.tileset
            .animations
            .get(&tile_id)
            .map_or(tile_id, |anim| anim.current_tile(elapsed_ms))
    }

    /// Fill a rectangular region on a layer.
    pub fn fill_rect(
        &mut self,
        layer_idx: usize,
        col_start: i64,
        row_start: i64,
        width: u32,
        height: u32,
        tile_id: u32,
    ) {
        if let Some(layer) = self.layers.get_mut(layer_idx) {
            for r in 0..height as i64 {
                for c in 0..width as i64 {
                    layer.set_tile(col_start + c, row_start + r, tile_id);
                }
            }
        }
    }

    /// Collect all non-zero tiles in a rectangular region on a layer.
    pub fn tiles_in_rect(
        &self,
        layer_idx: usize,
        col_start: i64,
        row_start: i64,
        width: u32,
        height: u32,
    ) -> Vec<(TileCoord, u32)> {
        let mut out = Vec::new();
        if let Some(layer) = self.layers.get(layer_idx) {
            for r in 0..height as i64 {
                for c in 0..width as i64 {
                    let col = col_start + c;
                    let row = row_start + r;
                    let tid = layer.get_tile(col, row);
                    if tid != 0 {
                        out.push((TileCoord { col, row }, tid));
                    }
                }
            }
        }
        out
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn make_tileset() -> Tileset {
        let mut ts = Tileset::new("test", 16, 16);
        ts.set_props(1, TileProps { solid: true, ..Default::default() });
        ts.set_props(2, TileProps { solid: false, damage: 5.0, ..Default::default() });
        ts
    }

    fn make_map() -> Tilemap {
        let ts = make_tileset();
        let mut map = Tilemap::new(CoordSystem::Rectangular, ts);
        let ground = TileLayer::new(LayerKind::Ground, 16);
        let objects = TileLayer::new(LayerKind::Objects, 16);
        map.add_layer(ground);
        map.add_layer(objects);
        map
    }

    #[test]
    fn rect_world_to_tile_basic() {
        let t = world_to_tile(WorldPos { x: 24.0, y: 35.0 }, 16.0, 16.0, CoordSystem::Rectangular);
        assert_eq!(t.col, 1);
        assert_eq!(t.row, 2);
    }

    #[test]
    fn rect_tile_to_world_basic() {
        let w = tile_to_world(TileCoord { col: 3, row: 2 }, 16.0, 16.0, CoordSystem::Rectangular);
        assert!((w.x - 48.0).abs() < EPS);
        assert!((w.y - 32.0).abs() < EPS);
    }

    #[test]
    fn rect_round_trip() {
        let coord = TileCoord { col: 5, row: 7 };
        let w = tile_to_world(coord, 16.0, 16.0, CoordSystem::Rectangular);
        let back = world_to_tile(w, 16.0, 16.0, CoordSystem::Rectangular);
        assert_eq!(back, coord);
    }

    #[test]
    fn iso_round_trip() {
        let coord = TileCoord { col: 3, row: 4 };
        let w = tile_to_world(coord, 64.0, 32.0, CoordSystem::Isometric);
        let back = world_to_tile(w, 64.0, 32.0, CoordSystem::Isometric);
        assert_eq!(back, coord);
    }

    #[test]
    fn iso_world_to_tile_origin() {
        let t = world_to_tile(WorldPos { x: 0.0, y: 0.0 }, 64.0, 32.0, CoordSystem::Isometric);
        assert_eq!(t.col, 0);
        assert_eq!(t.row, 0);
    }

    #[test]
    fn negative_coords_div_euclid() {
        let t = world_to_tile(WorldPos { x: -1.0, y: -1.0 }, 16.0, 16.0, CoordSystem::Rectangular);
        assert_eq!(t.col, -1);
        assert_eq!(t.row, -1);
    }

    #[test]
    fn chunk_local_negative() {
        let layer = TileLayer::new(LayerKind::Ground, 16);
        let (cc, lx, ly) = layer.chunk_and_local(-1, -1);
        assert_eq!(cc, ChunkCoord { cx: -1, cy: -1 });
        assert_eq!(lx, 15);
        assert_eq!(ly, 15);
    }

    #[test]
    fn layer_set_get_tile() {
        let mut layer = TileLayer::new(LayerKind::Ground, 16);
        layer.set_tile(5, 7, 42);
        assert_eq!(layer.get_tile(5, 7), 42);
        assert_eq!(layer.get_tile(0, 0), 0);
    }

    #[test]
    fn layer_negative_tile() {
        let mut layer = TileLayer::new(LayerKind::Ground, 8);
        layer.set_tile(-3, -5, 99);
        assert_eq!(layer.get_tile(-3, -5), 99);
    }

    #[test]
    fn layer_chunk_count() {
        let mut layer = TileLayer::new(LayerKind::Ground, 8);
        layer.set_tile(0, 0, 1);
        layer.set_tile(100, 100, 2);
        assert_eq!(layer.chunk_count(), 2);
    }

    #[test]
    fn tileset_solid_check() {
        let ts = make_tileset();
        assert!(ts.is_solid(1));
        assert!(!ts.is_solid(2));
        assert!(!ts.is_solid(999));
    }

    #[test]
    fn tile_animation_cycling() {
        let anim = TileAnimation {
            frames: vec![
                TileAnimFrame { tile_id: 10, duration_ms: 100 },
                TileAnimFrame { tile_id: 11, duration_ms: 100 },
                TileAnimFrame { tile_id: 12, duration_ms: 100 },
            ],
        };
        assert_eq!(anim.current_tile(0), 10);
        assert_eq!(anim.current_tile(99), 10);
        assert_eq!(anim.current_tile(100), 11);
        assert_eq!(anim.current_tile(250), 12);
        // Wraps at 300
        assert_eq!(anim.current_tile(300), 10);
        assert_eq!(anim.current_tile(400), 11);
    }

    #[test]
    fn tile_animation_empty() {
        let anim = TileAnimation { frames: vec![] };
        assert_eq!(anim.current_tile(500), 0);
    }

    #[test]
    fn map_add_layers() {
        let map = make_map();
        assert_eq!(map.layer_count(), 2);
    }

    #[test]
    fn map_query_world_pos() {
        let mut map = make_map();
        map.layer_mut(0).unwrap().set_tile(1, 2, 1);
        map.layer_mut(1).unwrap().set_tile(1, 2, 2);
        let result = map.query_world(WorldPos { x: 24.0, y: 35.0 });
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0, 1));
        assert_eq!(result[1], (1, 2));
    }

    #[test]
    fn map_is_solid() {
        let mut map = make_map();
        map.layer_mut(0).unwrap().set_tile(3, 3, 1);
        assert!(map.is_solid(3, 3));
        assert!(!map.is_solid(0, 0));
    }

    #[test]
    fn map_hidden_layer_not_queried() {
        let mut map = make_map();
        map.layer_mut(0).unwrap().set_tile(0, 0, 1);
        map.layer_mut(0).unwrap().visible = false;
        let result = map.query_world(WorldPos { x: 0.0, y: 0.0 });
        assert!(result.is_empty());
        assert!(!map.is_solid(0, 0));
    }

    #[test]
    fn map_fill_rect() {
        let mut map = make_map();
        map.fill_rect(0, 0, 0, 4, 3, 5);
        for r in 0..3i64 {
            for c in 0..4i64 {
                assert_eq!(map.layer(0).unwrap().get_tile(c, r), 5);
            }
        }
    }

    #[test]
    fn map_tiles_in_rect() {
        let mut map = make_map();
        map.layer_mut(0).unwrap().set_tile(2, 2, 7);
        map.layer_mut(0).unwrap().set_tile(3, 3, 8);
        let tiles = map.tiles_in_rect(0, 0, 0, 5, 5);
        assert_eq!(tiles.len(), 2);
    }

    #[test]
    fn map_animated_tile_lookup() {
        let mut map = make_map();
        map.tileset.set_animation(
            50,
            TileAnimation {
                frames: vec![
                    TileAnimFrame { tile_id: 50, duration_ms: 200 },
                    TileAnimFrame { tile_id: 51, duration_ms: 200 },
                ],
            },
        );
        assert_eq!(map.animated_tile(50, 0), 50);
        assert_eq!(map.animated_tile(50, 250), 51);
        // Non-animated tile returns itself
        assert_eq!(map.animated_tile(1, 0), 1);
    }

    #[test]
    fn tile_props_custom_fields() {
        let mut props = TileProps::default();
        props.custom.insert("material".to_string(), "stone".to_string());
        assert_eq!(props.custom.get("material").unwrap(), "stone");
    }

    #[test]
    fn chunk_boundary_tiles() {
        let mut layer = TileLayer::new(LayerKind::Ground, 4);
        // Tile at boundary between chunks
        layer.set_tile(3, 3, 10);
        layer.set_tile(4, 4, 20);
        assert_eq!(layer.get_tile(3, 3), 10);
        assert_eq!(layer.get_tile(4, 4), 20);
        assert!(layer.has_chunk(0, 0));
        assert!(layer.has_chunk(1, 1));
    }

    #[test]
    fn tileset_damage_property() {
        let ts = make_tileset();
        let props = ts.get_props(2).unwrap();
        assert!((props.damage - 5.0).abs() < EPS);
    }
}
