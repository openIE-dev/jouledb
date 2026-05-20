//! 2D tilemap — tile sets, tile layers, coordinate conversion, and isometric support.
//!
//! Pure Rust replacement for Tiled map editor runtime, Phaser tilemap,
//! and similar tile-based game map systems. Fully headless.

use std::collections::HashMap;

// ── TileSet ──────────────────────────────────────────────────

/// A tile set defines the grid of tiles in an image.
#[derive(Debug, Clone, PartialEq)]
pub struct TileSet {
    pub name: String,
    pub image_id: u64,
    pub tile_width: u32,
    pub tile_height: u32,
    pub columns: u32,
    pub rows: u32,
    pub first_gid: u32,
    pub margin: u32,
    pub spacing: u32,
    pub properties: HashMap<u32, TileProperties>,
}

impl TileSet {
    pub fn new(
        name: impl Into<String>,
        image_id: u64,
        tile_width: u32,
        tile_height: u32,
        columns: u32,
        rows: u32,
    ) -> Self {
        Self {
            name: name.into(),
            image_id,
            tile_width,
            tile_height,
            columns,
            rows,
            first_gid: 1,
            margin: 0,
            spacing: 0,
            properties: HashMap::new(),
        }
    }

    pub fn with_first_gid(mut self, gid: u32) -> Self {
        self.first_gid = gid;
        self
    }

    pub fn with_margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    pub fn with_spacing(mut self, spacing: u32) -> Self {
        self.spacing = spacing;
        self
    }

    /// Total number of tiles in this set.
    pub fn tile_count(&self) -> u32 {
        self.columns * self.rows
    }

    /// Whether a global tile ID belongs to this tile set.
    pub fn contains_gid(&self, gid: u32) -> bool {
        gid >= self.first_gid && gid < self.first_gid + self.tile_count()
    }

    /// Get the source rectangle (in pixels) for a local tile index.
    pub fn tile_rect(&self, local_id: u32) -> Option<TileRect> {
        if local_id >= self.tile_count() {
            return None;
        }
        let col = local_id % self.columns;
        let row = local_id / self.columns;
        let x = self.margin + col * (self.tile_width + self.spacing);
        let y = self.margin + row * (self.tile_height + self.spacing);
        Some(TileRect { x, y, width: self.tile_width, height: self.tile_height })
    }

    /// Get the source rectangle for a global tile ID.
    pub fn rect_for_gid(&self, gid: u32) -> Option<TileRect> {
        if !self.contains_gid(gid) {
            return None;
        }
        self.tile_rect(gid - self.first_gid)
    }

    /// Set tile properties for a local tile ID.
    pub fn set_properties(&mut self, local_id: u32, props: TileProperties) {
        self.properties.insert(local_id, props);
    }

    /// Get tile properties for a local tile ID.
    pub fn get_properties(&self, local_id: u32) -> Option<&TileProperties> {
        self.properties.get(&local_id)
    }
}

/// Source rectangle in a tile set image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Properties that can be assigned to tiles.
#[derive(Debug, Clone, PartialEq)]
pub struct TileProperties {
    pub solid: bool,
    pub animated: bool,
    pub animation_frames: Vec<AnimationFrame>,
    pub custom: HashMap<String, String>,
}

impl TileProperties {
    pub fn new() -> Self {
        Self {
            solid: false,
            animated: false,
            animation_frames: Vec::new(),
            custom: HashMap::new(),
        }
    }

    pub fn solid() -> Self {
        Self { solid: true, ..Self::new() }
    }

    pub fn animated(frames: Vec<AnimationFrame>) -> Self {
        Self {
            animated: true,
            animation_frames: frames,
            ..Self::new()
        }
    }
}

impl Default for TileProperties {
    fn default() -> Self {
        Self::new()
    }
}

/// A single frame of a tile animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationFrame {
    pub tile_id: u32,
    pub duration_ms: u32,
}

// ── TileLayer ────────────────────────────────────────────────

/// A single tile layer containing tile IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct TileLayer {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u32>,
    pub visible: bool,
    pub opacity: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

impl TileLayer {
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            name: name.into(),
            width,
            height,
            data: vec![0; size],
            visible: true,
            opacity: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    pub fn with_data(mut self, data: Vec<u32>) -> Self {
        assert_eq!(data.len(), (self.width * self.height) as usize);
        self.data = data;
        self
    }

    /// Get the tile ID at (x, y) in tile coordinates.
    pub fn get_tile(&self, x: u32, y: u32) -> Option<u32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) as usize;
        Some(self.data[idx])
    }

    /// Set the tile ID at (x, y) in tile coordinates.
    pub fn set_tile(&mut self, x: u32, y: u32, tile_id: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let idx = (y * self.width + x) as usize;
        self.data[idx] = tile_id;
        true
    }

    /// Check if a tile is empty (ID == 0) at (x, y).
    pub fn is_empty(&self, x: u32, y: u32) -> bool {
        self.get_tile(x, y).map_or(true, |id| id == 0)
    }
}

// ── TileMap ──────────────────────────────────────────────────

/// Tile map orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapOrientation {
    Orthogonal,
    Isometric,
}

/// A complete tile map with tile sets and layers.
#[derive(Debug, Clone)]
pub struct TileMap {
    pub orientation: MapOrientation,
    pub tile_sets: Vec<TileSet>,
    pub layers: Vec<TileLayer>,
    pub tile_width: u32,
    pub tile_height: u32,
}

impl TileMap {
    pub fn new(tile_width: u32, tile_height: u32) -> Self {
        Self {
            orientation: MapOrientation::Orthogonal,
            tile_sets: Vec::new(),
            layers: Vec::new(),
            tile_width,
            tile_height,
        }
    }

    pub fn with_orientation(mut self, orientation: MapOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    pub fn add_tile_set(&mut self, tile_set: TileSet) {
        self.tile_sets.push(tile_set);
    }

    pub fn add_layer(&mut self, layer: TileLayer) {
        self.layers.push(layer);
    }

    /// Find the tile set that contains the given global tile ID.
    pub fn tile_set_for_gid(&self, gid: u32) -> Option<&TileSet> {
        self.tile_sets.iter().find(|ts| ts.contains_gid(gid))
    }

    /// Get tile at (tile_x, tile_y) from a specific layer.
    pub fn get_tile(&self, layer_idx: usize, tile_x: u32, tile_y: u32) -> Option<u32> {
        self.layers.get(layer_idx)?.get_tile(tile_x, tile_y)
    }

    /// Convert world pixel coordinates to tile coordinates.
    pub fn world_to_tile(&self, world_x: f32, world_y: f32) -> (i32, i32) {
        match self.orientation {
            MapOrientation::Orthogonal => {
                let tx = (world_x / self.tile_width as f32).floor() as i32;
                let ty = (world_y / self.tile_height as f32).floor() as i32;
                (tx, ty)
            }
            MapOrientation::Isometric => {
                self.iso_world_to_tile(world_x, world_y)
            }
        }
    }

    /// Convert tile coordinates to world pixel coordinates (top-left of tile).
    pub fn tile_to_world(&self, tile_x: i32, tile_y: i32) -> (f32, f32) {
        match self.orientation {
            MapOrientation::Orthogonal => {
                let wx = tile_x as f32 * self.tile_width as f32;
                let wy = tile_y as f32 * self.tile_height as f32;
                (wx, wy)
            }
            MapOrientation::Isometric => {
                self.iso_tile_to_world(tile_x, tile_y)
            }
        }
    }

    /// Get visible tiles within a camera viewport.
    pub fn visible_tiles(
        &self,
        layer_idx: usize,
        camera_x: f32,
        camera_y: f32,
        viewport_w: f32,
        viewport_h: f32,
    ) -> Vec<VisibleTile> {
        let layer = match self.layers.get(layer_idx) {
            Some(l) => l,
            None => return Vec::new(),
        };

        let tw = self.tile_width as f32;
        let th = self.tile_height as f32;

        let start_x = ((camera_x / tw).floor() as i32).max(0) as u32;
        let start_y = ((camera_y / th).floor() as i32).max(0) as u32;
        let end_x = (((camera_x + viewport_w) / tw).ceil() as u32).min(layer.width);
        let end_y = (((camera_y + viewport_h) / th).ceil() as u32).min(layer.height);

        let mut result = Vec::new();
        for ty in start_y..end_y {
            for tx in start_x..end_x {
                if let Some(gid) = layer.get_tile(tx, ty) {
                    if gid != 0 {
                        let (wx, wy) = self.tile_to_world(tx as i32, ty as i32);
                        result.push(VisibleTile {
                            tile_x: tx,
                            tile_y: ty,
                            gid,
                            world_x: wx - camera_x,
                            world_y: wy - camera_y,
                        });
                    }
                }
            }
        }
        result
    }

    /// Check if a tile position has a solid tile (across all layers).
    pub fn is_solid(&self, tile_x: u32, tile_y: u32) -> bool {
        for layer in &self.layers {
            if let Some(gid) = layer.get_tile(tile_x, tile_y) {
                if gid != 0 {
                    if let Some(ts) = self.tile_set_for_gid(gid) {
                        let local_id = gid - ts.first_gid;
                        if let Some(props) = ts.get_properties(local_id) {
                            if props.solid {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    // ── Isometric helpers ────────────────────────────────────

    fn iso_tile_to_world(&self, tile_x: i32, tile_y: i32) -> (f32, f32) {
        let tw = self.tile_width as f32;
        let th = self.tile_height as f32;
        let wx = (tile_x - tile_y) as f32 * (tw / 2.0);
        let wy = (tile_x + tile_y) as f32 * (th / 2.0);
        (wx, wy)
    }

    fn iso_world_to_tile(&self, world_x: f32, world_y: f32) -> (i32, i32) {
        let tw = self.tile_width as f32;
        let th = self.tile_height as f32;
        let tx = ((world_x / (tw / 2.0)) + (world_y / (th / 2.0))) / 2.0;
        let ty = ((world_y / (th / 2.0)) - (world_x / (tw / 2.0))) / 2.0;
        (tx.floor() as i32, ty.floor() as i32)
    }
}

/// A tile that's visible in the current viewport.
#[derive(Debug, Clone, PartialEq)]
pub struct VisibleTile {
    pub tile_x: u32,
    pub tile_y: u32,
    pub gid: u32,
    pub world_x: f32,
    pub world_y: f32,
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tileset() -> TileSet {
        TileSet::new("terrain", 1, 16, 16, 8, 8)
    }

    fn test_map() -> TileMap {
        let mut ts = test_tileset();
        ts.set_properties(1, TileProperties::solid());

        let layer = TileLayer::new("ground", 10, 10)
            .with_data({
                let mut d = vec![0u32; 100];
                d[0] = 1;  // (0,0) = tile 1
                d[1] = 2;  // (1,0) = tile 2 (solid)
                d[11] = 3; // (1,1) = tile 3
                d[55] = 4; // (5,5) = tile 4
                d
            });

        let mut map = TileMap::new(16, 16);
        map.add_tile_set(ts);
        map.add_layer(layer);
        map
    }

    #[test]
    fn tileset_basics() {
        let ts = test_tileset();
        assert_eq!(ts.tile_count(), 64);
        assert!(ts.contains_gid(1));
        assert!(ts.contains_gid(64));
        assert!(!ts.contains_gid(0));
        assert!(!ts.contains_gid(65));
    }

    #[test]
    fn tileset_tile_rect() {
        let ts = test_tileset();
        let r = ts.tile_rect(0).unwrap();
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
        assert_eq!(r.width, 16);

        let r9 = ts.tile_rect(9).unwrap(); // col=1, row=1
        assert_eq!(r9.x, 16);
        assert_eq!(r9.y, 16);
    }

    #[test]
    fn tileset_rect_for_gid() {
        let ts = test_tileset();
        let r = ts.rect_for_gid(1).unwrap(); // gid 1 = local 0
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
    }

    #[test]
    fn tileset_out_of_bounds() {
        let ts = test_tileset();
        assert!(ts.tile_rect(64).is_none());
        assert!(ts.rect_for_gid(65).is_none());
    }

    #[test]
    fn layer_get_set() {
        let mut layer = TileLayer::new("test", 5, 5);
        assert!(layer.is_empty(0, 0));
        layer.set_tile(2, 3, 7);
        assert_eq!(layer.get_tile(2, 3), Some(7));
        assert!(!layer.is_empty(2, 3));
    }

    #[test]
    fn layer_out_of_bounds() {
        let layer = TileLayer::new("test", 5, 5);
        assert_eq!(layer.get_tile(5, 0), None);
        assert_eq!(layer.get_tile(0, 5), None);
    }

    #[test]
    fn map_get_tile() {
        let map = test_map();
        assert_eq!(map.get_tile(0, 0, 0), Some(1));
        assert_eq!(map.get_tile(0, 1, 0), Some(2));
        assert_eq!(map.get_tile(0, 5, 5), Some(4));
        assert_eq!(map.get_tile(0, 9, 9), Some(0));
    }

    #[test]
    fn world_to_tile_orthogonal() {
        let map = test_map();
        assert_eq!(map.world_to_tile(0.0, 0.0), (0, 0));
        assert_eq!(map.world_to_tile(16.0, 0.0), (1, 0));
        assert_eq!(map.world_to_tile(15.9, 15.9), (0, 0));
        assert_eq!(map.world_to_tile(32.0, 48.0), (2, 3));
    }

    #[test]
    fn tile_to_world_orthogonal() {
        let map = test_map();
        assert_eq!(map.tile_to_world(0, 0), (0.0, 0.0));
        assert_eq!(map.tile_to_world(3, 2), (48.0, 32.0));
    }

    #[test]
    fn visible_tiles_viewport() {
        let map = test_map();
        let visible = map.visible_tiles(0, 0.0, 0.0, 48.0, 48.0);
        assert_eq!(visible.len(), 3);
        let gids: Vec<u32> = visible.iter().map(|v| v.gid).collect();
        assert!(gids.contains(&1));
        assert!(gids.contains(&2));
        assert!(gids.contains(&3));
    }

    #[test]
    fn visible_tiles_offset_camera() {
        let map = test_map();
        let visible = map.visible_tiles(0, 64.0, 64.0, 48.0, 48.0);
        let gids: Vec<u32> = visible.iter().map(|v| v.gid).collect();
        assert!(gids.contains(&4));
    }

    #[test]
    fn is_solid() {
        let map = test_map();
        assert!(map.is_solid(1, 0));
        assert!(!map.is_solid(0, 0));
        assert!(!map.is_solid(9, 9));
    }

    #[test]
    fn tile_properties_animated() {
        let mut ts = test_tileset();
        ts.set_properties(5, TileProperties::animated(vec![
            AnimationFrame { tile_id: 5, duration_ms: 100 },
            AnimationFrame { tile_id: 6, duration_ms: 100 },
            AnimationFrame { tile_id: 7, duration_ms: 100 },
        ]));
        let props = ts.get_properties(5).unwrap();
        assert!(props.animated);
        assert_eq!(props.animation_frames.len(), 3);
    }

    #[test]
    fn isometric_coordinates() {
        let map = TileMap::new(64, 32).with_orientation(MapOrientation::Isometric);
        let (wx, wy) = map.tile_to_world(0, 0);
        assert_eq!(wx, 0.0);
        assert_eq!(wy, 0.0);

        let (wx, wy) = map.tile_to_world(1, 0);
        assert_eq!(wx, 32.0);
        assert_eq!(wy, 16.0);

        let (wx, wy) = map.tile_to_world(0, 1);
        assert_eq!(wx, -32.0);
        assert_eq!(wy, 16.0);
    }

    #[test]
    fn isometric_world_to_tile() {
        let map = TileMap::new(64, 32).with_orientation(MapOrientation::Isometric);
        let (tx, ty) = map.world_to_tile(32.0, 16.0);
        assert_eq!(tx, 1);
        assert_eq!(ty, 0);
    }

    #[test]
    fn tileset_with_spacing() {
        let ts = TileSet::new("spaced", 1, 16, 16, 4, 4).with_spacing(2).with_margin(1);
        let r = ts.tile_rect(1).unwrap();
        assert_eq!(r.x, 19);
        assert_eq!(r.y, 1);
    }

    #[test]
    fn layer_set_out_of_bounds() {
        let mut layer = TileLayer::new("test", 5, 5);
        assert!(!layer.set_tile(5, 0, 1));
        assert!(!layer.set_tile(0, 5, 1));
    }

    #[test]
    fn tile_set_for_gid() {
        let mut map = TileMap::new(16, 16);
        let ts1 = TileSet::new("a", 1, 16, 16, 4, 4).with_first_gid(1);
        let ts2 = TileSet::new("b", 2, 16, 16, 4, 4).with_first_gid(17);
        map.add_tile_set(ts1);
        map.add_tile_set(ts2);

        assert_eq!(map.tile_set_for_gid(1).unwrap().name, "a");
        assert_eq!(map.tile_set_for_gid(16).unwrap().name, "a");
        assert_eq!(map.tile_set_for_gid(17).unwrap().name, "b");
        assert!(map.tile_set_for_gid(33).is_none());
    }
}
