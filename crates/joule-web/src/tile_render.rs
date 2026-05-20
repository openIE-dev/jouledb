//! Tile rendering: vector tile encoding (simplified MVT-like), raster tile generation
//! from grid data, tile caching strategy, overzooming/underzooming, composite tile
//! layers, and `TileRenderConfig` builder.

use core::fmt;

// ── Tile Coordinate ───────────────────────────────────────────

/// A tile coordinate (z/x/y) in a slippy-map grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl TileCoord {
    pub fn new(z: u8, x: u32, y: u32) -> Self {
        Self { z, x, y }
    }

    /// Number of tiles along each axis at this zoom level.
    pub fn tile_count(&self) -> u32 {
        1u32 << self.z
    }

    /// Whether the x/y values are within the valid range for this zoom.
    pub fn is_valid(&self) -> bool {
        let n = self.tile_count();
        self.x < n && self.y < n
    }

    /// Return the parent tile at one zoom level up.
    pub fn parent(&self) -> Option<TileCoord> {
        if self.z == 0 {
            return None;
        }
        Some(TileCoord::new(self.z - 1, self.x / 2, self.y / 2))
    }

    /// Return the four children at one zoom level down.
    pub fn children(&self) -> [TileCoord; 4] {
        let z = self.z + 1;
        let x = self.x * 2;
        let y = self.y * 2;
        [
            TileCoord::new(z, x, y),
            TileCoord::new(z, x + 1, y),
            TileCoord::new(z, x, y + 1),
            TileCoord::new(z, x + 1, y + 1),
        ]
    }

    /// Quadkey string for this tile (used by some tile providers).
    pub fn quadkey(&self) -> String {
        let mut key = String::with_capacity(self.z as usize);
        for i in (0..self.z).rev() {
            let mut digit = 0u8;
            let mask = 1u32 << i;
            if self.x & mask != 0 {
                digit += 1;
            }
            if self.y & mask != 0 {
                digit += 2;
            }
            key.push((b'0' + digit) as char);
        }
        key
    }
}

impl fmt::Display for TileCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.z, self.x, self.y)
    }
}

// ── Geometry Types for Vector Tiles ───────────────────────────

/// Simplified geometry types for vector tile features.
#[derive(Debug, Clone)]
pub enum TileGeometry {
    Point(i32, i32),
    Line(Vec<(i32, i32)>),
    Polygon(Vec<(i32, i32)>),
}

/// A feature within a vector tile layer.
#[derive(Debug, Clone)]
pub struct TileFeature {
    pub id: u64,
    pub geometry: TileGeometry,
    pub properties: Vec<(String, PropertyValue)>,
}

/// Property value types.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl fmt::Display for PropertyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyValue::Str(v) => write!(f, "{v}"),
            PropertyValue::Int(v) => write!(f, "{v}"),
            PropertyValue::Float(v) => write!(f, "{v}"),
            PropertyValue::Bool(v) => write!(f, "{v}"),
        }
    }
}

// ── Vector Tile Layer ─────────────────────────────────────────

/// A named layer within a vector tile, containing features.
#[derive(Debug, Clone)]
pub struct VectorTileLayer {
    pub name: String,
    pub extent: u32,
    pub features: Vec<TileFeature>,
}

impl VectorTileLayer {
    pub fn new(name: &str, extent: u32) -> Self {
        Self { name: name.to_string(), extent, features: Vec::new() }
    }

    pub fn with_feature(mut self, feature: TileFeature) -> Self {
        self.features.push(feature);
        self
    }

    /// Encode features into a simplified MVT-like binary format.
    /// Returns packed bytes: [layer_name_len:u8][name bytes][feature count:u32 LE][features...].
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let name_bytes = self.name.as_bytes();
        buf.push(name_bytes.len().min(255) as u8);
        buf.extend_from_slice(&name_bytes[..name_bytes.len().min(255)]);
        buf.extend_from_slice(&(self.features.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.extent.to_le_bytes());
        for feat in &self.features {
            buf.extend_from_slice(&feat.id.to_le_bytes());
            match &feat.geometry {
                TileGeometry::Point(x, y) => {
                    buf.push(1);
                    buf.extend_from_slice(&x.to_le_bytes());
                    buf.extend_from_slice(&y.to_le_bytes());
                }
                TileGeometry::Line(pts) => {
                    buf.push(2);
                    buf.extend_from_slice(&(pts.len() as u32).to_le_bytes());
                    for (x, y) in pts {
                        buf.extend_from_slice(&x.to_le_bytes());
                        buf.extend_from_slice(&y.to_le_bytes());
                    }
                }
                TileGeometry::Polygon(pts) => {
                    buf.push(3);
                    buf.extend_from_slice(&(pts.len() as u32).to_le_bytes());
                    for (x, y) in pts {
                        buf.extend_from_slice(&x.to_le_bytes());
                        buf.extend_from_slice(&y.to_le_bytes());
                    }
                }
            }
        }
        buf
    }
}

// ── Raster Tile ───────────────────────────────────────────────

/// A raster tile: a 2D grid of RGBA pixels.
#[derive(Debug, Clone)]
pub struct RasterTile {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[u8; 4]>,
}

impl RasterTile {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self { width, height, pixels: vec![[0, 0, 0, 0]; n] }
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x < self.width && y < self.height {
            self.pixels[(y * self.width + x) as usize] = rgba;
        }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x < self.width && y < self.height {
            self.pixels[(y * self.width + x) as usize]
        } else {
            [0, 0, 0, 0]
        }
    }

    /// Generate a raster tile from a grid of f64 values using a color function.
    pub fn from_grid(grid: &[f64], grid_w: u32, grid_h: u32, color_fn: &dyn Fn(f64) -> [u8; 4]) -> Self {
        let mut tile = Self::new(grid_w, grid_h);
        for y in 0..grid_h {
            for x in 0..grid_w {
                let val = grid[(y * grid_w + x) as usize];
                tile.set_pixel(x, y, color_fn(val));
            }
        }
        tile
    }
}

// ── Tile Cache ────────────────────────────────────────────────

/// Simple LRU-style tile cache.
#[derive(Debug)]
pub struct TileCache {
    entries: Vec<(TileCoord, Vec<u8>)>,
    capacity: usize,
}

impl TileCache {
    pub fn new(capacity: usize) -> Self {
        Self { entries: Vec::new(), capacity: capacity.max(1) }
    }

    /// Get cached data for a tile coordinate.
    pub fn get(&self, coord: &TileCoord) -> Option<&[u8]> {
        self.entries.iter().find(|(c, _)| c == coord).map(|(_, d)| d.as_slice())
    }

    /// Insert (or update) tile data, evicting oldest if at capacity.
    pub fn insert(&mut self, coord: TileCoord, data: Vec<u8>) {
        if let Some(pos) = self.entries.iter().position(|(c, _)| c == &coord) {
            self.entries.remove(pos);
        }
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push((coord, data));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── Overzooming / Underzooming ────────────────────────────────

/// Strategy for handling tile requests beyond available zoom levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoomStrategy {
    /// Use the parent tile and crop/scale up.
    Overzoom,
    /// Use children tiles and composite down.
    Underzoom,
    /// Return empty.
    Empty,
}

/// Compute the source tile coordinate when overzooming.
pub fn overzoom_source(coord: &TileCoord, max_source_zoom: u8) -> (TileCoord, u8) {
    if coord.z <= max_source_zoom {
        return (*coord, 0);
    }
    let diff = coord.z - max_source_zoom;
    let source = TileCoord::new(max_source_zoom, coord.x >> diff, coord.y >> diff);
    (source, diff)
}

/// Compute the sub-tile region when overzooming (x_off, y_off, subdivisions).
pub fn overzoom_region(coord: &TileCoord, source: &TileCoord) -> (u32, u32, u32) {
    if coord.z <= source.z {
        return (0, 0, 1);
    }
    let diff = coord.z - source.z;
    let subdivisions = 1u32 << diff;
    let x_off = coord.x - (source.x << diff);
    let y_off = coord.y - (source.y << diff);
    (x_off, y_off, subdivisions)
}

// ── Composite Layers ──────────────────────────────────────────

/// Composite multiple raster tiles on top of each other using alpha blending.
pub fn composite_tiles(layers: &[&RasterTile]) -> Option<RasterTile> {
    if layers.is_empty() {
        return None;
    }
    let w = layers[0].width;
    let h = layers[0].height;
    let mut result = RasterTile::new(w, h);
    for layer in layers {
        if layer.width != w || layer.height != h {
            continue;
        }
        for i in 0..result.pixels.len() {
            let src = layer.pixels[i];
            let dst = &mut result.pixels[i];
            let sa = src[3] as f64 / 255.0;
            let da = dst[3] as f64 / 255.0;
            let out_a = sa + da * (1.0 - sa);
            if out_a > 0.0 {
                for c in 0..3 {
                    dst[c] = ((src[c] as f64 * sa + dst[c] as f64 * da * (1.0 - sa)) / out_a)
                        .round() as u8;
                }
            }
            dst[3] = (out_a * 255.0).round() as u8;
        }
    }
    Some(result)
}

// ── TileRenderConfig Builder ──────────────────────────────────

/// Configuration for the tile rendering pipeline.
#[derive(Debug, Clone)]
pub struct TileRenderConfig {
    pub tile_size: u32,
    pub min_zoom: u8,
    pub max_zoom: u8,
    pub max_source_zoom: u8,
    pub cache_capacity: usize,
    pub zoom_strategy: ZoomStrategy,
    pub buffer_pixels: u32,
}

impl Default for TileRenderConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            min_zoom: 0,
            max_zoom: 22,
            max_source_zoom: 14,
            cache_capacity: 256,
            zoom_strategy: ZoomStrategy::Overzoom,
            buffer_pixels: 64,
        }
    }
}

impl TileRenderConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tile_size(mut self, size: u32) -> Self {
        self.tile_size = size;
        self
    }

    pub fn with_zoom_range(mut self, min: u8, max: u8) -> Self {
        self.min_zoom = min;
        self.max_zoom = max;
        self
    }

    pub fn with_max_source_zoom(mut self, z: u8) -> Self {
        self.max_source_zoom = z;
        self
    }

    pub fn with_cache_capacity(mut self, cap: usize) -> Self {
        self.cache_capacity = cap;
        self
    }

    pub fn with_zoom_strategy(mut self, strategy: ZoomStrategy) -> Self {
        self.zoom_strategy = strategy;
        self
    }

    pub fn with_buffer(mut self, px: u32) -> Self {
        self.buffer_pixels = px;
        self
    }

    /// Compute tiles needed to cover a viewport (px_w x px_h) at center tile.
    pub fn visible_tiles(&self, center: &TileCoord, px_w: u32, px_h: u32) -> Vec<TileCoord> {
        let ts = self.tile_size;
        let half_w = ((px_w / 2 + self.buffer_pixels) / ts + 1) as i64;
        let half_h = ((px_h / 2 + self.buffer_pixels) / ts + 1) as i64;
        let n = center.tile_count() as i64;
        let mut tiles = Vec::new();
        for dy in -half_h..=half_h {
            for dx in -half_w..=half_w {
                let tx = ((center.x as i64 + dx) % n + n) % n;
                let ty = center.y as i64 + dy;
                if ty >= 0 && ty < n {
                    tiles.push(TileCoord::new(center.z, tx as u32, ty as u32));
                }
            }
        }
        tiles
    }
}

impl fmt::Display for TileRenderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TileRenderConfig({}px, z{}-{}, cache={})",
            self.tile_size, self.min_zoom, self.max_zoom, self.cache_capacity
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_coord_valid() {
        assert!(TileCoord::new(2, 3, 3).is_valid());
        assert!(!TileCoord::new(2, 4, 0).is_valid());
    }

    #[test]
    fn tile_coord_parent() {
        let t = TileCoord::new(3, 5, 6);
        let p = t.parent().unwrap();
        assert_eq!(p, TileCoord::new(2, 2, 3));
    }

    #[test]
    fn tile_coord_parent_z0() {
        assert!(TileCoord::new(0, 0, 0).parent().is_none());
    }

    #[test]
    fn tile_coord_children() {
        let t = TileCoord::new(1, 0, 1);
        let c = t.children();
        assert_eq!(c[0], TileCoord::new(2, 0, 2));
        assert_eq!(c[3], TileCoord::new(2, 1, 3));
    }

    #[test]
    fn tile_coord_quadkey() {
        let t = TileCoord::new(3, 5, 2);
        let qk = t.quadkey();
        assert_eq!(qk.len(), 3);
    }

    #[test]
    fn tile_coord_display() {
        assert_eq!(format!("{}", TileCoord::new(5, 10, 15)), "5/10/15");
    }

    #[test]
    fn vector_tile_encode_roundtrip_size() {
        let layer = VectorTileLayer::new("roads", 4096)
            .with_feature(TileFeature {
                id: 1,
                geometry: TileGeometry::Point(100, 200),
                properties: vec![],
            });
        let data = layer.encode();
        assert!(!data.is_empty());
        assert!(data.len() > 10);
    }

    #[test]
    fn raster_tile_set_get() {
        let mut tile = RasterTile::new(4, 4);
        tile.set_pixel(2, 3, [255, 0, 128, 255]);
        assert_eq!(tile.get_pixel(2, 3), [255, 0, 128, 255]);
        assert_eq!(tile.get_pixel(0, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn raster_tile_out_of_bounds() {
        let tile = RasterTile::new(4, 4);
        assert_eq!(tile.get_pixel(10, 10), [0, 0, 0, 0]);
    }

    #[test]
    fn raster_from_grid() {
        let grid = vec![0.0, 0.5, 1.0, 0.25];
        let tile = RasterTile::from_grid(&grid, 2, 2, &|v| {
            let c = (v * 255.0) as u8;
            [c, c, c, 255]
        });
        assert_eq!(tile.get_pixel(1, 0)[0], 127); // 0.5 * 255.0 = 127.5, truncated to 127
    }

    #[test]
    fn tile_cache_insert_get() {
        let mut cache = TileCache::new(4);
        let coord = TileCoord::new(2, 1, 1);
        cache.insert(coord, vec![1, 2, 3]);
        assert_eq!(cache.get(&coord), Some(&[1u8, 2, 3][..]));
    }

    #[test]
    fn tile_cache_eviction() {
        let mut cache = TileCache::new(2);
        cache.insert(TileCoord::new(0, 0, 0), vec![0]);
        cache.insert(TileCoord::new(1, 0, 0), vec![1]);
        cache.insert(TileCoord::new(2, 0, 0), vec![2]);
        assert!(cache.get(&TileCoord::new(0, 0, 0)).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn overzoom_source_within_range() {
        let coord = TileCoord::new(10, 500, 300);
        let (src, diff) = overzoom_source(&coord, 14);
        assert_eq!(src, coord);
        assert_eq!(diff, 0);
    }

    #[test]
    fn overzoom_source_beyond_range() {
        let coord = TileCoord::new(16, 40000, 30000);
        let (src, diff) = overzoom_source(&coord, 14);
        assert_eq!(src.z, 14);
        assert_eq!(diff, 2);
    }

    #[test]
    fn overzoom_region_same_level() {
        let coord = TileCoord::new(5, 10, 10);
        let (xo, yo, sub) = overzoom_region(&coord, &coord);
        assert_eq!((xo, yo, sub), (0, 0, 1));
    }

    #[test]
    fn composite_tiles_alpha_blend() {
        let mut bottom = RasterTile::new(2, 2);
        bottom.set_pixel(0, 0, [255, 0, 0, 255]);
        let mut top = RasterTile::new(2, 2);
        top.set_pixel(0, 0, [0, 0, 255, 128]);
        let result = composite_tiles(&[&bottom, &top]).unwrap();
        let px = result.get_pixel(0, 0);
        assert!(px[2] > 0); // blue channel present
        assert_eq!(px[3], 255); // full alpha
    }

    #[test]
    fn config_visible_tiles_nonempty() {
        let cfg = TileRenderConfig::new().with_tile_size(256);
        let center = TileCoord::new(4, 8, 8);
        let tiles = cfg.visible_tiles(&center, 512, 512);
        assert!(!tiles.is_empty());
        assert!(tiles.contains(&center));
    }

    #[test]
    fn config_display() {
        let cfg = TileRenderConfig::new();
        assert!(format!("{cfg}").contains("256px"));
    }

    #[test]
    fn tile_cache_clear() {
        let mut cache = TileCache::new(10);
        cache.insert(TileCoord::new(1, 0, 0), vec![1]);
        cache.clear();
        assert!(cache.is_empty());
    }
}
