//! Tile renderer: tile caching, loading states, URL template expansion,
//! parent/child tile relationships, over-zoom, and tile grid computation.

use std::collections::HashMap;

// ── TileCoord ───────────────────────────────────────────────────

/// A map tile coordinate (z = zoom, x = column, y = row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub z: u32,
    pub x: u32,
    pub y: u32,
}

impl TileCoord {
    pub fn new(z: u32, x: u32, y: u32) -> Self {
        Self { z, x, y }
    }

    /// Get the parent tile at zoom z-1.
    pub fn parent(&self) -> Option<TileCoord> {
        if self.z == 0 {
            return None;
        }
        Some(TileCoord {
            z: self.z - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// Get the four child tiles at zoom z+1.
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

    /// Check if another tile is an ancestor of this tile.
    pub fn is_descendant_of(&self, other: &TileCoord) -> bool {
        if self.z <= other.z {
            return false;
        }
        let dz = self.z - other.z;
        (self.x >> dz) == other.x && (self.y >> dz) == other.y
    }

    /// Get the quadkey string (used by Bing Maps and others).
    pub fn quadkey(&self) -> String {
        let mut key = String::with_capacity(self.z as usize);
        for i in (1..=self.z).rev() {
            let mut digit = 0u8;
            let mask = 1u32 << (i - 1);
            if (self.x & mask) != 0 {
                digit += 1;
            }
            if (self.y & mask) != 0 {
                digit += 2;
            }
            key.push((b'0' + digit) as char);
        }
        key
    }
}

// ── TileState ───────────────────────────────────────────────────

/// Loading state of a tile.
#[derive(Debug, Clone, PartialEq)]
pub enum TileState {
    /// Tile has been requested but not yet loaded.
    Pending,
    /// Tile data is loaded. `data` holds the raw bytes.
    Loaded { data: Vec<u8> },
    /// Tile failed to load.
    Error { message: String },
}

// ── TileCache ───────────────────────────────────────────────────

/// LRU tile cache with a maximum capacity.
#[derive(Debug)]
pub struct TileCache {
    max_capacity: usize,
    entries: HashMap<TileCoord, TileState>,
    /// Access order — most recently accessed at the end.
    order: Vec<TileCoord>,
}

impl TileCache {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            max_capacity,
            entries: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get a tile, updating its position in the LRU order.
    pub fn get(&mut self, coord: &TileCoord) -> Option<&TileState> {
        if self.entries.contains_key(coord) {
            self.touch(coord);
            self.entries.get(coord)
        } else {
            None
        }
    }

    /// Get without updating LRU order.
    pub fn peek(&self, coord: &TileCoord) -> Option<&TileState> {
        self.entries.get(coord)
    }

    /// Insert a tile into the cache, evicting the least-recently-used if full.
    pub fn insert(&mut self, coord: TileCoord, state: TileState) {
        if self.entries.contains_key(&coord) {
            self.entries.insert(coord, state);
            self.touch(&coord);
            return;
        }
        while self.entries.len() >= self.max_capacity {
            self.evict_lru();
        }
        self.entries.insert(coord, state);
        self.order.push(coord);
    }

    /// Remove a tile from the cache.
    pub fn remove(&mut self, coord: &TileCoord) -> Option<TileState> {
        if let Some(state) = self.entries.remove(coord) {
            self.order.retain(|c| c != coord);
            Some(state)
        } else {
            None
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    fn touch(&mut self, coord: &TileCoord) {
        self.order.retain(|c| c != coord);
        self.order.push(*coord);
    }

    fn evict_lru(&mut self) {
        if let Some(oldest) = self.order.first().copied() {
            self.entries.remove(&oldest);
            self.order.remove(0);
        }
    }
}

// ── URL Template ────────────────────────────────────────────────

/// Expand a tile URL template like `https://tile.osm.org/{z}/{x}/{y}.png`.
pub fn expand_tile_url(template: &str, coord: &TileCoord) -> String {
    template
        .replace("{z}", &coord.z.to_string())
        .replace("{x}", &coord.x.to_string())
        .replace("{y}", &coord.y.to_string())
}

/// Expand a tile URL with subdomains (e.g., `{s}` replaced with a/b/c round-robin).
pub fn expand_tile_url_with_subdomain(
    template: &str,
    coord: &TileCoord,
    subdomains: &[&str],
) -> String {
    let idx = ((coord.x + coord.y) as usize) % subdomains.len().max(1);
    let subdomain = subdomains.get(idx).copied().unwrap_or("a");
    expand_tile_url(template, coord).replace("{s}", subdomain)
}

// ── Visible Tiles ───────────────────────────────────────────────

/// Compute visible tile coordinates from bounds and zoom level.
pub fn visible_tiles_from_bounds(
    min_lat: f64,
    max_lat: f64,
    min_lng: f64,
    max_lng: f64,
    zoom: u32,
) -> Vec<TileCoord> {
    let n = 1u32 << zoom;
    let nf = n as f64;

    let x_min = ((min_lng + 180.0) / 360.0 * nf).floor() as i64;
    let x_max = ((max_lng + 180.0) / 360.0 * nf).floor() as i64;

    let y_min = ((1.0 - (max_lat.to_radians().tan() + 1.0 / max_lat.to_radians().cos()).ln()
        / std::f64::consts::PI)
        / 2.0
        * nf)
        .floor() as i64;
    let y_max = ((1.0 - (min_lat.to_radians().tan() + 1.0 / min_lat.to_radians().cos()).ln()
        / std::f64::consts::PI)
        / 2.0
        * nf)
        .floor() as i64;

    let mut tiles = Vec::new();
    for y in y_min.max(0)..=y_max.min(n as i64 - 1) {
        for x in x_min..=x_max {
            let wrapped = ((x % nf as i64) + nf as i64) as u32 % n;
            tiles.push(TileCoord::new(zoom, wrapped, y as u32));
        }
    }
    tiles
}

// ── Over-zoom ───────────────────────────────────────────────────

/// Find the best available parent tile for over-zoom, searching up the tree.
pub fn find_overzoom_parent(coord: &TileCoord, cache: &TileCache, min_zoom: u32) -> Option<TileCoord> {
    let mut current = *coord;
    while current.z > min_zoom {
        if let Some(parent) = current.parent() {
            if let Some(TileState::Loaded { .. }) = cache.peek(&parent) {
                return Some(parent);
            }
            current = parent;
        } else {
            break;
        }
    }
    None
}

// ── Tile Grid ───────────────────────────────────────────────────

/// A tile positioned in the grid for rendering.
#[derive(Debug, Clone)]
pub struct PositionedTile {
    pub coord: TileCoord,
    /// Pixel position (top-left) within the viewport.
    pub screen_x: f64,
    pub screen_y: f64,
    /// Size in screen pixels (256 * scale factor for fractional zoom).
    pub size: f64,
}

/// Compute positioned tiles for rendering given a viewport.
pub fn tile_grid(
    center_lat: f64,
    center_lng: f64,
    zoom: f64,
    viewport_width: u32,
    viewport_height: u32,
) -> Vec<PositionedTile> {
    let z = zoom.floor() as u32;
    let frac = zoom - z as f64;
    let tile_screen_size = 256.0 * 2.0_f64.powf(frac);
    let n = 1u32 << z;
    let nf = n as f64;

    // Center in tile coordinates (fractional)
    let cx = (center_lng + 180.0) / 360.0 * nf;
    let lat_rad = center_lat.to_radians();
    let cy = (1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI)
        / 2.0
        * nf;

    let half_w = (viewport_width as f64) / 2.0;
    let half_h = (viewport_height as f64) / 2.0;

    let min_tx = (cx - half_w / tile_screen_size).floor() as i64;
    let max_tx = (cx + half_w / tile_screen_size).ceil() as i64;
    let min_ty = (cy - half_h / tile_screen_size).floor() as i64;
    let max_ty = (cy + half_h / tile_screen_size).ceil() as i64;

    let mut positioned = Vec::new();
    for ty in min_ty..=max_ty {
        if ty < 0 || ty >= n as i64 {
            continue;
        }
        for tx in min_tx..=max_tx {
            let wrapped = ((tx % n as i64) + n as i64) as u32 % n;
            let screen_x = (tx as f64 - cx) * tile_screen_size + half_w;
            let screen_y = (ty as f64 - cy) * tile_screen_size + half_h;
            positioned.push(PositionedTile {
                coord: TileCoord::new(z, wrapped, ty as u32),
                screen_x,
                screen_y,
                size: tile_screen_size,
            });
        }
    }
    positioned
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_parent() {
        let t = TileCoord::new(3, 5, 3);
        let p = t.parent().unwrap();
        assert_eq!(p, TileCoord::new(2, 2, 1));
    }

    #[test]
    fn test_tile_parent_at_zero() {
        let t = TileCoord::new(0, 0, 0);
        assert!(t.parent().is_none());
    }

    #[test]
    fn test_tile_children() {
        let t = TileCoord::new(1, 0, 1);
        let children = t.children();
        assert_eq!(children[0], TileCoord::new(2, 0, 2));
        assert_eq!(children[1], TileCoord::new(2, 1, 2));
        assert_eq!(children[2], TileCoord::new(2, 0, 3));
        assert_eq!(children[3], TileCoord::new(2, 1, 3));
    }

    #[test]
    fn test_is_descendant() {
        let parent = TileCoord::new(1, 0, 0);
        let child = TileCoord::new(3, 1, 1);
        assert!(child.is_descendant_of(&parent));
        assert!(!parent.is_descendant_of(&child));
    }

    #[test]
    fn test_quadkey() {
        let t = TileCoord::new(3, 5, 3);
        let qk = t.quadkey();
        assert_eq!(qk, "123");
    }

    #[test]
    fn test_tile_cache_lru() {
        let mut cache = TileCache::new(3);
        cache.insert(TileCoord::new(1, 0, 0), TileState::Pending);
        cache.insert(TileCoord::new(1, 1, 0), TileState::Pending);
        cache.insert(TileCoord::new(1, 0, 1), TileState::Pending);
        assert_eq!(cache.len(), 3);
        // Access first to make it recently used
        cache.get(&TileCoord::new(1, 0, 0));
        // Insert fourth — should evict (1,1,0) which is now LRU
        cache.insert(TileCoord::new(1, 1, 1), TileState::Pending);
        assert_eq!(cache.len(), 3);
        assert!(cache.peek(&TileCoord::new(1, 1, 0)).is_none());
        assert!(cache.peek(&TileCoord::new(1, 0, 0)).is_some());
    }

    #[test]
    fn test_tile_cache_update() {
        let mut cache = TileCache::new(10);
        let coord = TileCoord::new(2, 1, 1);
        cache.insert(coord, TileState::Pending);
        cache.insert(coord, TileState::Loaded { data: vec![1, 2, 3] });
        assert_eq!(cache.len(), 1);
        if let Some(TileState::Loaded { data }) = cache.peek(&coord) {
            assert_eq!(data, &[1, 2, 3]);
        } else {
            panic!("expected Loaded");
        }
    }

    #[test]
    fn test_expand_tile_url() {
        let url = expand_tile_url(
            "https://tile.osm.org/{z}/{x}/{y}.png",
            &TileCoord::new(10, 512, 340),
        );
        assert_eq!(url, "https://tile.osm.org/10/512/340.png");
    }

    #[test]
    fn test_expand_tile_url_with_subdomain() {
        let url = expand_tile_url_with_subdomain(
            "https://{s}.tile.osm.org/{z}/{x}/{y}.png",
            &TileCoord::new(5, 15, 10),
            &["a", "b", "c"],
        );
        assert!(url.starts_with("https://"));
        assert!(url.contains("tile.osm.org/5/15/10.png"));
    }

    #[test]
    fn test_visible_tiles_from_bounds() {
        let tiles = visible_tiles_from_bounds(-10.0, 10.0, -10.0, 10.0, 2);
        assert!(!tiles.is_empty());
        for t in &tiles {
            assert_eq!(t.z, 2);
        }
    }

    #[test]
    fn test_find_overzoom_parent() {
        let mut cache = TileCache::new(100);
        let parent = TileCoord::new(2, 1, 1);
        cache.insert(parent, TileState::Loaded { data: vec![0] });
        let child = TileCoord::new(4, 4, 4);
        let found = find_overzoom_parent(&child, &cache, 0);
        assert_eq!(found, Some(parent));
    }

    #[test]
    fn test_tile_grid_not_empty() {
        let grid = tile_grid(0.0, 0.0, 2.0, 512, 512);
        assert!(!grid.is_empty());
        for pt in &grid {
            assert!((pt.size - 256.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = TileCache::new(10);
        let coord = TileCoord::new(1, 0, 0);
        cache.insert(coord, TileState::Pending);
        assert!(cache.remove(&coord).is_some());
        assert!(cache.peek(&coord).is_none());
        assert_eq!(cache.len(), 0);
    }
}
