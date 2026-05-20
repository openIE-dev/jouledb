// Glyph Cache — LRU glyph rasterization cache with atlas management
// Shelf-based packing, cache eviction, subpixel positioning, miss-rate statistics

use std::collections::HashMap;

/// Key for a cached glyph: font, glyph, size, and subpixel position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    pub font_id: u32,
    pub glyph_id: u32,
    /// Size in pixels, quantized to integer.
    pub size_px: u32,
    /// Subpixel position (0..3), quantized to 4 sub-positions.
    pub subpixel_x: u8,
}

impl GlyphKey {
    pub fn new(font_id: u32, glyph_id: u32, size_px: u32, fractional_x: f32) -> Self {
        let subpixel_x = ((fractional_x.fract().abs() * 4.0) as u8).min(3);
        Self {
            font_id,
            glyph_id,
            size_px,
            subpixel_x,
        }
    }
}

/// UV rectangle in the atlas texture [0..1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvRect {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
}

/// A cached glyph entry.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphEntry {
    pub key: GlyphKey,
    pub uv: UvRect,
    /// Pixel offset in atlas.
    pub atlas_x: u32,
    pub atlas_y: u32,
    pub width: u32,
    pub height: u32,
    /// Bearing from pen position.
    pub bearing_x: f32,
    pub bearing_y: f32,
    /// Horizontal advance.
    pub advance: f32,
}

/// Shelf in the atlas packer.
#[derive(Debug, Clone)]
struct Shelf {
    y: u32,
    height: u32,
    x_cursor: u32,
}

/// Atlas texture with shelf-based bin packing.
#[derive(Debug, Clone)]
pub struct GlyphAtlas {
    pub width: u32,
    pub height: u32,
    /// Grayscale pixel data (alpha channel).
    pub data: Vec<u8>,
    shelves: Vec<Shelf>,
    padding: u32,
}

impl GlyphAtlas {
    pub fn new(width: u32, height: u32, padding: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; (width * height) as usize],
            shelves: Vec::new(),
            padding,
        }
    }

    /// Try to allocate a region of (w, h) in the atlas.
    /// Returns (x, y) on success, None if full.
    pub fn allocate(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        let pw = w + self.padding;
        let ph = h + self.padding;

        // Try existing shelves
        for shelf in &mut self.shelves {
            if ph <= shelf.height && shelf.x_cursor + pw <= self.width {
                let x = shelf.x_cursor;
                let y = shelf.y;
                shelf.x_cursor += pw;
                return Some((x, y));
            }
        }

        // New shelf
        let shelf_y = self
            .shelves
            .last()
            .map(|s| s.y + s.height + self.padding)
            .unwrap_or(0);

        if shelf_y + ph > self.height {
            return None; // Full
        }

        let x = 0u32;
        self.shelves.push(Shelf {
            y: shelf_y,
            height: ph,
            x_cursor: pw,
        });

        Some((x, shelf_y))
    }

    /// Write glyph data into the atlas at (ax, ay).
    pub fn write_glyph(&mut self, ax: u32, ay: u32, w: u32, h: u32, glyph_data: &[u8]) {
        for row in 0..h {
            for col in 0..w {
                let src = (row * w + col) as usize;
                let dst_x = ax + col;
                let dst_y = ay + row;
                if dst_x < self.width && dst_y < self.height {
                    let dst = (dst_y * self.width + dst_x) as usize;
                    if src < glyph_data.len() && dst < self.data.len() {
                        self.data[dst] = glyph_data[src];
                    }
                }
            }
        }
    }

    /// Grow the atlas by doubling height.
    pub fn grow(&mut self) -> bool {
        let new_height = self.height * 2;
        if new_height > 8192 {
            return false; // Limit
        }
        self.data.resize((self.width * new_height) as usize, 0);
        self.height = new_height;
        true
    }

    /// Compute UV rect for a region.
    pub fn uv_for(&self, x: u32, y: u32, w: u32, h: u32) -> UvRect {
        let aw = self.width as f32;
        let ah = self.height as f32;
        UvRect {
            u_min: x as f32 / aw,
            v_min: y as f32 / ah,
            u_max: (x + w) as f32 / aw,
            v_max: (y + h) as f32 / ah,
        }
    }
}

/// Cache hit/miss statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheStats {
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn miss_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.misses as f64 / total as f64
        }
    }

    pub fn hit_rate(&self) -> f64 {
        1.0 - self.miss_rate()
    }
}

/// LRU-based glyph cache backed by an atlas.
#[derive(Debug)]
pub struct GlyphCache {
    pub atlas: GlyphAtlas,
    entries: HashMap<GlyphKey, usize>, // key -> index in order vec
    order: Vec<(GlyphKey, GlyphEntry)>,
    pub max_entries: usize,
    pub stats: CacheStats,
}

impl GlyphCache {
    pub fn new(atlas_width: u32, atlas_height: u32, max_entries: usize) -> Self {
        Self {
            atlas: GlyphAtlas::new(atlas_width, atlas_height, 1),
            entries: HashMap::new(),
            order: Vec::new(),
            max_entries,
            stats: CacheStats::new(),
        }
    }

    /// Look up a glyph. Returns entry and marks as recently used.
    pub fn get(&mut self, key: &GlyphKey) -> Option<&GlyphEntry> {
        if let Some(&idx) = self.entries.get(key) {
            self.stats.hits += 1;
            // Move to end (most recently used)
            if idx < self.order.len() - 1 {
                let item = self.order.remove(idx);
                self.order.push(item);
                // Rebuild index
                self.rebuild_index();
            }
            let last = self.order.len() - 1;
            Some(&self.order[last].1)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a glyph into the cache and atlas.
    /// `rasterize` is called to produce glyph bitmap data.
    pub fn insert(
        &mut self,
        key: GlyphKey,
        width: u32,
        height: u32,
        bearing_x: f32,
        bearing_y: f32,
        advance: f32,
        glyph_data: &[u8],
    ) -> Option<&GlyphEntry> {
        // Evict if at capacity
        while self.order.len() >= self.max_entries {
            self.evict_lru();
        }

        // Try to allocate in atlas (grow if needed)
        let pos = match self.atlas.allocate(width, height) {
            Some(p) => p,
            None => {
                if self.atlas.grow() {
                    self.atlas.allocate(width, height)?
                } else {
                    // Evict and rebuild atlas (simplified: just fail)
                    return None;
                }
            }
        };

        self.atlas.write_glyph(pos.0, pos.1, width, height, glyph_data);
        let uv = self.atlas.uv_for(pos.0, pos.1, width, height);

        let entry = GlyphEntry {
            key,
            uv,
            atlas_x: pos.0,
            atlas_y: pos.1,
            width,
            height,
            bearing_x,
            bearing_y,
            advance,
        };

        let idx = self.order.len();
        self.order.push((key, entry));
        self.entries.insert(key, idx);

        Some(&self.order[idx].1)
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if self.order.is_empty() {
            return;
        }
        let (evicted_key, _) = self.order.remove(0);
        self.entries.remove(&evicted_key);
        self.stats.evictions += 1;
        self.rebuild_index();
    }

    fn rebuild_index(&mut self) {
        self.entries.clear();
        for (i, (key, _)) in self.order.iter().enumerate() {
            self.entries.insert(*key, i);
        }
    }

    /// Batch lookup: returns entries for all found keys and list of missing keys.
    pub fn batch_lookup(&mut self, keys: &[GlyphKey]) -> (Vec<GlyphEntry>, Vec<GlyphKey>) {
        let mut found = Vec::new();
        let mut missing = Vec::new();
        // Clone keys to avoid borrow issues
        let keys_vec: Vec<GlyphKey> = keys.to_vec();
        for key in &keys_vec {
            if self.entries.contains_key(key) {
                // Record hit
                self.stats.hits += 1;
                let idx = self.entries[key];
                found.push(self.order[idx].1.clone());
            } else {
                self.stats.misses += 1;
                missing.push(*key);
            }
        }
        (found, missing)
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(glyph_id: u32) -> GlyphKey {
        GlyphKey {
            font_id: 0,
            glyph_id,
            size_px: 16,
            subpixel_x: 0,
        }
    }

    #[test]
    fn test_glyph_key_new() {
        let k = GlyphKey::new(0, 65, 16, 0.3);
        assert_eq!(k.subpixel_x, 1); // 0.3 * 4 = 1.2 -> 1
    }

    #[test]
    fn test_glyph_key_subpixel_quantization() {
        let k0 = GlyphKey::new(0, 65, 16, 0.0);
        let k1 = GlyphKey::new(0, 65, 16, 0.25);
        let k2 = GlyphKey::new(0, 65, 16, 0.5);
        let k3 = GlyphKey::new(0, 65, 16, 0.75);
        assert_eq!(k0.subpixel_x, 0);
        assert_eq!(k1.subpixel_x, 1);
        assert_eq!(k2.subpixel_x, 2);
        assert_eq!(k3.subpixel_x, 3);
    }

    #[test]
    fn test_atlas_allocate_single() {
        let mut atlas = GlyphAtlas::new(64, 64, 1);
        let pos = atlas.allocate(10, 12);
        assert!(pos.is_some());
        let (x, y) = pos.unwrap();
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_atlas_allocate_multiple_same_shelf() {
        let mut atlas = GlyphAtlas::new(128, 128, 1);
        let p1 = atlas.allocate(10, 10).unwrap();
        let p2 = atlas.allocate(10, 10).unwrap();
        // Same shelf, different x
        assert_eq!(p1.1, p2.1);
        assert!(p2.0 > p1.0);
    }

    #[test]
    fn test_atlas_allocate_new_shelf() {
        let mut atlas = GlyphAtlas::new(32, 128, 1);
        // Fill first shelf
        atlas.allocate(15, 10).unwrap();
        atlas.allocate(15, 10).unwrap();
        // Third should go to new shelf
        let p3 = atlas.allocate(15, 10).unwrap();
        assert!(p3.1 > 0);
    }

    #[test]
    fn test_atlas_full() {
        let mut atlas = GlyphAtlas::new(16, 16, 0);
        let r = atlas.allocate(20, 20);
        assert!(r.is_none());
    }

    #[test]
    fn test_atlas_write_glyph() {
        let mut atlas = GlyphAtlas::new(32, 32, 0);
        let pos = atlas.allocate(4, 4).unwrap();
        let data = vec![255u8; 16];
        atlas.write_glyph(pos.0, pos.1, 4, 4, &data);
        assert_eq!(atlas.data[0], 255);
    }

    #[test]
    fn test_atlas_grow() {
        let mut atlas = GlyphAtlas::new(64, 64, 0);
        assert!(atlas.grow());
        assert_eq!(atlas.height, 128);
        assert_eq!(atlas.data.len(), 64 * 128);
    }

    #[test]
    fn test_atlas_uv_rect() {
        let atlas = GlyphAtlas::new(256, 256, 0);
        let uv = atlas.uv_for(0, 0, 128, 128);
        assert!((uv.u_min).abs() < 1e-6);
        assert!((uv.v_min).abs() < 1e-6);
        assert!((uv.u_max - 0.5).abs() < 1e-6);
        assert!((uv.v_max - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cache_stats_new() {
        let s = CacheStats::new();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert!((s.miss_rate()).abs() < 1e-10);
    }

    #[test]
    fn test_cache_stats_miss_rate() {
        let s = CacheStats {
            hits: 80,
            misses: 20,
            evictions: 0,
        };
        assert!((s.miss_rate() - 0.2).abs() < 1e-10);
        assert!((s.hit_rate() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = GlyphCache::new(128, 128, 100);
        let k = key(65);
        let data = vec![200u8; 64];
        cache.insert(k, 8, 8, 1.0, 7.0, 10.0, &data);
        let entry = cache.get(&k);
        assert!(entry.is_some());
        let e = entry.unwrap();
        assert_eq!(e.width, 8);
        assert!((e.advance - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = GlyphCache::new(128, 128, 100);
        let k = key(99);
        assert!(cache.get(&k).is_none());
        assert_eq!(cache.stats.misses, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = GlyphCache::new(512, 512, 3);
        for i in 0..5 {
            let k = key(i);
            let data = vec![100u8; 16];
            cache.insert(k, 4, 4, 0.0, 4.0, 5.0, &data);
        }
        // Max 3, so 2 should have been evicted
        assert_eq!(cache.len(), 3);
        assert!(cache.stats.evictions >= 2);
    }

    #[test]
    fn test_cache_lru_order() {
        let mut cache = GlyphCache::new(512, 512, 3);
        let data = vec![100u8; 16];
        for i in 0..3 {
            cache.insert(key(i), 4, 4, 0.0, 4.0, 5.0, &data);
        }
        // Access key(0) to make it recently used
        cache.get(&key(0));
        // Insert key(3) — should evict key(1), not key(0)
        cache.insert(key(3), 4, 4, 0.0, 4.0, 5.0, &data);
        assert!(cache.entries.contains_key(&key(0)));
        assert!(!cache.entries.contains_key(&key(1)));
    }

    #[test]
    fn test_batch_lookup() {
        let mut cache = GlyphCache::new(256, 256, 100);
        let data = vec![100u8; 16];
        cache.insert(key(65), 4, 4, 0.0, 4.0, 5.0, &data);
        cache.insert(key(66), 4, 4, 0.0, 4.0, 5.0, &data);

        let keys = vec![key(65), key(66), key(67)];
        cache.reset_stats();
        let (found, missing) = cache.batch_lookup(&keys);
        assert_eq!(found.len(), 2);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].glyph_id, 67);
        assert_eq!(cache.stats.hits, 2);
        assert_eq!(cache.stats.misses, 1);
    }

    #[test]
    fn test_cache_len_and_empty() {
        let mut cache = GlyphCache::new(128, 128, 100);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        let data = vec![0u8; 4];
        cache.insert(key(1), 2, 2, 0.0, 2.0, 3.0, &data);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_reset_stats() {
        let mut cache = GlyphCache::new(128, 128, 100);
        cache.get(&key(1));
        assert_eq!(cache.stats.misses, 1);
        cache.reset_stats();
        assert_eq!(cache.stats.misses, 0);
    }

    #[test]
    fn test_subpixel_keys_distinct() {
        let k1 = GlyphKey::new(0, 65, 16, 0.0);
        let k2 = GlyphKey::new(0, 65, 16, 0.3);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_atlas_grow_limit() {
        let mut atlas = GlyphAtlas::new(64, 8192, 0);
        assert!(!atlas.grow()); // Already at limit
    }

    #[test]
    fn test_cache_insert_returns_entry() {
        let mut cache = GlyphCache::new(128, 128, 100);
        let data = vec![255u8; 36];
        let entry = cache.insert(key(42), 6, 6, 1.0, 5.0, 7.0, &data);
        assert!(entry.is_some());
        let e = entry.unwrap();
        assert_eq!(e.key.glyph_id, 42);
        assert!((e.bearing_x - 1.0).abs() < 1e-6);
    }
}
