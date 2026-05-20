//! Shadow map atlas manager for the lighting engine.
//!
//! Packs multiple shadow maps into a single large texture atlas using
//! rectangular bin-packing. Supports allocate/free of rectangular regions,
//! resolution tiering based on light importance/distance, atlas
//! defragmentation, viewport-to-atlas UV transforms, and used/free
//! space tracking.

use std::collections::HashMap;

// ── Types ──────────────────────────────────────────────────────

/// A unique identifier for an allocated shadow map region.
pub type RegionId = u64;

/// A rectangle within the atlas (in texels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtlasRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl AtlasRect {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    pub fn area(&self) -> u32 { self.width * self.height }

    /// Does this rectangle overlap another?
    pub fn overlaps(&self, other: &Self) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Is this rectangle fully contained within `outer`?
    pub fn contained_in(&self, outer: &Self) -> bool {
        self.x >= outer.x
            && self.y >= outer.y
            && self.x + self.width <= outer.x + outer.width
            && self.y + self.height <= outer.y + outer.height
    }
}

/// UV transform to map a viewport-relative [0,1]^2 coordinate
/// into the atlas UV space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvTransform {
    pub offset_u: f64,
    pub offset_v: f64,
    pub scale_u: f64,
    pub scale_v: f64,
}

impl UvTransform {
    /// Transform a local (u, v) in [0,1]^2 to atlas UV.
    pub fn apply(&self, u: f64, v: f64) -> (f64, f64) {
        (self.offset_u + u * self.scale_u, self.offset_v + v * self.scale_v)
    }
}

// ── Resolution tiering ─────────────────────────────────────────

/// Resolution tier for shadow map allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResolutionTier {
    /// Full resolution.
    High,
    /// Half resolution.
    Medium,
    /// Quarter resolution.
    Low,
    /// Eighth resolution.
    VeryLow,
}

impl ResolutionTier {
    /// Compute the actual resolution from a base resolution.
    pub fn resolve(self, base: u32) -> u32 {
        match self {
            ResolutionTier::High => base,
            ResolutionTier::Medium => (base / 2).max(1),
            ResolutionTier::Low => (base / 4).max(1),
            ResolutionTier::VeryLow => (base / 8).max(1),
        }
    }

    /// Select tier based on light importance and distance.
    pub fn from_importance(importance: f64, distance: f64, max_distance: f64) -> Self {
        let dist_factor = (distance / max_distance.max(1.0)).clamp(0.0, 1.0);
        let combined = importance * (1.0 - dist_factor);
        if combined > 0.75 {
            ResolutionTier::High
        } else if combined > 0.5 {
            ResolutionTier::Medium
        } else if combined > 0.25 {
            ResolutionTier::Low
        } else {
            ResolutionTier::VeryLow
        }
    }
}

// ── Free rectangle tracking ────────────────────────────────────

/// A free rectangle in the atlas available for allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRect {
    rect: AtlasRect,
}

// ── Shadow Atlas ───────────────────────────────────────────────

/// Shadow map atlas manager.
///
/// Uses a shelf / guillotine bin-packing algorithm to allocate
/// rectangular shadow map regions within a single large atlas texture.
#[derive(Debug, Clone)]
pub struct ShadowAtlas {
    /// Atlas width in texels.
    pub width: u32,
    /// Atlas height in texels.
    pub height: u32,
    /// Allocated regions by ID.
    allocated: HashMap<RegionId, AtlasRect>,
    /// Free rectangles available for allocation.
    free_rects: Vec<FreeRect>,
    /// Next region ID.
    next_id: RegionId,
}

impl ShadowAtlas {
    /// Create a new empty atlas.
    pub fn new(width: u32, height: u32) -> Self {
        let full = FreeRect {
            rect: AtlasRect::new(0, 0, width, height),
        };
        Self {
            width,
            height,
            allocated: HashMap::new(),
            free_rects: vec![full],
            next_id: 1,
        }
    }

    /// Total atlas area in texels.
    pub fn total_area(&self) -> u32 { self.width * self.height }

    /// Area currently used by allocated regions.
    pub fn used_area(&self) -> u32 {
        self.allocated.values().map(|r| r.area()).sum()
    }

    /// Area currently free.
    pub fn free_area(&self) -> u32 {
        self.total_area() - self.used_area()
    }

    /// Fraction of atlas that is used.
    pub fn utilization(&self) -> f64 {
        let total = self.total_area();
        if total == 0 { return 0.0; }
        self.used_area() as f64 / total as f64
    }

    /// Number of allocated regions.
    pub fn allocation_count(&self) -> usize { self.allocated.len() }

    /// Try to allocate a rectangular region of the given size.
    /// Returns the region ID and rectangle, or None if no space.
    pub fn allocate(&mut self, req_width: u32, req_height: u32) -> Option<(RegionId, AtlasRect)> {
        if req_width == 0 || req_height == 0 { return None; }
        if req_width > self.width || req_height > self.height { return None; }

        // Find the best-fitting free rectangle (best short-side fit).
        let mut best_idx = None;
        let mut best_score = u32::MAX;

        for (i, fr) in self.free_rects.iter().enumerate() {
            if fr.rect.width >= req_width && fr.rect.height >= req_height {
                let leftover_w = fr.rect.width - req_width;
                let leftover_h = fr.rect.height - req_height;
                let score = leftover_w.min(leftover_h);
                if score < best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
        }

        let idx = best_idx?;
        let free = self.free_rects[idx];

        // Place the allocation at the top-left of the free rect.
        let placed = AtlasRect::new(free.rect.x, free.rect.y, req_width, req_height);

        // Split the remaining space (guillotine: horizontal then vertical).
        self.free_rects.swap_remove(idx);

        // Right remainder.
        if free.rect.width > req_width {
            self.free_rects.push(FreeRect {
                rect: AtlasRect::new(
                    free.rect.x + req_width,
                    free.rect.y,
                    free.rect.width - req_width,
                    req_height,
                ),
            });
        }

        // Bottom remainder.
        if free.rect.height > req_height {
            self.free_rects.push(FreeRect {
                rect: AtlasRect::new(
                    free.rect.x,
                    free.rect.y + req_height,
                    free.rect.width,
                    free.rect.height - req_height,
                ),
            });
        }

        let id = self.next_id;
        self.next_id += 1;
        self.allocated.insert(id, placed);
        Some((id, placed))
    }

    /// Free a previously allocated region.
    pub fn free(&mut self, id: RegionId) -> bool {
        if let Some(rect) = self.allocated.remove(&id) {
            self.free_rects.push(FreeRect { rect });
            self.merge_free_rects();
            true
        } else {
            false
        }
    }

    /// Get the rectangle for a given allocation.
    pub fn get_rect(&self, id: RegionId) -> Option<&AtlasRect> {
        self.allocated.get(&id)
    }

    /// Compute the UV transform for a given allocation.
    pub fn uv_transform(&self, id: RegionId) -> Option<UvTransform> {
        let rect = self.allocated.get(&id)?;
        Some(UvTransform {
            offset_u: rect.x as f64 / self.width as f64,
            offset_v: rect.y as f64 / self.height as f64,
            scale_u: rect.width as f64 / self.width as f64,
            scale_v: rect.height as f64 / self.height as f64,
        })
    }

    /// Merge adjacent free rectangles where possible.
    fn merge_free_rects(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            let len = self.free_rects.len();
            let mut to_remove = Vec::new();
            let mut to_add = Vec::new();

            for i in 0..len {
                if to_remove.contains(&i) { continue; }
                for j in (i + 1)..len {
                    if to_remove.contains(&j) { continue; }
                    let a = self.free_rects[i].rect;
                    let b = self.free_rects[j].rect;

                    // Merge horizontally: same y, same height, adjacent x.
                    if a.y == b.y && a.height == b.height && a.x + a.width == b.x {
                        to_remove.push(i);
                        to_remove.push(j);
                        to_add.push(FreeRect {
                            rect: AtlasRect::new(a.x, a.y, a.width + b.width, a.height),
                        });
                        changed = true;
                        break;
                    }
                    if a.y == b.y && a.height == b.height && b.x + b.width == a.x {
                        to_remove.push(i);
                        to_remove.push(j);
                        to_add.push(FreeRect {
                            rect: AtlasRect::new(b.x, b.y, a.width + b.width, a.height),
                        });
                        changed = true;
                        break;
                    }

                    // Merge vertically: same x, same width, adjacent y.
                    if a.x == b.x && a.width == b.width && a.y + a.height == b.y {
                        to_remove.push(i);
                        to_remove.push(j);
                        to_add.push(FreeRect {
                            rect: AtlasRect::new(a.x, a.y, a.width, a.height + b.height),
                        });
                        changed = true;
                        break;
                    }
                    if a.x == b.x && a.width == b.width && b.y + b.height == a.y {
                        to_remove.push(i);
                        to_remove.push(j);
                        to_add.push(FreeRect {
                            rect: AtlasRect::new(b.x, b.y, b.width, a.height + b.height),
                        });
                        changed = true;
                        break;
                    }
                }
                if changed { break; }
            }

            // Apply removals (sort descending to preserve indices).
            to_remove.sort_unstable();
            to_remove.dedup();
            for &idx in to_remove.iter().rev() {
                self.free_rects.swap_remove(idx);
            }
            self.free_rects.extend(to_add);
        }
    }

    /// Defragment the atlas by repacking all allocations.
    /// Returns a map from RegionId to new AtlasRect.
    pub fn defragment(&mut self) -> HashMap<RegionId, AtlasRect> {
        // Collect all allocations sorted by area (largest first).
        let mut allocs: Vec<(RegionId, AtlasRect)> = self.allocated.drain().collect();
        allocs.sort_by(|a, b| b.1.area().cmp(&a.1.area()));

        // Reset free space.
        self.free_rects.clear();
        self.free_rects.push(FreeRect {
            rect: AtlasRect::new(0, 0, self.width, self.height),
        });

        let mut remap = HashMap::new();
        for (id, old_rect) in allocs {
            if let Some((_, new_rect)) = self.allocate_internal(old_rect.width, old_rect.height) {
                self.allocated.insert(id, new_rect);
                remap.insert(id, new_rect);
            } else {
                // Should not happen if atlas was consistent.
                // Put it back in the old position as fallback.
                self.allocated.insert(id, old_rect);
                remap.insert(id, old_rect);
            }
        }
        remap
    }

    /// Internal allocate (does not bump next_id).
    fn allocate_internal(&mut self, req_width: u32, req_height: u32) -> Option<(RegionId, AtlasRect)> {
        if req_width == 0 || req_height == 0 { return None; }

        let mut best_idx = None;
        let mut best_score = u32::MAX;

        for (i, fr) in self.free_rects.iter().enumerate() {
            if fr.rect.width >= req_width && fr.rect.height >= req_height {
                let leftover_w = fr.rect.width - req_width;
                let leftover_h = fr.rect.height - req_height;
                let score = leftover_w.min(leftover_h);
                if score < best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
        }

        let idx = best_idx?;
        let free = self.free_rects[idx];
        let placed = AtlasRect::new(free.rect.x, free.rect.y, req_width, req_height);

        self.free_rects.swap_remove(idx);

        if free.rect.width > req_width {
            self.free_rects.push(FreeRect {
                rect: AtlasRect::new(
                    free.rect.x + req_width,
                    free.rect.y,
                    free.rect.width - req_width,
                    req_height,
                ),
            });
        }
        if free.rect.height > req_height {
            self.free_rects.push(FreeRect {
                rect: AtlasRect::new(
                    free.rect.x,
                    free.rect.y + req_height,
                    free.rect.width,
                    free.rect.height - req_height,
                ),
            });
        }

        Some((0, placed))
    }

    /// List all allocated regions.
    pub fn allocations(&self) -> Vec<(RegionId, AtlasRect)> {
        let mut result: Vec<(RegionId, AtlasRect)> = self.allocated.iter().map(|(k, v)| (*k, *v)).collect();
        result.sort_by_key(|(id, _)| *id);
        result
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn atlas_new_empty() {
        let a = ShadowAtlas::new(1024, 1024);
        assert_eq!(a.allocation_count(), 0);
        assert_eq!(a.used_area(), 0);
        assert_eq!(a.free_area(), 1024 * 1024);
    }

    #[test]
    fn atlas_allocate_single() {
        let mut a = ShadowAtlas::new(1024, 1024);
        let r = a.allocate(256, 256);
        assert!(r.is_some());
        let (id, rect) = r.unwrap();
        assert_eq!(rect.width, 256);
        assert_eq!(rect.height, 256);
        assert_eq!(a.allocation_count(), 1);
        assert_eq!(a.used_area(), 256 * 256);
        assert!(id > 0);
    }

    #[test]
    fn atlas_allocate_multiple() {
        let mut a = ShadowAtlas::new(512, 512);
        let r1 = a.allocate(128, 128);
        let r2 = a.allocate(128, 128);
        assert!(r1.is_some());
        assert!(r2.is_some());
        let (id1, rect1) = r1.unwrap();
        let (id2, rect2) = r2.unwrap();
        assert_ne!(id1, id2);
        assert!(!rect1.overlaps(&rect2));
    }

    #[test]
    fn atlas_allocate_too_large() {
        let mut a = ShadowAtlas::new(64, 64);
        assert!(a.allocate(128, 128).is_none());
    }

    #[test]
    fn atlas_allocate_zero() {
        let mut a = ShadowAtlas::new(64, 64);
        assert!(a.allocate(0, 32).is_none());
        assert!(a.allocate(32, 0).is_none());
    }

    #[test]
    fn atlas_free_and_reallocate() {
        let mut a = ShadowAtlas::new(256, 256);
        let (id1, _) = a.allocate(256, 128).unwrap();
        let (id2, _) = a.allocate(256, 128).unwrap();
        // Atlas is now full.
        assert!(a.allocate(64, 64).is_none());
        // Free one.
        assert!(a.free(id1));
        // Now we can allocate again.
        assert!(a.allocate(64, 64).is_some());
        assert!(!a.free(id1)); // double-free
        assert!(a.free(id2));
    }

    #[test]
    fn atlas_utilization() {
        let mut a = ShadowAtlas::new(100, 100);
        a.allocate(50, 100).unwrap();
        assert!(approx(a.utilization(), 0.5));
    }

    #[test]
    fn atlas_uv_transform() {
        let mut a = ShadowAtlas::new(1024, 1024);
        let (id, rect) = a.allocate(256, 256).unwrap();
        let uv = a.uv_transform(id).unwrap();
        let (u0, v0) = uv.apply(0.0, 0.0);
        let (u1, v1) = uv.apply(1.0, 1.0);
        // Top-left.
        assert!(approx(u0, rect.x as f64 / 1024.0));
        assert!(approx(v0, rect.y as f64 / 1024.0));
        // Bottom-right.
        assert!(approx(u1, (rect.x + rect.width) as f64 / 1024.0));
        assert!(approx(v1, (rect.y + rect.height) as f64 / 1024.0));
    }

    #[test]
    fn atlas_uv_missing() {
        let a = ShadowAtlas::new(64, 64);
        assert!(a.uv_transform(999).is_none());
    }

    #[test]
    fn atlas_get_rect() {
        let mut a = ShadowAtlas::new(128, 128);
        let (id, rect) = a.allocate(32, 32).unwrap();
        assert_eq!(a.get_rect(id), Some(&rect));
        assert!(a.get_rect(999).is_none());
    }

    #[test]
    fn atlas_no_overlaps_many() {
        let mut a = ShadowAtlas::new(512, 512);
        let mut rects = Vec::new();
        for _ in 0..16 {
            if let Some((_, r)) = a.allocate(64, 64) {
                rects.push(r);
            }
        }
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                assert!(
                    !rects[i].overlaps(&rects[j]),
                    "regions must not overlap"
                );
            }
        }
    }

    #[test]
    fn atlas_defragment() {
        let mut a = ShadowAtlas::new(256, 256);
        let (id1, _) = a.allocate(128, 128).unwrap();
        let (_id2, _) = a.allocate(128, 128).unwrap();
        let (id3, _) = a.allocate(128, 128).unwrap();
        // Free the middle one.
        a.free(id1);
        // Defragment.
        let remap = a.defragment();
        assert!(remap.contains_key(&id3));
        // No overlaps after defrag.
        let allocs = a.allocations();
        for i in 0..allocs.len() {
            for j in (i + 1)..allocs.len() {
                assert!(!allocs[i].1.overlaps(&allocs[j].1));
            }
        }
    }

    #[test]
    fn rect_overlaps() {
        let a = AtlasRect::new(0, 0, 10, 10);
        let b = AtlasRect::new(5, 5, 10, 10);
        assert!(a.overlaps(&b));
        let c = AtlasRect::new(20, 20, 5, 5);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn rect_contained_in() {
        let outer = AtlasRect::new(0, 0, 100, 100);
        let inner = AtlasRect::new(10, 10, 20, 20);
        assert!(inner.contained_in(&outer));
        assert!(!outer.contained_in(&inner));
    }

    #[test]
    fn rect_no_self_overlap_at_edge() {
        let a = AtlasRect::new(0, 0, 10, 10);
        let b = AtlasRect::new(10, 0, 10, 10);
        assert!(!a.overlaps(&b)); // touching but not overlapping
    }

    #[test]
    fn resolution_tier_high() {
        assert_eq!(ResolutionTier::High.resolve(1024), 1024);
    }

    #[test]
    fn resolution_tier_medium() {
        assert_eq!(ResolutionTier::Medium.resolve(1024), 512);
    }

    #[test]
    fn resolution_tier_from_importance_close() {
        let t = ResolutionTier::from_importance(1.0, 1.0, 100.0);
        assert_eq!(t, ResolutionTier::High);
    }

    #[test]
    fn resolution_tier_from_importance_far() {
        let t = ResolutionTier::from_importance(0.5, 90.0, 100.0);
        assert_eq!(t, ResolutionTier::VeryLow);
    }

    #[test]
    fn allocations_sorted_by_id() {
        let mut a = ShadowAtlas::new(256, 256);
        let (id1, _) = a.allocate(64, 64).unwrap();
        let (id2, _) = a.allocate(64, 64).unwrap();
        let (id3, _) = a.allocate(64, 64).unwrap();
        let allocs = a.allocations();
        assert_eq!(allocs[0].0, id1);
        assert_eq!(allocs[1].0, id2);
        assert_eq!(allocs[2].0, id3);
    }

    #[test]
    fn atlas_fill_completely() {
        let mut a = ShadowAtlas::new(64, 64);
        let r = a.allocate(64, 64);
        assert!(r.is_some());
        assert!(a.allocate(1, 1).is_none());
        assert!(approx(a.utilization(), 1.0));
    }
}
