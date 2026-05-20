//! Multiple viewport support.
//!
//! Each viewport has position, size (pixels or percentage), camera
//! reference, and render order. Split-screen layouts (horizontal,
//! vertical, quad). Letterboxing / pillarboxing for aspect ratio
//! preservation. Mouse-to-viewport hit testing.

use std::collections::HashMap;

// ── Dimension unit ──────────────────────────────────────────────

/// Size expressed in pixels or as a percentage of the parent container.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DimUnit {
    Px(f64),
    Pct(f64),
}

impl DimUnit {
    /// Resolve to pixels given the parent dimension in pixels.
    pub fn resolve(&self, parent_px: f64) -> f64 {
        match self {
            DimUnit::Px(v) => *v,
            DimUnit::Pct(p) => parent_px * p / 100.0,
        }
    }
}

// ── Viewport rect ───────────────────────────────────────────────

/// Resolved pixel rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, width: w, height: h }
    }

    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px < self.x + self.width
            && py >= self.y && py < self.y + self.height
    }

    pub fn aspect_ratio(&self) -> f64 {
        if self.height.abs() < 1e-12 { return 1.0; }
        self.width / self.height
    }

    /// Letterbox / pillarbox to fit `target_aspect` inside this rect.
    pub fn fit_aspect(&self, target_aspect: f64) -> Rect {
        let my_aspect = self.aspect_ratio();
        if (my_aspect - target_aspect).abs() < 1e-9 {
            return *self;
        }
        if my_aspect > target_aspect {
            // Pillarbox: viewport is wider than target.
            let new_w = self.height * target_aspect;
            let offset = (self.width - new_w) * 0.5;
            Rect::new(self.x + offset, self.y, new_w, self.height)
        } else {
            // Letterbox: viewport is taller than target.
            let new_h = self.width / target_aspect;
            let offset = (self.height - new_h) * 0.5;
            Rect::new(self.x, self.y + offset, self.width, new_h)
        }
    }
}

// ── Viewport ────────────────────────────────────────────────────

type ViewportId = u32;

/// A viewport definition with position, size, camera reference, and render order.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewport {
    pub id: ViewportId,
    pub name: String,
    pub x: DimUnit,
    pub y: DimUnit,
    pub width: DimUnit,
    pub height: DimUnit,
    /// Identifier of the camera to use (managed externally).
    pub camera_id: u64,
    /// Lower values render first (background); higher values render on top.
    pub render_order: i32,
    pub enabled: bool,
    /// If set, the viewport content will be letterboxed/pillarboxed
    /// to preserve this aspect ratio.
    pub target_aspect: Option<f64>,
}

impl Viewport {
    /// Resolve this viewport into a pixel `Rect` given the container size.
    pub fn resolve(&self, container_w: f64, container_h: f64) -> Rect {
        let rx = self.x.resolve(container_w);
        let ry = self.y.resolve(container_h);
        let rw = self.width.resolve(container_w);
        let rh = self.height.resolve(container_h);
        let rect = Rect::new(rx, ry, rw, rh);
        match self.target_aspect {
            Some(ar) => rect.fit_aspect(ar),
            None => rect,
        }
    }
}

// ── Split-screen presets ────────────────────────────────────────

/// Pre-defined split-screen layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitLayout {
    /// Two viewports side-by-side (left / right).
    Horizontal,
    /// Two viewports stacked (top / bottom).
    Vertical,
    /// Four viewports in a 2x2 grid.
    Quad,
}

/// Generate viewport definitions for a split-screen layout.
/// `camera_ids` must have the correct length: 2 for H/V, 4 for Quad.
pub fn split_layout(layout: SplitLayout, camera_ids: &[u64]) -> Vec<Viewport> {
    match layout {
        SplitLayout::Horizontal => {
            assert!(camera_ids.len() >= 2, "Horizontal split needs 2 cameras");
            vec![
                Viewport {
                    id: 0, name: "left".into(),
                    x: DimUnit::Pct(0.0), y: DimUnit::Pct(0.0),
                    width: DimUnit::Pct(50.0), height: DimUnit::Pct(100.0),
                    camera_id: camera_ids[0], render_order: 0, enabled: true,
                    target_aspect: None,
                },
                Viewport {
                    id: 1, name: "right".into(),
                    x: DimUnit::Pct(50.0), y: DimUnit::Pct(0.0),
                    width: DimUnit::Pct(50.0), height: DimUnit::Pct(100.0),
                    camera_id: camera_ids[1], render_order: 0, enabled: true,
                    target_aspect: None,
                },
            ]
        }
        SplitLayout::Vertical => {
            assert!(camera_ids.len() >= 2, "Vertical split needs 2 cameras");
            vec![
                Viewport {
                    id: 0, name: "top".into(),
                    x: DimUnit::Pct(0.0), y: DimUnit::Pct(0.0),
                    width: DimUnit::Pct(100.0), height: DimUnit::Pct(50.0),
                    camera_id: camera_ids[0], render_order: 0, enabled: true,
                    target_aspect: None,
                },
                Viewport {
                    id: 1, name: "bottom".into(),
                    x: DimUnit::Pct(0.0), y: DimUnit::Pct(50.0),
                    width: DimUnit::Pct(100.0), height: DimUnit::Pct(50.0),
                    camera_id: camera_ids[1], render_order: 0, enabled: true,
                    target_aspect: None,
                },
            ]
        }
        SplitLayout::Quad => {
            assert!(camera_ids.len() >= 4, "Quad split needs 4 cameras");
            vec![
                Viewport {
                    id: 0, name: "top_left".into(),
                    x: DimUnit::Pct(0.0), y: DimUnit::Pct(0.0),
                    width: DimUnit::Pct(50.0), height: DimUnit::Pct(50.0),
                    camera_id: camera_ids[0], render_order: 0, enabled: true,
                    target_aspect: None,
                },
                Viewport {
                    id: 1, name: "top_right".into(),
                    x: DimUnit::Pct(50.0), y: DimUnit::Pct(0.0),
                    width: DimUnit::Pct(50.0), height: DimUnit::Pct(50.0),
                    camera_id: camera_ids[1], render_order: 0, enabled: true,
                    target_aspect: None,
                },
                Viewport {
                    id: 2, name: "bottom_left".into(),
                    x: DimUnit::Pct(0.0), y: DimUnit::Pct(50.0),
                    width: DimUnit::Pct(50.0), height: DimUnit::Pct(50.0),
                    camera_id: camera_ids[2], render_order: 0, enabled: true,
                    target_aspect: None,
                },
                Viewport {
                    id: 3, name: "bottom_right".into(),
                    x: DimUnit::Pct(50.0), y: DimUnit::Pct(50.0),
                    width: DimUnit::Pct(50.0), height: DimUnit::Pct(50.0),
                    camera_id: camera_ids[3], render_order: 0, enabled: true,
                    target_aspect: None,
                },
            ]
        }
    }
}

// ── Viewport manager ────────────────────────────────────────────

/// Manages multiple viewports and provides hit-testing.
#[derive(Debug)]
pub struct ViewportManager {
    viewports: HashMap<ViewportId, Viewport>,
    next_id: ViewportId,
    container_width: f64,
    container_height: f64,
}

impl ViewportManager {
    pub fn new(container_width: f64, container_height: f64) -> Self {
        Self {
            viewports: HashMap::new(),
            next_id: 0,
            container_width,
            container_height,
        }
    }

    pub fn set_container_size(&mut self, w: f64, h: f64) {
        self.container_width = w;
        self.container_height = h;
    }

    pub fn container_size(&self) -> (f64, f64) {
        (self.container_width, self.container_height)
    }

    /// Add a viewport and return its ID.
    pub fn add(&mut self, mut vp: Viewport) -> ViewportId {
        let id = self.next_id;
        self.next_id += 1;
        vp.id = id;
        self.viewports.insert(id, vp);
        id
    }

    /// Add multiple viewports from a split layout preset.
    pub fn add_split(&mut self, layout: SplitLayout, camera_ids: &[u64]) -> Vec<ViewportId> {
        let vps = split_layout(layout, camera_ids);
        vps.into_iter().map(|vp| self.add(vp)).collect()
    }

    pub fn remove(&mut self, id: ViewportId) -> bool {
        self.viewports.remove(&id).is_some()
    }

    pub fn get(&self, id: ViewportId) -> Option<&Viewport> {
        self.viewports.get(&id)
    }

    pub fn get_mut(&mut self, id: ViewportId) -> Option<&mut Viewport> {
        self.viewports.get_mut(&id)
    }

    pub fn count(&self) -> usize {
        self.viewports.len()
    }

    /// Resolve a viewport to a pixel rect.
    pub fn resolve(&self, id: ViewportId) -> Option<Rect> {
        self.viewports.get(&id).map(|vp| vp.resolve(self.container_width, self.container_height))
    }

    /// Return all enabled viewports sorted by render order (ascending).
    pub fn sorted_viewports(&self) -> Vec<(ViewportId, Rect)> {
        let mut items: Vec<_> = self.viewports.values()
            .filter(|vp| vp.enabled)
            .map(|vp| (vp.id, vp.render_order, vp.resolve(self.container_width, self.container_height)))
            .collect();
        items.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        items.into_iter().map(|(id, _, rect)| (id, rect)).collect()
    }

    /// Hit-test a mouse position (in container pixels).
    /// Returns the viewport with the **highest** render order that contains the point.
    pub fn hit_test(&self, px: f64, py: f64) -> Option<ViewportId> {
        let mut candidates: Vec<_> = self.viewports.values()
            .filter(|vp| vp.enabled)
            .filter_map(|vp| {
                let rect = vp.resolve(self.container_width, self.container_height);
                if rect.contains(px, py) {
                    Some((vp.id, vp.render_order))
                } else {
                    None
                }
            })
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates.first().map(|(id, _)| *id)
    }

    /// Convert container pixel coordinates to normalized viewport
    /// coordinates [0, 1] for the given viewport.
    pub fn to_viewport_coords(&self, id: ViewportId, px: f64, py: f64) -> Option<(f64, f64)> {
        let rect = self.resolve(id)?;
        if rect.width < 1e-12 || rect.height < 1e-12 {
            return None;
        }
        Some((
            (px - rect.x) / rect.width,
            (py - rect.y) / rect.height,
        ))
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn vp(name: &str, x: DimUnit, y: DimUnit, w: DimUnit, h: DimUnit, cam: u64, order: i32) -> Viewport {
        Viewport {
            id: 0, name: name.into(), x, y, width: w, height: h,
            camera_id: cam, render_order: order, enabled: true,
            target_aspect: None,
        }
    }

    #[test]
    fn test_dim_px() {
        assert!((DimUnit::Px(100.0).resolve(800.0) - 100.0).abs() < EPS);
    }

    #[test]
    fn test_dim_pct() {
        assert!((DimUnit::Pct(50.0).resolve(800.0) - 400.0).abs() < EPS);
    }

    #[test]
    fn test_rect_contains() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(50.0, 40.0));
        assert!(!r.contains(5.0, 40.0));
        assert!(!r.contains(50.0, 80.0));
    }

    #[test]
    fn test_rect_aspect() {
        let r = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        assert!((r.aspect_ratio() - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_letterbox() {
        let r = Rect::new(0.0, 0.0, 800.0, 800.0);
        let fitted = r.fit_aspect(16.0 / 9.0);
        assert!((fitted.width - 800.0).abs() < EPS);
        assert!(fitted.height < 800.0);
        // Centered.
        assert!(fitted.y > 0.0);
    }

    #[test]
    fn test_pillarbox() {
        let r = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        let fitted = r.fit_aspect(1.0);
        assert!((fitted.height - 1080.0).abs() < EPS);
        assert!((fitted.width - 1080.0).abs() < EPS);
        assert!(fitted.x > 0.0);
    }

    #[test]
    fn test_fit_same_aspect() {
        let r = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        let fitted = r.fit_aspect(1920.0 / 1080.0);
        assert!((fitted.x - r.x).abs() < EPS);
        assert!((fitted.y - r.y).abs() < EPS);
    }

    #[test]
    fn test_viewport_resolve_px() {
        let vp = vp("test", DimUnit::Px(10.0), DimUnit::Px(20.0),
            DimUnit::Px(200.0), DimUnit::Px(100.0), 1, 0);
        let rect = vp.resolve(800.0, 600.0);
        assert!((rect.x - 10.0).abs() < EPS);
        assert!((rect.y - 20.0).abs() < EPS);
        assert!((rect.width - 200.0).abs() < EPS);
    }

    #[test]
    fn test_viewport_resolve_pct() {
        let vp = vp("test", DimUnit::Pct(0.0), DimUnit::Pct(0.0),
            DimUnit::Pct(50.0), DimUnit::Pct(100.0), 1, 0);
        let rect = vp.resolve(1920.0, 1080.0);
        assert!((rect.width - 960.0).abs() < EPS);
        assert!((rect.height - 1080.0).abs() < EPS);
    }

    #[test]
    fn test_manager_add_remove() {
        let mut mgr = ViewportManager::new(1920.0, 1080.0);
        let id = mgr.add(vp("a", DimUnit::Px(0.0), DimUnit::Px(0.0),
            DimUnit::Px(960.0), DimUnit::Px(1080.0), 1, 0));
        assert_eq!(mgr.count(), 1);
        assert!(mgr.remove(id));
        assert_eq!(mgr.count(), 0);
    }

    #[test]
    fn test_manager_hit_test() {
        let mut mgr = ViewportManager::new(800.0, 600.0);
        mgr.add(vp("full", DimUnit::Px(0.0), DimUnit::Px(0.0),
            DimUnit::Px(800.0), DimUnit::Px(600.0), 1, 0));
        let hit = mgr.hit_test(400.0, 300.0);
        assert!(hit.is_some());
    }

    #[test]
    fn test_hit_test_highest_order() {
        let mut mgr = ViewportManager::new(800.0, 600.0);
        let _bg = mgr.add(vp("bg", DimUnit::Px(0.0), DimUnit::Px(0.0),
            DimUnit::Px(800.0), DimUnit::Px(600.0), 1, 0));
        let overlay = mgr.add(vp("overlay", DimUnit::Px(100.0), DimUnit::Px(100.0),
            DimUnit::Px(200.0), DimUnit::Px(200.0), 2, 10));
        let hit = mgr.hit_test(150.0, 150.0);
        assert_eq!(hit, Some(overlay));
    }

    #[test]
    fn test_hit_test_miss() {
        let mut mgr = ViewportManager::new(800.0, 600.0);
        mgr.add(vp("small", DimUnit::Px(100.0), DimUnit::Px(100.0),
            DimUnit::Px(50.0), DimUnit::Px(50.0), 1, 0));
        assert!(mgr.hit_test(0.0, 0.0).is_none());
    }

    #[test]
    fn test_sorted_viewports() {
        let mut mgr = ViewportManager::new(800.0, 600.0);
        mgr.add(vp("b", DimUnit::Px(0.0), DimUnit::Px(0.0),
            DimUnit::Px(100.0), DimUnit::Px(100.0), 1, 5));
        mgr.add(vp("a", DimUnit::Px(0.0), DimUnit::Px(0.0),
            DimUnit::Px(100.0), DimUnit::Px(100.0), 2, 1));
        let sorted = mgr.sorted_viewports();
        assert_eq!(sorted.len(), 2);
        assert!(sorted[0].0 != sorted[1].0);
        // First should have lower render_order.
        let first_vp = mgr.get(sorted[0].0).unwrap();
        let second_vp = mgr.get(sorted[1].0).unwrap();
        assert!(first_vp.render_order <= second_vp.render_order);
    }

    #[test]
    fn test_disabled_viewport_excluded() {
        let mut mgr = ViewportManager::new(800.0, 600.0);
        let id = mgr.add(vp("disabled", DimUnit::Px(0.0), DimUnit::Px(0.0),
            DimUnit::Px(800.0), DimUnit::Px(600.0), 1, 0));
        mgr.get_mut(id).unwrap().enabled = false;
        assert!(mgr.hit_test(400.0, 300.0).is_none());
        assert!(mgr.sorted_viewports().is_empty());
    }

    #[test]
    fn test_horizontal_split() {
        let vps = split_layout(SplitLayout::Horizontal, &[1, 2]);
        assert_eq!(vps.len(), 2);
        let r0 = vps[0].resolve(1920.0, 1080.0);
        let r1 = vps[1].resolve(1920.0, 1080.0);
        assert!((r0.width - 960.0).abs() < EPS);
        assert!((r1.x - 960.0).abs() < EPS);
    }

    #[test]
    fn test_vertical_split() {
        let vps = split_layout(SplitLayout::Vertical, &[1, 2]);
        assert_eq!(vps.len(), 2);
        let r0 = vps[0].resolve(1920.0, 1080.0);
        let r1 = vps[1].resolve(1920.0, 1080.0);
        assert!((r0.height - 540.0).abs() < EPS);
        assert!((r1.y - 540.0).abs() < EPS);
    }

    #[test]
    fn test_quad_split() {
        let vps = split_layout(SplitLayout::Quad, &[1, 2, 3, 4]);
        assert_eq!(vps.len(), 4);
        for vp in &vps {
            let r = vp.resolve(800.0, 600.0);
            assert!((r.width - 400.0).abs() < EPS);
            assert!((r.height - 300.0).abs() < EPS);
        }
    }

    #[test]
    fn test_to_viewport_coords() {
        let mut mgr = ViewportManager::new(800.0, 600.0);
        let id = mgr.add(vp("test", DimUnit::Px(100.0), DimUnit::Px(200.0),
            DimUnit::Px(400.0), DimUnit::Px(200.0), 1, 0));
        let (u, v) = mgr.to_viewport_coords(id, 300.0, 300.0).unwrap();
        assert!((u - 0.5).abs() < EPS);
        assert!((v - 0.5).abs() < EPS);
    }

    #[test]
    fn test_container_size() {
        let mut mgr = ViewportManager::new(800.0, 600.0);
        assert_eq!(mgr.container_size(), (800.0, 600.0));
        mgr.set_container_size(1024.0, 768.0);
        assert_eq!(mgr.container_size(), (1024.0, 768.0));
    }

    #[test]
    fn test_add_split_returns_ids() {
        let mut mgr = ViewportManager::new(1920.0, 1080.0);
        let ids = mgr.add_split(SplitLayout::Horizontal, &[10, 20]);
        assert_eq!(ids.len(), 2);
        assert_eq!(mgr.count(), 2);
    }

    #[test]
    fn test_viewport_with_target_aspect() {
        let mut v = vp("test", DimUnit::Px(0.0), DimUnit::Px(0.0),
            DimUnit::Px(800.0), DimUnit::Px(800.0), 1, 0);
        v.target_aspect = Some(16.0 / 9.0);
        let rect = v.resolve(800.0, 800.0);
        // Should be pillarboxed.
        assert!((rect.width - 800.0).abs() < EPS);
        assert!(rect.height < 800.0);
    }
}
