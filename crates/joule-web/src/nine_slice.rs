// Nine-Slice — Nine-patch sprite scaling for UI elements
// 9 regions, stretch/tile modes, content padding, atlas definitions

/// Definition of a nine-slice region within a source image/atlas.
#[derive(Debug, Clone, PartialEq)]
pub struct NineSliceDef {
    /// Source image/atlas region in pixels.
    pub source_x: f32,
    pub source_y: f32,
    pub source_width: f32,
    pub source_height: f32,
    /// Inset from left edge to start of center column.
    pub left: f32,
    /// Inset from right edge to start of center column.
    pub right: f32,
    /// Inset from top edge to start of center row.
    pub top: f32,
    /// Inset from bottom edge to start of center row.
    pub bottom: f32,
    /// Content padding (inset area for text/children).
    pub content_padding: Padding,
    /// Fill mode for edges and center.
    pub fill_mode: FillMode,
}

/// Content padding (inset from nine-slice edges for child content).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Padding {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl Padding {
    pub const ZERO: Self = Self {
        left: 0.0,
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
    };

    pub fn uniform(v: f32) -> Self {
        Self {
            left: v,
            top: v,
            right: v,
            bottom: v,
        }
    }

    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }
}

/// How to fill stretching regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    /// Stretch edges/center to fill.
    Stretch,
    /// Tile (repeat) edges/center.
    Tile,
}

/// UV coordinates for a region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvRegion {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
}

/// A positioned quad with UV mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct NineSliceQuad {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub uv: UvRegion,
}

/// Result of nine-slice generation.
#[derive(Debug, Clone, PartialEq)]
pub struct NineSliceResult {
    /// The quads to render (up to 9 for stretch, more for tile mode).
    pub quads: Vec<NineSliceQuad>,
    /// Content area (where text/children go).
    pub content_rect: ContentRect,
    /// Actual rendered size.
    pub rendered_width: f32,
    pub rendered_height: f32,
}

/// Rectangle for content placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContentRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl NineSliceDef {
    pub fn new(
        source_x: f32,
        source_y: f32,
        source_width: f32,
        source_height: f32,
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
    ) -> Self {
        Self {
            source_x,
            source_y,
            source_width,
            source_height,
            left,
            right,
            top,
            bottom,
            content_padding: Padding::ZERO,
            fill_mode: FillMode::Stretch,
        }
    }

    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.content_padding = padding;
        self
    }

    pub fn with_fill_mode(mut self, mode: FillMode) -> Self {
        self.fill_mode = mode;
        self
    }

    /// Minimum target size (sum of corners).
    pub fn min_width(&self) -> f32 {
        self.left + self.right
    }

    pub fn min_height(&self) -> f32 {
        self.top + self.bottom
    }

    /// Center column width in source.
    fn center_width(&self) -> f32 {
        self.source_width - self.left - self.right
    }

    /// Center row height in source.
    fn center_height(&self) -> f32 {
        self.source_height - self.top - self.bottom
    }

    /// Generate nine-slice quads for a target rectangle.
    pub fn generate(
        &self,
        target_x: f32,
        target_y: f32,
        target_width: f32,
        target_height: f32,
        atlas_width: f32,
        atlas_height: f32,
    ) -> NineSliceResult {
        let tw = target_width.max(self.min_width());
        let th = target_height.max(self.min_height());

        let mid_w = tw - self.left - self.right;
        let mid_h = th - self.top - self.bottom;

        let src_mid_w = self.center_width();
        let src_mid_h = self.center_height();

        let sx = self.source_x;
        let sy = self.source_y;

        // Source regions (3 columns x 3 rows)
        let col_src = [sx, sx + self.left, sx + self.left + src_mid_w];
        let col_src_w = [self.left, src_mid_w, self.right];
        let row_src = [sy, sy + self.top, sy + self.top + src_mid_h];
        let row_src_h = [self.top, src_mid_h, self.bottom];

        // Target regions
        let col_dst = [target_x, target_x + self.left, target_x + self.left + mid_w];
        let col_dst_w = [self.left, mid_w, self.right];
        let row_dst = [target_y, target_y + self.top, target_y + self.top + mid_h];
        let row_dst_h = [self.top, mid_h, self.bottom];

        let mut quads = Vec::new();

        for row in 0..3 {
            for col in 0..3 {
                let dw = col_dst_w[col];
                let dh = row_dst_h[row];
                if dw <= 0.0 || dh <= 0.0 {
                    continue;
                }

                let is_corner = (row == 0 || row == 2) && (col == 0 || col == 2);
                let sw = col_src_w[col];
                let sh = row_src_h[row];

                if is_corner || self.fill_mode == FillMode::Stretch {
                    // Single quad
                    quads.push(NineSliceQuad {
                        x: col_dst[col],
                        y: row_dst[row],
                        width: dw,
                        height: dh,
                        uv: uv_from_pixels(
                            col_src[col],
                            row_src[row],
                            sw,
                            sh,
                            atlas_width,
                            atlas_height,
                        ),
                    });
                } else {
                    // Tile mode: repeat source region
                    let tile_quads = generate_tiled(
                        col_dst[col],
                        row_dst[row],
                        dw,
                        dh,
                        col_src[col],
                        row_src[row],
                        sw,
                        sh,
                        atlas_width,
                        atlas_height,
                    );
                    quads.extend(tile_quads);
                }
            }
        }

        let pad = &self.content_padding;
        let content_rect = ContentRect {
            x: target_x + self.left + pad.left,
            y: target_y + self.top + pad.top,
            width: (mid_w - pad.left - pad.right).max(0.0),
            height: (mid_h - pad.top - pad.bottom).max(0.0),
        };

        NineSliceResult {
            quads,
            content_rect,
            rendered_width: tw,
            rendered_height: th,
        }
    }
}

fn uv_from_pixels(
    px: f32,
    py: f32,
    pw: f32,
    ph: f32,
    atlas_w: f32,
    atlas_h: f32,
) -> UvRegion {
    UvRegion {
        u_min: px / atlas_w,
        v_min: py / atlas_h,
        u_max: (px + pw) / atlas_w,
        v_max: (py + ph) / atlas_h,
    }
}

fn generate_tiled(
    dst_x: f32,
    dst_y: f32,
    dst_w: f32,
    dst_h: f32,
    src_x: f32,
    src_y: f32,
    src_w: f32,
    src_h: f32,
    atlas_w: f32,
    atlas_h: f32,
) -> Vec<NineSliceQuad> {
    let mut quads = Vec::new();
    if src_w <= 0.0 || src_h <= 0.0 {
        return quads;
    }

    let mut y = dst_y;
    while y < dst_y + dst_h - 1e-4 {
        let tile_h = src_h.min(dst_y + dst_h - y);
        let mut x = dst_x;
        while x < dst_x + dst_w - 1e-4 {
            let tile_w = src_w.min(dst_x + dst_w - x);
            quads.push(NineSliceQuad {
                x,
                y,
                width: tile_w,
                height: tile_h,
                uv: uv_from_pixels(src_x, src_y, tile_w, tile_h, atlas_w, atlas_h),
            });
            x += src_w;
        }
        y += src_h;
    }

    quads
}

/// Collection of nine-slice definitions for an atlas.
#[derive(Debug, Clone)]
pub struct NineSliceAtlas {
    pub atlas_width: f32,
    pub atlas_height: f32,
    definitions: Vec<(String, NineSliceDef)>,
}

impl NineSliceAtlas {
    pub fn new(atlas_width: f32, atlas_height: f32) -> Self {
        Self {
            atlas_width,
            atlas_height,
            definitions: Vec::new(),
        }
    }

    pub fn add(&mut self, name: &str, def: NineSliceDef) {
        self.definitions.push((name.to_string(), def));
    }

    pub fn get(&self, name: &str) -> Option<&NineSliceDef> {
        self.definitions
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d)
    }

    pub fn generate_for(
        &self,
        name: &str,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
    ) -> Option<NineSliceResult> {
        let def = self.get(name)?;
        Some(def.generate(x, y, w, h, self.atlas_width, self.atlas_height))
    }

    pub fn count(&self) -> usize {
        self.definitions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_def() -> NineSliceDef {
        NineSliceDef::new(0.0, 0.0, 48.0, 48.0, 16.0, 16.0, 16.0, 16.0)
    }

    #[test]
    fn test_min_size() {
        let d = basic_def();
        assert!((d.min_width() - 32.0).abs() < 1e-6);
        assert!((d.min_height() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_center_width() {
        let d = basic_def();
        assert!((d.center_width() - 16.0).abs() < 1e-6);
        assert!((d.center_height() - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_stretch_nine_quads() {
        let d = basic_def();
        let result = d.generate(0.0, 0.0, 96.0, 96.0, 256.0, 256.0);
        // Should have 9 quads (all regions visible)
        assert_eq!(result.quads.len(), 9);
    }

    #[test]
    fn test_generate_clamps_to_min_size() {
        let d = basic_def();
        let result = d.generate(0.0, 0.0, 10.0, 10.0, 256.0, 256.0);
        assert!((result.rendered_width - 32.0).abs() < 1e-6);
        assert!((result.rendered_height - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_corner_positions() {
        let d = basic_def();
        let result = d.generate(10.0, 20.0, 80.0, 80.0, 256.0, 256.0);
        // Top-left corner
        let tl = &result.quads[0];
        assert!((tl.x - 10.0).abs() < 1e-6);
        assert!((tl.y - 20.0).abs() < 1e-6);
        assert!((tl.width - 16.0).abs() < 1e-6);
        assert!((tl.height - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_uv_normalization() {
        let d = basic_def();
        let result = d.generate(0.0, 0.0, 96.0, 96.0, 256.0, 256.0);
        for q in &result.quads {
            assert!(q.uv.u_min >= 0.0 && q.uv.u_min <= 1.0);
            assert!(q.uv.v_min >= 0.0 && q.uv.v_min <= 1.0);
            assert!(q.uv.u_max >= 0.0 && q.uv.u_max <= 1.0);
            assert!(q.uv.v_max >= 0.0 && q.uv.v_max <= 1.0);
        }
    }

    #[test]
    fn test_content_rect_no_padding() {
        let d = basic_def();
        let result = d.generate(0.0, 0.0, 80.0, 80.0, 256.0, 256.0);
        assert!((result.content_rect.x - 16.0).abs() < 1e-6);
        assert!((result.content_rect.y - 16.0).abs() < 1e-6);
        assert!((result.content_rect.width - 48.0).abs() < 1e-6);
        assert!((result.content_rect.height - 48.0).abs() < 1e-6);
    }

    #[test]
    fn test_content_rect_with_padding() {
        let d = basic_def().with_padding(Padding::uniform(4.0));
        let result = d.generate(0.0, 0.0, 80.0, 80.0, 256.0, 256.0);
        assert!((result.content_rect.x - 20.0).abs() < 1e-6); // 16 + 4
        assert!((result.content_rect.width - 40.0).abs() < 1e-6); // 48 - 4 - 4
    }

    #[test]
    fn test_tile_mode() {
        let d = basic_def().with_fill_mode(FillMode::Tile);
        let result = d.generate(0.0, 0.0, 96.0, 96.0, 256.0, 256.0);
        // Tiled: should have more quads than 9 because edges and center get tiled
        assert!(result.quads.len() >= 9);
    }

    #[test]
    fn test_tile_generates_multiple_quads() {
        let quads = generate_tiled(0.0, 0.0, 50.0, 50.0, 0.0, 0.0, 16.0, 16.0, 256.0, 256.0);
        // 50/16 ~= 4 tiles per axis = 16 quads
        assert!(quads.len() >= 9);
    }

    #[test]
    fn test_padding_zero() {
        let p = Padding::ZERO;
        assert!((p.left).abs() < 1e-6);
        assert!((p.top).abs() < 1e-6);
    }

    #[test]
    fn test_padding_uniform() {
        let p = Padding::uniform(8.0);
        assert!((p.left - 8.0).abs() < 1e-6);
        assert!((p.right - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_padding_new() {
        let p = Padding::new(1.0, 2.0, 3.0, 4.0);
        assert!((p.left - 1.0).abs() < 1e-6);
        assert!((p.bottom - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_atlas_add_and_get() {
        let mut atlas = NineSliceAtlas::new(256.0, 256.0);
        atlas.add("button", basic_def());
        assert_eq!(atlas.count(), 1);
        assert!(atlas.get("button").is_some());
        assert!(atlas.get("missing").is_none());
    }

    #[test]
    fn test_atlas_generate_for() {
        let mut atlas = NineSliceAtlas::new(256.0, 256.0);
        atlas.add("panel", basic_def());
        let result = atlas.generate_for("panel", 0.0, 0.0, 100.0, 100.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().quads.len(), 9);
    }

    #[test]
    fn test_atlas_generate_for_missing() {
        let atlas = NineSliceAtlas::new(256.0, 256.0);
        assert!(atlas.generate_for("nonexistent", 0.0, 0.0, 50.0, 50.0).is_none());
    }

    #[test]
    fn test_fill_mode_enum() {
        assert_ne!(FillMode::Stretch, FillMode::Tile);
    }

    #[test]
    fn test_uv_from_pixels() {
        let uv = uv_from_pixels(64.0, 128.0, 32.0, 32.0, 256.0, 256.0);
        assert!((uv.u_min - 0.25).abs() < 1e-6);
        assert!((uv.v_min - 0.5).abs() < 1e-6);
        assert!((uv.u_max - 0.375).abs() < 1e-6);
        assert!((uv.v_max - 0.625).abs() < 1e-6);
    }

    #[test]
    fn test_nine_slice_exact_min_size() {
        let d = basic_def();
        let result = d.generate(0.0, 0.0, 32.0, 32.0, 256.0, 256.0);
        // At exactly min size, center has 0 width/height, so only corners/edges with area
        // Corners should still be present
        assert!(result.quads.len() >= 4);
    }

    #[test]
    fn test_large_target() {
        let d = basic_def();
        let result = d.generate(0.0, 0.0, 500.0, 500.0, 256.0, 256.0);
        assert_eq!(result.quads.len(), 9);
        // Center quad should be large
        let center = &result.quads[4]; // row1, col1
        assert!(center.width > 400.0);
        assert!(center.height > 400.0);
    }

    #[test]
    fn test_offset_target() {
        let d = basic_def();
        let result = d.generate(100.0, 200.0, 80.0, 80.0, 256.0, 256.0);
        assert!((result.quads[0].x - 100.0).abs() < 1e-6);
        assert!((result.quads[0].y - 200.0).abs() < 1e-6);
    }
}
