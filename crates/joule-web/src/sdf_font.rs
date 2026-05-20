// SDF Font — Signed Distance Field font rendering
// Glyph SDF generation, threshold rendering, anti-aliasing, outlines, shadows, font atlas

use std::collections::HashMap;

/// A 2D grid of signed distance values for a single glyph.
/// Positive values = outside the glyph shape. Negative = inside.
#[derive(Debug, Clone, PartialEq)]
pub struct SdfGlyph {
    pub width: usize,
    pub height: usize,
    /// Row-major distance values. length = width * height.
    pub distances: Vec<f32>,
}

impl SdfGlyph {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            distances: vec![f32::MAX; width * height],
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.distances[y * self.width + x]
        } else {
            f32::MAX
        }
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, val: f32) {
        if x < self.width && y < self.height {
            self.distances[y * self.width + x] = val;
        }
    }

    /// Generate SDF from a binary bitmap glyph (brute-force).
    /// `bitmap`: row-major, true = inside glyph.
    /// `spread`: max distance to compute (in pixels).
    pub fn from_bitmap(bitmap: &[bool], bw: usize, bh: usize, spread: f32) -> Self {
        let mut sdf = Self::new(bw, bh);
        for y in 0..bh {
            for x in 0..bw {
                let inside = bitmap[y * bw + x];
                let mut min_dist = spread;
                // Brute-force: scan all pixels for nearest opposite
                let search = spread.ceil() as usize + 1;
                let x_lo = x.saturating_sub(search);
                let y_lo = y.saturating_sub(search);
                let x_hi = (x + search + 1).min(bw);
                let y_hi = (y + search + 1).min(bh);
                for sy in y_lo..y_hi {
                    for sx in x_lo..x_hi {
                        let other = bitmap[sy * bw + sx];
                        if other != inside {
                            let dx = x as f32 - sx as f32;
                            let dy = y as f32 - sy as f32;
                            let d = (dx * dx + dy * dy).sqrt();
                            if d < min_dist {
                                min_dist = d;
                            }
                        }
                    }
                }
                let signed = if inside { -min_dist } else { min_dist };
                sdf.set(x, y, signed.clamp(-spread, spread));
            }
        }
        sdf
    }
}

/// Smoothstep interpolation (Hermite).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Render modes for SDF glyphs.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderParams {
    /// Distance threshold for the glyph edge (typically 0.0).
    pub threshold: f32,
    /// Smoothing half-width for anti-aliasing.
    pub smooth_radius: f32,
    /// Optional outline: outline_width > 0 enables it.
    pub outline_width: f32,
    /// Outline color alpha (0..1).
    pub outline_alpha: f32,
    /// Shadow offset (dx, dy) in pixels.
    pub shadow_offset: (f32, f32),
    /// Shadow softness (blur radius in distance units).
    pub shadow_softness: f32,
    /// Shadow alpha.
    pub shadow_alpha: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            smooth_radius: 0.75,
            outline_width: 0.0,
            outline_alpha: 1.0,
            shadow_offset: (0.0, 0.0),
            shadow_softness: 0.0,
            shadow_alpha: 0.0,
        }
    }
}

/// Rendered pixel with fill, outline, and shadow alpha.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedPixel {
    pub fill_alpha: f32,
    pub outline_alpha: f32,
    pub shadow_alpha: f32,
}

/// Render a single SDF sample into alphas.
pub fn render_sdf_pixel(distance: f32, params: &RenderParams) -> RenderedPixel {
    // Fill: smoothstep from threshold
    let fill_alpha = 1.0
        - smoothstep(
            params.threshold - params.smooth_radius,
            params.threshold + params.smooth_radius,
            distance,
        );

    // Outline
    let outline_alpha = if params.outline_width > 0.0 {
        let outer = params.threshold + params.outline_width;
        let outer_alpha = 1.0
            - smoothstep(
                outer - params.smooth_radius,
                outer + params.smooth_radius,
                distance,
            );
        // Outline is the ring between fill and outer
        (outer_alpha - fill_alpha).max(0.0) * params.outline_alpha
    } else {
        0.0
    };

    // Shadow (not computed here — needs offset sampling)
    RenderedPixel {
        fill_alpha,
        outline_alpha,
        shadow_alpha: 0.0,
    }
}

/// Render an entire SDF glyph to an alpha image (row-major).
pub fn render_sdf_glyph(glyph: &SdfGlyph, params: &RenderParams) -> Vec<RenderedPixel> {
    let mut pixels = Vec::with_capacity(glyph.width * glyph.height);
    for y in 0..glyph.height {
        for x in 0..glyph.width {
            let d = glyph.get(x, y);
            let mut px = render_sdf_pixel(d, params);

            // Shadow: sample with offset
            if params.shadow_alpha > 0.0 {
                let sx = x as f32 - params.shadow_offset.0;
                let sy = y as f32 - params.shadow_offset.1;
                let sd = bilinear_sample(glyph, sx, sy);
                let shadow_fill = 1.0
                    - smoothstep(
                        params.threshold - params.shadow_softness,
                        params.threshold + params.shadow_softness,
                        sd,
                    );
                px.shadow_alpha = shadow_fill * params.shadow_alpha;
            }

            pixels.push(px);
        }
    }
    pixels
}

/// Bilinear sample from an SDF glyph at fractional coordinates.
fn bilinear_sample(glyph: &SdfGlyph, x: f32, y: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let ix = x0 as isize;
    let iy = y0 as isize;

    let sample = |px: isize, py: isize| -> f32 {
        if px >= 0 && py >= 0 && (px as usize) < glyph.width && (py as usize) < glyph.height {
            glyph.get(px as usize, py as usize)
        } else {
            glyph.distances.last().copied().unwrap_or(f32::MAX)
        }
    };

    let s00 = sample(ix, iy);
    let s10 = sample(ix + 1, iy);
    let s01 = sample(ix, iy + 1);
    let s11 = sample(ix + 1, iy + 1);

    let top = s00 * (1.0 - fx) + s10 * fx;
    let bot = s01 * (1.0 - fx) + s11 * fx;
    top * (1.0 - fy) + bot * fy
}

/// Glyph metrics for text layout.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphMetrics {
    pub glyph_id: u32,
    /// Horizontal advance in pixels.
    pub advance_x: f32,
    /// Bearing from baseline to top-left of glyph.
    pub bearing_x: f32,
    pub bearing_y: f32,
    /// Size of the glyph bounding box.
    pub size_x: f32,
    pub size_y: f32,
}

/// A packed glyph in a font atlas.
#[derive(Debug, Clone, PartialEq)]
pub struct AtlasEntry {
    pub glyph_id: u32,
    /// UV coordinates in atlas [0..1].
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
    /// Pixel offset in atlas.
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// Font atlas: packs multiple SDF glyphs into a single texture.
/// Uses a simple shelf-based packing algorithm.
#[derive(Debug, Clone)]
pub struct FontAtlas {
    pub atlas_width: usize,
    pub atlas_height: usize,
    /// Combined distance field texture (row-major).
    pub data: Vec<f32>,
    /// Entries keyed by glyph_id.
    pub entries: HashMap<u32, AtlasEntry>,
    /// Metrics keyed by glyph_id.
    pub metrics: HashMap<u32, GlyphMetrics>,
    // Shelf packer state
    shelf_y: usize,
    shelf_height: usize,
    shelf_x: usize,
    padding: usize,
}

impl FontAtlas {
    pub fn new(width: usize, height: usize, padding: usize) -> Self {
        Self {
            atlas_width: width,
            atlas_height: height,
            data: vec![f32::MAX; width * height],
            entries: HashMap::new(),
            metrics: HashMap::new(),
            shelf_y: 0,
            shelf_height: 0,
            shelf_x: 0,
            padding,
        }
    }

    /// Pack a glyph SDF into the atlas. Returns true on success.
    pub fn pack_glyph(
        &mut self,
        glyph_id: u32,
        sdf: &SdfGlyph,
        glyph_metrics: GlyphMetrics,
    ) -> bool {
        let gw = sdf.width + self.padding;
        let gh = sdf.height + self.padding;

        // Check if glyph fits on current shelf
        if self.shelf_x + gw > self.atlas_width {
            // New shelf
            self.shelf_y += self.shelf_height + self.padding;
            self.shelf_x = 0;
            self.shelf_height = 0;
        }

        if self.shelf_y + gh > self.atlas_height {
            return false; // Atlas full
        }

        let px = self.shelf_x;
        let py = self.shelf_y;

        // Copy SDF data into atlas
        for row in 0..sdf.height {
            for col in 0..sdf.width {
                let dst = (py + row) * self.atlas_width + (px + col);
                if dst < self.data.len() {
                    self.data[dst] = sdf.get(col, row);
                }
            }
        }

        let aw = self.atlas_width as f32;
        let ah = self.atlas_height as f32;

        let entry = AtlasEntry {
            glyph_id,
            u_min: px as f32 / aw,
            v_min: py as f32 / ah,
            u_max: (px + sdf.width) as f32 / aw,
            v_max: (py + sdf.height) as f32 / ah,
            x: px,
            y: py,
            width: sdf.width,
            height: sdf.height,
        };

        self.entries.insert(glyph_id, entry);
        self.metrics.insert(glyph_id, glyph_metrics);

        self.shelf_x += gw;
        if gh > self.shelf_height {
            self.shelf_height = gh;
        }

        true
    }

    /// Look up atlas entry for a glyph.
    pub fn get_entry(&self, glyph_id: u32) -> Option<&AtlasEntry> {
        self.entries.get(&glyph_id)
    }

    /// Look up metrics for a glyph.
    pub fn get_metrics(&self, glyph_id: u32) -> Option<&GlyphMetrics> {
        self.metrics.get(&glyph_id)
    }

    /// Number of packed glyphs.
    pub fn glyph_count(&self) -> usize {
        self.entries.len()
    }

    /// Sample the atlas at integer coordinates.
    pub fn sample(&self, x: usize, y: usize) -> f32 {
        if x < self.atlas_width && y < self.atlas_height {
            self.data[y * self.atlas_width + x]
        } else {
            f32::MAX
        }
    }
}

/// Composite alpha layers: shadow behind outline behind fill.
pub fn composite_alpha(pixel: &RenderedPixel) -> f32 {
    // Over operator: front over back
    let outline_over_shadow =
        pixel.outline_alpha + pixel.shadow_alpha * (1.0 - pixel.outline_alpha);
    let fill_over_rest = pixel.fill_alpha + outline_over_shadow * (1.0 - pixel.fill_alpha);
    fill_over_rest.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square_bitmap(size: usize, border: usize) -> Vec<bool> {
        let mut bmp = vec![false; size * size];
        for y in border..(size - border) {
            for x in border..(size - border) {
                bmp[y * size + x] = true;
            }
        }
        bmp
    }

    #[test]
    fn test_sdf_glyph_new() {
        let g = SdfGlyph::new(4, 3);
        assert_eq!(g.width, 4);
        assert_eq!(g.height, 3);
        assert_eq!(g.distances.len(), 12);
        assert_eq!(g.get(0, 0), f32::MAX);
    }

    #[test]
    fn test_sdf_glyph_set_get() {
        let mut g = SdfGlyph::new(5, 5);
        g.set(2, 3, -1.5);
        assert!((g.get(2, 3) - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn test_sdf_glyph_out_of_bounds() {
        let g = SdfGlyph::new(3, 3);
        assert_eq!(g.get(10, 10), f32::MAX);
    }

    #[test]
    fn test_from_bitmap_center_inside() {
        // 5x5 bitmap with 3x3 filled center
        let bmp = make_square_bitmap(5, 1);
        let sdf = SdfGlyph::from_bitmap(&bmp, 5, 5, 4.0);
        // Center pixel (2,2) should be inside (negative)
        assert!(sdf.get(2, 2) < 0.0);
        // Corner pixel (0,0) should be outside (positive)
        assert!(sdf.get(0, 0) > 0.0);
    }

    #[test]
    fn test_from_bitmap_edge_near_zero() {
        let bmp = make_square_bitmap(7, 2);
        let sdf = SdfGlyph::from_bitmap(&bmp, 7, 7, 5.0);
        // Pixel at boundary between inside/outside should have small distance
        // (2,2) is inside, (1,2) is outside, boundary is between them
        let inner = sdf.get(2, 2);
        let outer = sdf.get(1, 2);
        assert!(inner < 0.0);
        assert!(outer > 0.0);
        assert!(inner.abs() < 2.0);
        assert!(outer.abs() < 2.0);
    }

    #[test]
    fn test_smoothstep_edges() {
        assert!((smoothstep(0.0, 1.0, -1.0)).abs() < 1e-6);
        assert!((smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < 1e-6);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_render_pixel_inside() {
        let params = RenderParams::default();
        let px = render_sdf_pixel(-5.0, &params);
        assert!((px.fill_alpha - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_render_pixel_outside() {
        let params = RenderParams::default();
        let px = render_sdf_pixel(5.0, &params);
        assert!(px.fill_alpha < 1e-6);
    }

    #[test]
    fn test_render_pixel_edge_antialiased() {
        let params = RenderParams {
            smooth_radius: 1.0,
            ..Default::default()
        };
        let px = render_sdf_pixel(0.0, &params);
        assert!((px.fill_alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_outline_nonzero() {
        let params = RenderParams {
            outline_width: 2.0,
            outline_alpha: 1.0,
            smooth_radius: 0.5,
            ..Default::default()
        };
        // Just outside the fill but inside the outline band
        let px = render_sdf_pixel(1.0, &params);
        assert!(px.outline_alpha > 0.0);
        assert!(px.fill_alpha < 0.5);
    }

    #[test]
    fn test_outline_disabled() {
        let params = RenderParams::default();
        let px = render_sdf_pixel(0.5, &params);
        assert!(px.outline_alpha.abs() < 1e-6);
    }

    #[test]
    fn test_render_full_glyph() {
        let bmp = make_square_bitmap(8, 2);
        let sdf = SdfGlyph::from_bitmap(&bmp, 8, 8, 4.0);
        let params = RenderParams::default();
        let pixels = render_sdf_glyph(&sdf, &params);
        assert_eq!(pixels.len(), 64);
        // Center should be opaque
        assert!(pixels[3 * 8 + 3].fill_alpha > 0.8);
        // Corner should be transparent
        assert!(pixels[0].fill_alpha < 0.2);
    }

    #[test]
    fn test_render_with_shadow() {
        let bmp = make_square_bitmap(8, 2);
        let sdf = SdfGlyph::from_bitmap(&bmp, 8, 8, 4.0);
        let params = RenderParams {
            shadow_offset: (1.0, 1.0),
            shadow_softness: 1.0,
            shadow_alpha: 0.8,
            ..Default::default()
        };
        let pixels = render_sdf_glyph(&sdf, &params);
        // Somewhere near the glyph should have shadow
        let has_shadow = pixels.iter().any(|p| p.shadow_alpha > 0.1);
        assert!(has_shadow);
    }

    #[test]
    fn test_bilinear_sample_integer() {
        let mut g = SdfGlyph::new(3, 3);
        g.set(1, 1, -2.0);
        let v = bilinear_sample(&g, 1.0, 1.0);
        assert!((v - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_sample_interpolated() {
        let mut g = SdfGlyph::new(3, 3);
        g.set(0, 0, 0.0);
        g.set(1, 0, 4.0);
        g.set(0, 1, 0.0);
        g.set(1, 1, 4.0);
        let v = bilinear_sample(&g, 0.5, 0.0);
        assert!((v - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_font_atlas_pack_single() {
        let mut atlas = FontAtlas::new(64, 64, 1);
        let sdf = SdfGlyph::new(10, 12);
        let metrics = GlyphMetrics {
            glyph_id: 65,
            advance_x: 12.0,
            bearing_x: 1.0,
            bearing_y: 10.0,
            size_x: 10.0,
            size_y: 12.0,
        };
        assert!(atlas.pack_glyph(65, &sdf, metrics));
        assert_eq!(atlas.glyph_count(), 1);
        let entry = atlas.get_entry(65).unwrap();
        assert_eq!(entry.width, 10);
        assert_eq!(entry.height, 12);
    }

    #[test]
    fn test_font_atlas_pack_multiple() {
        let mut atlas = FontAtlas::new(128, 128, 1);
        for i in 0..10 {
            let sdf = SdfGlyph::new(10, 10);
            let metrics = GlyphMetrics {
                glyph_id: i,
                advance_x: 10.0,
                bearing_x: 0.0,
                bearing_y: 8.0,
                size_x: 10.0,
                size_y: 10.0,
            };
            assert!(atlas.pack_glyph(i, &sdf, metrics));
        }
        assert_eq!(atlas.glyph_count(), 10);
    }

    #[test]
    fn test_font_atlas_overflow() {
        let mut atlas = FontAtlas::new(16, 16, 0);
        let sdf = SdfGlyph::new(20, 20);
        let m = GlyphMetrics {
            glyph_id: 1,
            advance_x: 20.0,
            bearing_x: 0.0,
            bearing_y: 18.0,
            size_x: 20.0,
            size_y: 20.0,
        };
        assert!(!atlas.pack_glyph(1, &sdf, m));
    }

    #[test]
    fn test_atlas_uv_range() {
        let mut atlas = FontAtlas::new(64, 64, 0);
        let sdf = SdfGlyph::new(16, 16);
        let m = GlyphMetrics {
            glyph_id: 0,
            advance_x: 16.0,
            bearing_x: 0.0,
            bearing_y: 14.0,
            size_x: 16.0,
            size_y: 16.0,
        };
        atlas.pack_glyph(0, &sdf, m);
        let e = atlas.get_entry(0).unwrap();
        assert!(e.u_min >= 0.0 && e.u_max <= 1.0);
        assert!(e.v_min >= 0.0 && e.v_max <= 1.0);
        assert!((e.u_max - 16.0 / 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_atlas_sample() {
        let mut atlas = FontAtlas::new(32, 32, 0);
        let mut sdf = SdfGlyph::new(4, 4);
        sdf.set(1, 1, -3.0);
        let m = GlyphMetrics {
            glyph_id: 0,
            advance_x: 4.0,
            bearing_x: 0.0,
            bearing_y: 3.0,
            size_x: 4.0,
            size_y: 4.0,
        };
        atlas.pack_glyph(0, &sdf, m);
        assert!((atlas.sample(1, 1) - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_composite_alpha_fill_only() {
        let px = RenderedPixel {
            fill_alpha: 0.8,
            outline_alpha: 0.0,
            shadow_alpha: 0.0,
        };
        let a = composite_alpha(&px);
        assert!((a - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_composite_alpha_all_layers() {
        let px = RenderedPixel {
            fill_alpha: 0.5,
            outline_alpha: 0.3,
            shadow_alpha: 0.2,
        };
        let a = composite_alpha(&px);
        // fill over (outline over shadow)
        let os = 0.3 + 0.2 * (1.0 - 0.3);
        let expected = 0.5 + os * (1.0 - 0.5);
        assert!((a - expected).abs() < 1e-6);
    }

    #[test]
    fn test_glyph_metrics_fields() {
        let m = GlyphMetrics {
            glyph_id: 42,
            advance_x: 8.5,
            bearing_x: 1.0,
            bearing_y: 7.0,
            size_x: 7.0,
            size_y: 9.0,
        };
        assert_eq!(m.glyph_id, 42);
        assert!((m.advance_x - 8.5).abs() < 1e-6);
    }

    #[test]
    fn test_atlas_get_metrics() {
        let mut atlas = FontAtlas::new(64, 64, 0);
        let sdf = SdfGlyph::new(8, 8);
        let m = GlyphMetrics {
            glyph_id: 99,
            advance_x: 10.0,
            bearing_x: 1.0,
            bearing_y: 7.0,
            size_x: 8.0,
            size_y: 8.0,
        };
        atlas.pack_glyph(99, &sdf, m.clone());
        let got = atlas.get_metrics(99).unwrap();
        assert_eq!(got.glyph_id, 99);
        assert!((got.advance_x - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sdf_symmetry() {
        // A centered square bitmap should have symmetric SDF
        let bmp = make_square_bitmap(9, 3);
        let sdf = SdfGlyph::from_bitmap(&bmp, 9, 9, 5.0);
        // Left-right symmetry
        assert!((sdf.get(1, 4) - sdf.get(7, 4)).abs() < 1e-4);
        // Top-bottom symmetry
        assert!((sdf.get(4, 1) - sdf.get(4, 7)).abs() < 1e-4);
    }
}
