//! Font metric data model.
//!
//! Provides `FontMetrics` and `GlyphMetrics` types for computing line
//! heights, text widths, and baseline alignment — no font-file parsing,
//! just the numeric model that layout engines consume.

use std::collections::HashMap;

// ── Font-level metrics ────────────────────────────────────────────

/// Top-level metrics for an entire font face.
#[derive(Debug, Clone, PartialEq)]
pub struct FontMetrics {
    /// Design units per em-square (typically 1000 or 2048).
    pub units_per_em: u16,
    /// Typographic ascender (positive, above baseline).
    pub ascender: i16,
    /// Typographic descender (negative, below baseline).
    pub descender: i16,
    /// Extra inter-line gap beyond ascender + |descender|.
    pub line_gap: i16,
    /// Capital letter height.
    pub cap_height: i16,
    /// Lowercase letter height (x-height).
    pub x_height: i16,
    /// Underline position (negative = below baseline).
    pub underline_position: i16,
    /// Underline thickness.
    pub underline_thickness: i16,
}

impl FontMetrics {
    /// Compute the recommended line height in design units.
    ///
    /// `line_height = ascender - descender + line_gap`
    pub fn line_height(&self) -> i32 {
        self.ascender as i32 - self.descender as i32 + self.line_gap as i32
    }

    /// Convert design units to pixels at a given font size.
    pub fn to_pixels(&self, design_units: i32, font_size_px: f64) -> f64 {
        (design_units as f64) * font_size_px / (self.units_per_em as f64)
    }

    /// Line height in pixels at a given font size.
    pub fn line_height_px(&self, font_size_px: f64) -> f64 {
        self.to_pixels(self.line_height(), font_size_px)
    }

    /// Ascender in pixels.
    pub fn ascender_px(&self, font_size_px: f64) -> f64 {
        self.to_pixels(self.ascender as i32, font_size_px)
    }

    /// Descender in pixels (will be negative).
    pub fn descender_px(&self, font_size_px: f64) -> f64 {
        self.to_pixels(self.descender as i32, font_size_px)
    }

    /// Cap height in pixels.
    pub fn cap_height_px(&self, font_size_px: f64) -> f64 {
        self.to_pixels(self.cap_height as i32, font_size_px)
    }

    /// x-height in pixels.
    pub fn x_height_px(&self, font_size_px: f64) -> f64 {
        self.to_pixels(self.x_height as i32, font_size_px)
    }
}

/// A commonly-used set of metrics resembling a standard sans-serif font.
pub fn default_sans_serif() -> FontMetrics {
    FontMetrics {
        units_per_em: 1000,
        ascender: 800,
        descender: -200,
        line_gap: 0,
        cap_height: 700,
        x_height: 500,
        underline_position: -100,
        underline_thickness: 50,
    }
}

// ── Glyph-level metrics ──────────────────────────────────────────

/// Bounding box in design units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BBox {
    pub x_min: i16,
    pub y_min: i16,
    pub x_max: i16,
    pub y_max: i16,
}

impl BBox {
    pub fn width(&self) -> i16 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> i16 {
        self.y_max - self.y_min
    }
}

/// Metrics for a single glyph.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphMetrics {
    /// Advance width in design units.
    pub advance_width: u16,
    /// Left side bearing (distance from origin to left edge of bbox).
    pub lsb: i16,
    /// Glyph bounding box.
    pub bbox: BBox,
}

impl GlyphMetrics {
    /// Right side bearing.
    pub fn rsb(&self) -> i16 {
        self.advance_width as i16 - self.lsb - self.bbox.width()
    }
}

// ── Font instance (metrics + glyph table) ─────────────────────────

/// A kerning table: (left_glyph_id, right_glyph_id) → adjustment.
pub type KerningTable = HashMap<(u32, u32), i16>;

/// A font instance combining face metrics with per-glyph data.
#[derive(Debug, Clone)]
pub struct FontInstance {
    pub metrics: FontMetrics,
    pub glyphs: HashMap<u32, GlyphMetrics>,
    pub kerning: KerningTable,
}

impl FontInstance {
    pub fn new(metrics: FontMetrics) -> Self {
        Self {
            metrics,
            glyphs: HashMap::new(),
            kerning: HashMap::new(),
        }
    }

    /// Add glyph metrics for a glyph ID.
    pub fn add_glyph(&mut self, glyph_id: u32, gm: GlyphMetrics) {
        self.glyphs.insert(glyph_id, gm);
    }

    /// Add a kerning pair.
    pub fn add_kerning(&mut self, left: u32, right: u32, adjustment: i16) {
        self.kerning.insert((left, right), adjustment);
    }

    /// Get kerning adjustment (0 if no pair).
    pub fn get_kerning(&self, left: u32, right: u32) -> i16 {
        self.kerning.get(&(left, right)).copied().unwrap_or(0)
    }

    /// Compute text width in design units from a sequence of glyph IDs.
    pub fn text_width(&self, glyph_ids: &[u32]) -> i32 {
        if glyph_ids.is_empty() {
            return 0;
        }
        let mut width: i32 = 0;
        for (i, &gid) in glyph_ids.iter().enumerate() {
            if let Some(gm) = self.glyphs.get(&gid) {
                width += gm.advance_width as i32;
            }
            // Apply kerning between consecutive glyphs.
            if i + 1 < glyph_ids.len() {
                width += self.get_kerning(gid, glyph_ids[i + 1]) as i32;
            }
        }
        width
    }

    /// Text width in pixels.
    pub fn text_width_px(&self, glyph_ids: &[u32], font_size_px: f64) -> f64 {
        self.metrics.to_pixels(self.text_width(glyph_ids), font_size_px)
    }
}

// ── Baseline alignment ────────────────────────────────────────────

/// Vertical alignment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineAlignment {
    /// Align to the top of the em square (ascender line).
    Top,
    /// Align to the vertical center of cap height.
    Middle,
    /// Align to the alphabetic baseline.
    Baseline,
    /// Align to the bottom of the em square (descender line).
    Bottom,
}

/// Compute the vertical offset (in design units) to align a font at
/// the given mode within a container of the given height (design units).
pub fn alignment_offset(
    metrics: &FontMetrics,
    alignment: BaselineAlignment,
    container_height_du: i32,
) -> i32 {
    let line_h = metrics.line_height();
    match alignment {
        BaselineAlignment::Top => 0,
        BaselineAlignment::Bottom => container_height_du - line_h,
        BaselineAlignment::Baseline => {
            // Place so baseline sits at `ascender` from top.
            // Offset = (container - line_height) / 2 — baseline sits at ascender.
            (container_height_du - line_h) / 2
        }
        BaselineAlignment::Middle => {
            // Center cap-height within container.
            (container_height_du - metrics.cap_height as i32) / 2
                - (metrics.ascender as i32 - metrics.cap_height as i32)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics() -> FontMetrics {
        FontMetrics {
            units_per_em: 1000,
            ascender: 800,
            descender: -200,
            line_gap: 0,
            cap_height: 700,
            x_height: 500,
            underline_position: -100,
            underline_thickness: 50,
        }
    }

    fn sample_glyph() -> GlyphMetrics {
        GlyphMetrics {
            advance_width: 600,
            lsb: 50,
            bbox: BBox {
                x_min: 50,
                y_min: 0,
                x_max: 550,
                y_max: 700,
            },
        }
    }

    #[test]
    fn line_height_calculation() {
        let m = sample_metrics();
        assert_eq!(m.line_height(), 1000); // 800 - (-200) + 0
    }

    #[test]
    fn line_height_with_gap() {
        let mut m = sample_metrics();
        m.line_gap = 200;
        assert_eq!(m.line_height(), 1200);
    }

    #[test]
    fn to_pixels_conversion() {
        let m = sample_metrics();
        let px = m.to_pixels(500, 16.0);
        assert!((px - 8.0).abs() < 1e-6); // 500/1000 * 16
    }

    #[test]
    fn line_height_px() {
        let m = sample_metrics();
        let px = m.line_height_px(16.0);
        assert!((px - 16.0).abs() < 1e-6); // 1000/1000 * 16
    }

    #[test]
    fn glyph_rsb() {
        let g = sample_glyph();
        // rsb = advance_width - lsb - bbox_width = 600 - 50 - 500 = 50
        assert_eq!(g.rsb(), 50);
    }

    #[test]
    fn glyph_bbox_dimensions() {
        let g = sample_glyph();
        assert_eq!(g.bbox.width(), 500);
        assert_eq!(g.bbox.height(), 700);
    }

    #[test]
    fn text_width_no_kerning() {
        let mut fi = FontInstance::new(sample_metrics());
        fi.add_glyph(1, sample_glyph());
        fi.add_glyph(2, sample_glyph());
        let w = fi.text_width(&[1, 2]);
        assert_eq!(w, 1200); // 600 + 600 + 0 kerning
    }

    #[test]
    fn text_width_with_kerning() {
        let mut fi = FontInstance::new(sample_metrics());
        fi.add_glyph(1, sample_glyph());
        fi.add_glyph(2, sample_glyph());
        fi.add_kerning(1, 2, -50);
        let w = fi.text_width(&[1, 2]);
        assert_eq!(w, 1150); // 600 + 600 - 50
    }

    #[test]
    fn text_width_empty() {
        let fi = FontInstance::new(sample_metrics());
        assert_eq!(fi.text_width(&[]), 0);
    }

    #[test]
    fn text_width_px_calculation() {
        let mut fi = FontInstance::new(sample_metrics());
        fi.add_glyph(1, sample_glyph());
        let px = fi.text_width_px(&[1], 16.0);
        assert!((px - 9.6).abs() < 1e-6); // 600/1000 * 16
    }

    #[test]
    fn alignment_top() {
        let m = sample_metrics();
        assert_eq!(alignment_offset(&m, BaselineAlignment::Top, 2000), 0);
    }

    #[test]
    fn alignment_bottom() {
        let m = sample_metrics();
        let off = alignment_offset(&m, BaselineAlignment::Bottom, 2000);
        assert_eq!(off, 1000); // 2000 - 1000
    }

    #[test]
    fn default_sans_serif_sane() {
        let m = default_sans_serif();
        assert!(m.ascender > 0);
        assert!(m.descender < 0);
        assert!(m.line_height() > 0);
        assert!(m.cap_height > m.x_height);
    }
}
