// MSDF Text — Multi-channel Signed Distance Field text rendering
// Three SDF channels (R, G, B) with median filter for sharp corners,
// text rendering, kerning, alignment, word wrapping, rich text spans

use std::collections::HashMap;

/// Three-channel SDF pixel (R, G, B distances).
#[derive(Debug, Clone, PartialEq)]
pub struct MsdfPixel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl MsdfPixel {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Median of three channels — the key MSDF operation.
    pub fn median(&self) -> f32 {
        median3(self.r, self.g, self.b)
    }
}

/// Median of three values.
fn median3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b.min(c)).min(b.max(c))
}

/// A multi-channel SDF glyph (3-channel distance field).
#[derive(Debug, Clone, PartialEq)]
pub struct MsdfGlyph {
    pub width: usize,
    pub height: usize,
    /// Row-major MSDF pixels.
    pub pixels: Vec<MsdfPixel>,
}

impl MsdfGlyph {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![
                MsdfPixel {
                    r: f32::MAX,
                    g: f32::MAX,
                    b: f32::MAX
                };
                width * height
            ],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &MsdfPixel {
        &self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, pixel: MsdfPixel) {
        self.pixels[y * self.width + x] = pixel;
    }

    /// Evaluate edge coverage at a pixel using median filter.
    pub fn coverage(&self, x: usize, y: usize, smooth_radius: f32) -> f32 {
        let p = self.get(x, y);
        let d = p.median();
        let edge = 0.0;
        1.0 - smoothstep_f32(edge - smooth_radius, edge + smooth_radius, d)
    }
}

fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Text alignment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

/// Glyph metrics for a single character.
#[derive(Debug, Clone, PartialEq)]
pub struct CharMetrics {
    pub glyph_id: u32,
    pub advance: f32,
    pub bearing_x: f32,
    pub bearing_y: f32,
    pub width: f32,
    pub height: f32,
}

/// Kerning pair adjustment.
#[derive(Debug, Clone, PartialEq)]
pub struct KerningPair {
    pub left: char,
    pub right: char,
    pub amount: f32,
}

/// A positioned glyph quad for rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphQuad {
    pub glyph_id: u32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Color from rich text span.
    pub color: [f32; 4],
    /// Font size for this glyph.
    pub font_size: f32,
}

/// Rich text span: style applied to a range of characters.
#[derive(Debug, Clone, PartialEq)]
pub struct TextSpan {
    /// Byte offset start (inclusive).
    pub start: usize,
    /// Byte offset end (exclusive).
    pub end: usize,
    pub color: [f32; 4],
    pub font_size: f32,
}

/// Font data needed for text layout.
#[derive(Debug, Clone)]
pub struct FontData {
    pub line_height: f32,
    pub ascent: f32,
    pub descent: f32,
    pub default_size: f32,
    pub char_metrics: HashMap<char, CharMetrics>,
    pub kerning: Vec<KerningPair>,
}

impl FontData {
    pub fn new(line_height: f32, ascent: f32, descent: f32, default_size: f32) -> Self {
        Self {
            line_height,
            ascent,
            descent,
            default_size,
            char_metrics: HashMap::new(),
            kerning: Vec::new(),
        }
    }

    pub fn add_char(&mut self, ch: char, metrics: CharMetrics) {
        self.char_metrics.insert(ch, metrics);
    }

    pub fn add_kerning(&mut self, left: char, right: char, amount: f32) {
        self.kerning.push(KerningPair {
            left,
            right,
            amount,
        });
    }

    pub fn get_kerning(&self, left: char, right: char) -> f32 {
        for kp in &self.kerning {
            if kp.left == left && kp.right == right {
                return kp.amount;
            }
        }
        0.0
    }

    pub fn get_advance(&self, ch: char) -> f32 {
        self.char_metrics
            .get(&ch)
            .map(|m| m.advance)
            .unwrap_or(self.default_size * 0.5)
    }
}

/// A line of positioned glyphs.
#[derive(Debug, Clone, PartialEq)]
pub struct TextLine {
    pub quads: Vec<GlyphQuad>,
    pub width: f32,
    pub y_offset: f32,
}

/// Word-wrap and lay out text with optional rich text spans.
pub fn layout_text(
    text: &str,
    font: &FontData,
    max_width: f32,
    alignment: TextAlign,
    spans: &[TextSpan],
) -> Vec<TextLine> {
    let default_color = [1.0, 1.0, 1.0, 1.0];
    let default_size = font.default_size;

    // Break text into words
    let words = split_words(text);
    let mut lines: Vec<TextLine> = Vec::new();
    let mut current_line_words: Vec<(usize, usize)> = Vec::new(); // (start, end) byte offsets
    let mut current_width: f32 = 0.0;
    let mut byte_pos = 0usize;

    for word in &words {
        let word_start = byte_pos;
        let word_end = byte_pos + word.len();
        let word_width = measure_word(word, font, word_start, spans, default_size);
        let space_width = font.get_advance(' ') * (default_size / font.default_size);

        let needed = if current_line_words.is_empty() {
            word_width
        } else {
            space_width + word_width
        };

        if !current_line_words.is_empty() && current_width + needed > max_width {
            // Flush current line
            let line =
                build_line(text, &current_line_words, font, spans, default_color, default_size);
            lines.push(line);
            current_line_words.clear();
            current_width = 0.0;
        }

        if current_line_words.is_empty() {
            current_width = word_width;
        } else {
            current_width += space_width + word_width;
        }
        current_line_words.push((word_start, word_end));
        byte_pos = word_end;
        // Skip whitespace between words
        let rest = &text[byte_pos..];
        for ch in rest.chars() {
            if ch == ' ' || ch == '\t' {
                byte_pos += ch.len_utf8();
            } else {
                break;
            }
        }
    }

    // Final line
    if !current_line_words.is_empty() {
        let line = build_line(text, &current_line_words, font, spans, default_color, default_size);
        lines.push(line);
    }

    // Apply alignment and y offsets
    let mut y = 0.0f32;
    for line in &mut lines {
        line.y_offset = y;
        y += font.line_height;

        let shift = match alignment {
            TextAlign::Left => 0.0,
            TextAlign::Center => (max_width - line.width) / 2.0,
            TextAlign::Right => max_width - line.width,
        };

        for quad in &mut line.quads {
            quad.x += shift;
        }
    }

    lines
}

fn split_words(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut start = None;
    for (i, ch) in text.char_indices() {
        if ch == ' ' || ch == '\t' {
            if let Some(s) = start {
                words.push(&text[s..i]);
                start = None;
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        words.push(&text[s..]);
    }
    words
}

fn measure_word(
    word: &str,
    font: &FontData,
    _byte_offset: usize,
    spans: &[TextSpan],
    default_size: f32,
) -> f32 {
    let mut width = 0.0f32;
    let mut prev_ch: Option<char> = None;
    for ch in word.chars() {
        let size = span_size_at(spans, 0, default_size); // simplified
        let scale = size / font.default_size;
        if let Some(pc) = prev_ch {
            width += font.get_kerning(pc, ch) * scale;
        }
        width += font.get_advance(ch) * scale;
        prev_ch = Some(ch);
    }
    width
}

fn span_size_at(spans: &[TextSpan], byte_pos: usize, default_size: f32) -> f32 {
    for span in spans {
        if byte_pos >= span.start && byte_pos < span.end {
            return span.font_size;
        }
    }
    default_size
}

fn span_color_at(spans: &[TextSpan], byte_pos: usize, default_color: [f32; 4]) -> [f32; 4] {
    for span in spans {
        if byte_pos >= span.start && byte_pos < span.end {
            return span.color;
        }
    }
    default_color
}

fn build_line(
    text: &str,
    word_ranges: &[(usize, usize)],
    font: &FontData,
    spans: &[TextSpan],
    default_color: [f32; 4],
    default_size: f32,
) -> TextLine {
    let mut quads = Vec::new();
    let mut x = 0.0f32;
    let mut prev_ch: Option<char> = None;

    for (wi, (ws, we)) in word_ranges.iter().enumerate() {
        if wi > 0 {
            // Space between words
            let scale = default_size / font.default_size;
            x += font.get_advance(' ') * scale;
            prev_ch = Some(' ');
        }

        let word = &text[*ws..*we];
        let mut byte_pos = *ws;
        for ch in word.chars() {
            let size = span_size_at(spans, byte_pos, default_size);
            let color = span_color_at(spans, byte_pos, default_color);
            let scale = size / font.default_size;

            if let Some(pc) = prev_ch {
                x += font.get_kerning(pc, ch) * scale;
            }

            let cm = font.char_metrics.get(&ch);
            let (w, h) = cm.map(|m| (m.width * scale, m.height * scale)).unwrap_or((size * 0.5, size));
            let glyph_id = cm.map(|m| m.glyph_id).unwrap_or(0);
            let bearing_x = cm.map(|m| m.bearing_x * scale).unwrap_or(0.0);
            let bearing_y = cm.map(|m| m.bearing_y * scale).unwrap_or(size * 0.8);

            quads.push(GlyphQuad {
                glyph_id,
                x: x + bearing_x,
                y: -bearing_y,
                width: w,
                height: h,
                color,
                font_size: size,
            });

            x += font.get_advance(ch) * scale;
            prev_ch = Some(ch);
            byte_pos += ch.len_utf8();
        }
    }

    TextLine {
        width: x,
        quads,
        y_offset: 0.0,
    }
}

/// Measure text width without full layout.
pub fn measure_text_width(text: &str, font: &FontData) -> f32 {
    let mut width = 0.0f32;
    let mut prev: Option<char> = None;
    for ch in text.chars() {
        if let Some(pc) = prev {
            width += font.get_kerning(pc, ch);
        }
        width += font.get_advance(ch);
        prev = Some(ch);
    }
    width
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_font() -> FontData {
        let mut f = FontData::new(20.0, 16.0, -4.0, 16.0);
        for ch in 'A'..='Z' {
            let id = ch as u32;
            f.add_char(
                ch,
                CharMetrics {
                    glyph_id: id,
                    advance: 10.0,
                    bearing_x: 1.0,
                    bearing_y: 12.0,
                    width: 8.0,
                    height: 14.0,
                },
            );
        }
        for ch in 'a'..='z' {
            let id = ch as u32;
            f.add_char(
                ch,
                CharMetrics {
                    glyph_id: id,
                    advance: 8.0,
                    bearing_x: 0.5,
                    bearing_y: 9.0,
                    width: 7.0,
                    height: 10.0,
                },
            );
        }
        f.add_char(
            ' ',
            CharMetrics {
                glyph_id: 32,
                advance: 5.0,
                bearing_x: 0.0,
                bearing_y: 0.0,
                width: 0.0,
                height: 0.0,
            },
        );
        f.add_kerning('A', 'V', -1.5);
        f.add_kerning('T', 'o', -1.0);
        f
    }

    #[test]
    fn test_median3_basic() {
        assert!((median3(1.0, 2.0, 3.0) - 2.0).abs() < 1e-6);
        assert!((median3(3.0, 1.0, 2.0) - 2.0).abs() < 1e-6);
        assert!((median3(2.0, 3.0, 1.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_median3_equal() {
        assert!((median3(5.0, 5.0, 5.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_msdf_pixel_median() {
        let p = MsdfPixel::new(1.0, 3.0, 2.0);
        assert!((p.median() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_msdf_glyph_new() {
        let g = MsdfGlyph::new(4, 4);
        assert_eq!(g.pixels.len(), 16);
    }

    #[test]
    fn test_msdf_glyph_set_get() {
        let mut g = MsdfGlyph::new(4, 4);
        g.set(1, 2, MsdfPixel::new(-1.0, -2.0, -1.5));
        let p = g.get(1, 2);
        assert!((p.r - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_coverage_inside() {
        let mut g = MsdfGlyph::new(3, 3);
        g.set(1, 1, MsdfPixel::new(-5.0, -5.0, -5.0));
        let c = g.coverage(1, 1, 0.5);
        assert!((c - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_coverage_outside() {
        let mut g = MsdfGlyph::new(3, 3);
        g.set(1, 1, MsdfPixel::new(5.0, 5.0, 5.0));
        let c = g.coverage(1, 1, 0.5);
        assert!(c < 1e-4);
    }

    #[test]
    fn test_font_data_kerning() {
        let f = test_font();
        assert!((f.get_kerning('A', 'V') - (-1.5)).abs() < 1e-6);
        assert!((f.get_kerning('A', 'B')).abs() < 1e-6);
    }

    #[test]
    fn test_font_data_advance() {
        let f = test_font();
        assert!((f.get_advance('A') - 10.0).abs() < 1e-6);
        assert!((f.get_advance('a') - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_measure_text_width() {
        let f = test_font();
        let w = measure_text_width("AB", &f);
        assert!((w - 20.0).abs() < 1e-6); // 10 + 10
    }

    #[test]
    fn test_measure_text_width_with_kerning() {
        let f = test_font();
        let w = measure_text_width("AV", &f);
        // 10 + (-1.5) + 10 = 18.5
        assert!((w - 18.5).abs() < 1e-6);
    }

    #[test]
    fn test_layout_single_line() {
        let f = test_font();
        let lines = layout_text("Hello", &f, 500.0, TextAlign::Left, &[]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].quads.len(), 5);
    }

    #[test]
    fn test_layout_word_wrap() {
        let f = test_font();
        // Each character is 8-10px advance. "Hello World" ~ 85px.
        // max_width=50 should cause wrap.
        let lines = layout_text("Hello World", &f, 50.0, TextAlign::Left, &[]);
        assert!(lines.len() >= 2);
    }

    #[test]
    fn test_layout_center_alignment() {
        let f = test_font();
        let lines = layout_text("Hi", &f, 200.0, TextAlign::Center, &[]);
        assert_eq!(lines.len(), 1);
        let first_x = lines[0].quads[0].x;
        // Should be shifted right for centering
        assert!(first_x > 10.0);
    }

    #[test]
    fn test_layout_right_alignment() {
        let f = test_font();
        let lines = layout_text("Hi", &f, 200.0, TextAlign::Right, &[]);
        assert_eq!(lines.len(), 1);
        let first_x = lines[0].quads[0].x;
        // Should be shifted far right
        assert!(first_x > 100.0);
    }

    #[test]
    fn test_layout_y_offsets() {
        let f = test_font();
        let lines = layout_text("A B C D E F G", &f, 30.0, TextAlign::Left, &[]);
        if lines.len() >= 2 {
            assert!((lines[0].y_offset).abs() < 1e-6);
            assert!((lines[1].y_offset - 20.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_layout_empty_text() {
        let f = test_font();
        let lines = layout_text("", &f, 100.0, TextAlign::Left, &[]);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_layout_rich_text_span() {
        let f = test_font();
        let spans = vec![TextSpan {
            start: 0,
            end: 5,
            color: [1.0, 0.0, 0.0, 1.0],
            font_size: 16.0,
        }];
        let lines = layout_text("Hello World", &f, 500.0, TextAlign::Left, &spans);
        assert_eq!(lines.len(), 1);
        // First 5 chars should have red color
        assert!((lines[0].quads[0].color[0] - 1.0).abs() < 1e-6);
        assert!((lines[0].quads[0].color[1]).abs() < 1e-6);
    }

    #[test]
    fn test_split_words() {
        let words = split_words("hello world foo");
        assert_eq!(words, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_split_words_single() {
        let words = split_words("hello");
        assert_eq!(words, vec!["hello"]);
    }

    #[test]
    fn test_split_words_empty() {
        let words = split_words("");
        assert!(words.is_empty());
    }

    #[test]
    fn test_text_align_enum() {
        assert_ne!(TextAlign::Left, TextAlign::Center);
        assert_ne!(TextAlign::Center, TextAlign::Right);
    }

    #[test]
    fn test_glyph_quad_fields() {
        let q = GlyphQuad {
            glyph_id: 65,
            x: 10.0,
            y: -12.0,
            width: 8.0,
            height: 14.0,
            color: [1.0; 4],
            font_size: 16.0,
        };
        assert_eq!(q.glyph_id, 65);
        assert!((q.x - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_msdf_coverage_at_edge() {
        let mut g = MsdfGlyph::new(3, 3);
        g.set(1, 1, MsdfPixel::new(0.0, 0.0, 0.0));
        let c = g.coverage(1, 1, 1.0);
        assert!((c - 0.5).abs() < 1e-4);
    }
}
