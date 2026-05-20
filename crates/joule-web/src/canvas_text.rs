//! Canvas text rendering: font selection, text measurement (width, ascent, descent),
//! text alignment (start, center, end), baseline options, text wrapping within bounds,
//! text decoration (underline, strikethrough), text path following.
//!
//! Pure math — no browser dependency. Models text layout geometry without actual
//! rasterization.

use std::fmt;

// ── Font ───────────────────────────────────────────────────────

/// Font weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontWeight {
    Thin,       // 100
    Light,      // 300
    Normal,     // 400
    Medium,     // 500
    Bold,       // 700
    Black,      // 900
}

impl FontWeight {
    pub fn numeric(&self) -> u32 {
        match self {
            FontWeight::Thin => 100,
            FontWeight::Light => 300,
            FontWeight::Normal => 400,
            FontWeight::Medium => 500,
            FontWeight::Bold => 700,
            FontWeight::Black => 900,
        }
    }
}

/// Font style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

/// Font specification.
#[derive(Debug, Clone, PartialEq)]
pub struct FontSpec {
    pub family: String,
    pub size: f64,
    pub weight: FontWeight,
    pub style: FontStyle,
    pub line_height: f64,
}

impl FontSpec {
    pub fn new(family: &str, size: f64) -> Self {
        Self {
            family: family.to_string(),
            size,
            weight: FontWeight::Normal,
            style: FontStyle::Normal,
            line_height: size * 1.2,
        }
    }

    pub fn with_weight(mut self, weight: FontWeight) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_style(mut self, style: FontStyle) -> Self {
        self.style = style;
        self
    }

    pub fn with_line_height(mut self, lh: f64) -> Self {
        self.line_height = lh;
        self
    }
}

impl fmt::Display for FontSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let style = match self.style {
            FontStyle::Normal => "",
            FontStyle::Italic => "italic ",
            FontStyle::Oblique => "oblique ",
        };
        write!(f, "{style}{}px {}", self.size, self.family)
    }
}

// ── Text Metrics ───────────────────────────────────────────────

/// Metrics for a measured text string.
#[derive(Debug, Clone, PartialEq)]
pub struct TextMetrics {
    pub width: f64,
    pub ascent: f64,
    pub descent: f64,
    pub line_height: f64,
    pub em_height: f64,
    pub char_count: usize,
}

impl TextMetrics {
    pub fn height(&self) -> f64 {
        self.ascent + self.descent
    }

    pub fn bounding_height(&self) -> f64 {
        self.line_height
    }
}

/// Measure text using simple per-character width estimation.
/// Uses average character width as ~0.6 × font size for proportional fonts,
/// ~0.6 × font size for monospace.
pub fn measure_text(text: &str, font: &FontSpec) -> TextMetrics {
    let avg_char_width = font.size * 0.6;
    let bold_factor = if font.weight.numeric() >= 700 { 1.05 } else { 1.0 };
    let italic_factor = if font.style == FontStyle::Italic { 1.02 } else { 1.0 };

    let width: f64 = text
        .chars()
        .map(|c| char_width_factor(c) * avg_char_width * bold_factor * italic_factor)
        .sum();

    TextMetrics {
        width,
        ascent: font.size * 0.8,
        descent: font.size * 0.2,
        line_height: font.line_height,
        em_height: font.size,
        char_count: text.chars().count(),
    }
}

fn char_width_factor(c: char) -> f64 {
    match c {
        ' ' => 0.5,
        'i' | 'l' | '!' | '|' | '.' | ',' | ':' | ';' | '\'' => 0.4,
        'm' | 'w' | 'M' | 'W' => 1.3,
        'f' | 'j' | 't' => 0.5,
        _ if c.is_uppercase() => 1.1,
        _ => 1.0,
    }
}

// ── Text Alignment ─────────────────────────────────────────────

/// Horizontal text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Start,
    Center,
    End,
    Left,
    Right,
}

/// Text baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextBaseline {
    Top,
    Hanging,
    Middle,
    Alphabetic,
    Ideographic,
    Bottom,
}

/// Compute the x-offset for text given alignment and measured width.
pub fn align_offset(align: TextAlign, text_width: f64) -> f64 {
    match align {
        TextAlign::Start | TextAlign::Left => 0.0,
        TextAlign::Center => -text_width / 2.0,
        TextAlign::End | TextAlign::Right => -text_width,
    }
}

/// Compute the y-offset for text given baseline and metrics.
pub fn baseline_offset(baseline: TextBaseline, metrics: &TextMetrics) -> f64 {
    match baseline {
        TextBaseline::Top => metrics.ascent,
        TextBaseline::Hanging => metrics.ascent * 0.8,
        TextBaseline::Middle => (metrics.ascent - metrics.descent) / 2.0,
        TextBaseline::Alphabetic => 0.0,
        TextBaseline::Ideographic => -metrics.descent * 0.5,
        TextBaseline::Bottom => -metrics.descent,
    }
}

// ── Text Position ──────────────────────────────────────────────

/// A positioned text draw command.
#[derive(Debug, Clone, PartialEq)]
pub struct TextPosition {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub line_index: usize,
}

// ── Text Wrapping ──────────────────────────────────────────────

/// Word-wrap text to fit within `max_width`, returning positioned lines.
pub fn wrap_text(
    text: &str,
    font: &FontSpec,
    max_width: f64,
    x: f64,
    y: f64,
    align: TextAlign,
) -> Vec<TextPosition> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![];
    }

    let space_width = measure_text(" ", font).width;
    let mut lines: Vec<String> = Vec::new();
    let mut current_line = String::new();
    let mut current_width = 0.0;

    for word in &words {
        let wm = measure_text(word, font);
        if current_line.is_empty() {
            current_line = word.to_string();
            current_width = wm.width;
        } else if current_width + space_width + wm.width <= max_width {
            current_line.push(' ');
            current_line.push_str(word);
            current_width += space_width + wm.width;
        } else {
            lines.push(current_line);
            current_line = word.to_string();
            current_width = wm.width;
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines
        .into_iter()
        .enumerate()
        .map(|(i, line)| {
            let lm = measure_text(&line, font);
            let offset = align_offset(align, lm.width);
            TextPosition {
                text: line,
                x: x + offset,
                y: y + i as f64 * font.line_height,
                width: lm.width,
                line_index: i,
            }
        })
        .collect()
}

// ── Text Decoration ────────────────────────────────────────────

/// Text decoration kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecorationKind {
    Underline,
    Overline,
    LineThrough,
}

/// A text decoration line segment.
#[derive(Debug, Clone, PartialEq)]
pub struct DecorationLine {
    pub kind: DecorationKind,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub thickness: f64,
}

/// Compute decoration lines for text at a given position.
pub fn compute_decorations(
    kinds: &[DecorationKind],
    x: f64,
    y: f64,
    metrics: &TextMetrics,
) -> Vec<DecorationLine> {
    let thickness = metrics.em_height * 0.05;
    kinds
        .iter()
        .map(|kind| {
            let dy = match kind {
                DecorationKind::Underline => metrics.descent * 0.5,
                DecorationKind::Overline => -metrics.ascent,
                DecorationKind::LineThrough => -(metrics.ascent - metrics.descent) / 2.0,
            };
            DecorationLine {
                kind: *kind,
                x,
                y: y + dy,
                width: metrics.width,
                thickness,
            }
        })
        .collect()
}

// ── Text Path ──────────────────────────────────────────────────

/// A glyph positioned along a path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathGlyph {
    pub character: char,
    pub x: f64,
    pub y: f64,
    pub angle: f64, // rotation in degrees
}

/// Place text along a path defined by waypoints.
pub fn text_on_path(
    text: &str,
    path: &[(f64, f64)],
    font: &FontSpec,
    offset: f64,
) -> Vec<PathGlyph> {
    if path.len() < 2 {
        return vec![];
    }

    // Compute cumulative arc lengths.
    let mut lengths = vec![0.0f64];
    for i in 1..path.len() {
        let dx = path[i].0 - path[i - 1].0;
        let dy = path[i].1 - path[i - 1].1;
        lengths.push(lengths[i - 1] + (dx * dx + dy * dy).sqrt());
    }
    let total_length = *lengths.last().unwrap();

    let avg_char_width = font.size * 0.6;
    let mut glyphs = Vec::new();
    let mut char_offset = offset;

    for ch in text.chars() {
        let cw = char_width_factor(ch) * avg_char_width;
        let mid = char_offset + cw / 2.0;

        if mid > total_length {
            break;
        }

        // Find segment.
        let seg = lengths
            .windows(2)
            .position(|w| w[0] <= mid && mid <= w[1])
            .unwrap_or(0);

        let seg_start = lengths[seg];
        let seg_len = lengths[seg + 1] - seg_start;
        let t = if seg_len > 0.0 {
            (mid - seg_start) / seg_len
        } else {
            0.0
        };

        let (x0, y0) = path[seg];
        let (x1, y1) = path[seg + 1];
        let px = x0 + (x1 - x0) * t;
        let py = y0 + (y1 - y0) * t;
        let angle = (y1 - y0).atan2(x1 - x0) * 180.0 / std::f64::consts::PI;

        glyphs.push(PathGlyph {
            character: ch,
            x: px,
            y: py,
            angle,
        });

        char_offset += cw;
    }

    glyphs
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_font() -> FontSpec {
        FontSpec::new("sans-serif", 16.0)
    }

    #[test]
    fn test_measure_text_basic() {
        let m = measure_text("hello", &test_font());
        assert!(m.width > 0.0);
        assert!(m.ascent > 0.0);
        assert!(m.descent > 0.0);
        assert_eq!(m.char_count, 5);
    }

    #[test]
    fn test_measure_empty_text() {
        let m = measure_text("", &test_font());
        assert_eq!(m.width, 0.0);
        assert_eq!(m.char_count, 0);
    }

    #[test]
    fn test_bold_wider() {
        let normal = measure_text("test", &FontSpec::new("sans", 16.0));
        let bold = measure_text("test", &FontSpec::new("sans", 16.0).with_weight(FontWeight::Bold));
        assert!(bold.width > normal.width);
    }

    #[test]
    fn test_align_start() {
        assert_eq!(align_offset(TextAlign::Start, 100.0), 0.0);
    }

    #[test]
    fn test_align_center() {
        assert!((align_offset(TextAlign::Center, 100.0) - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_align_end() {
        assert!((align_offset(TextAlign::End, 100.0) - (-100.0)).abs() < 1e-10);
    }

    #[test]
    fn test_baseline_alphabetic() {
        let m = measure_text("x", &test_font());
        assert_eq!(baseline_offset(TextBaseline::Alphabetic, &m), 0.0);
    }

    #[test]
    fn test_wrap_text_single_line() {
        let lines = wrap_text("hello world", &test_font(), 1000.0, 0.0, 0.0, TextAlign::Start);
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_wrap_text_multiple_lines() {
        let lines = wrap_text(
            "the quick brown fox jumps over the lazy dog",
            &test_font(),
            80.0, // narrow width
            0.0,
            0.0,
            TextAlign::Start,
        );
        assert!(lines.len() > 1);
        assert_eq!(lines[0].line_index, 0);
    }

    #[test]
    fn test_decoration_underline() {
        let m = measure_text("test", &test_font());
        let decs = compute_decorations(&[DecorationKind::Underline], 0.0, 0.0, &m);
        assert_eq!(decs.len(), 1);
        assert_eq!(decs[0].kind, DecorationKind::Underline);
        assert!(decs[0].width > 0.0);
        assert!(decs[0].thickness > 0.0);
    }

    #[test]
    fn test_decoration_line_through() {
        let m = measure_text("test", &test_font());
        let decs = compute_decorations(&[DecorationKind::LineThrough], 0.0, 0.0, &m);
        assert_eq!(decs.len(), 1);
        // Line-through should be roughly mid-height.
        assert!(decs[0].y < 0.0); // above baseline
    }

    #[test]
    fn test_text_on_path() {
        let path = vec![(0.0, 0.0), (200.0, 0.0)];
        let glyphs = text_on_path("AB", &path, &test_font(), 0.0);
        assert_eq!(glyphs.len(), 2);
        assert!(glyphs[0].x < glyphs[1].x);
        assert!(glyphs[0].angle.abs() < 1e-10); // horizontal path
    }

    #[test]
    fn test_text_on_curved_path() {
        let path = vec![(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)];
        let glyphs = text_on_path("ABC", &path, &test_font().with_weight(FontWeight::Normal), 0.0);
        assert!(!glyphs.is_empty());
    }

    #[test]
    fn test_font_display() {
        let f = FontSpec::new("Arial", 14.0).with_style(FontStyle::Italic);
        let s = format!("{f}");
        assert!(s.contains("italic"));
        assert!(s.contains("14"));
    }

    #[test]
    fn test_text_metrics_height() {
        let m = measure_text("x", &test_font());
        assert!((m.height() - (m.ascent + m.descent)).abs() < 1e-10);
    }
}
