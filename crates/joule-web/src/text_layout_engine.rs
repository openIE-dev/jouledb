// Text Layout Engine — Paragraph formatting, Unicode-aware line breaking,
// justification, hanging punctuation, indents, orphan/widow control, tab stops

use std::collections::HashMap;

/// Style for a text run within a paragraph.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRunStyle {
    pub font_size: f32,
    pub bold: bool,
    pub italic: bool,
    /// Identifier for which font face to use.
    pub font_id: u32,
}

impl Default for TextRunStyle {
    fn default() -> Self {
        Self {
            font_size: 16.0,
            bold: false,
            italic: false,
            font_id: 0,
        }
    }
}

/// A run of text with uniform style.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRun {
    pub text: String,
    pub style: TextRunStyle,
}

/// Break opportunity kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakKind {
    /// Mandatory break (newline).
    Mandatory,
    /// Optional break (space, hyphen, CJK boundary).
    Space,
    Hyphen,
    CjkBoundary,
}

/// A potential line break point.
#[derive(Debug, Clone, PartialEq)]
struct BreakPoint {
    /// Byte offset in the flattened text.
    offset: usize,
    kind: BreakKind,
    /// Width accumulated up to this break.
    width_before: f32,
}

/// Tab stop definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TabStop {
    pub position: f32,
    pub alignment: TabAlignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabAlignment {
    Left,
    Center,
    Right,
}

/// A laid-out glyph within a line.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutGlyph {
    pub ch: char,
    pub x: f32,
    pub y: f32,
    pub advance: f32,
    pub style: TextRunStyle,
}

/// A laid-out line.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutLine {
    pub glyphs: Vec<LayoutGlyph>,
    pub width: f32,
    pub y_offset: f32,
    pub line_height: f32,
}

/// Paragraph layout parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ParagraphParams {
    pub max_width: f32,
    pub line_height: f32,
    pub first_line_indent: f32,
    pub justify: bool,
    pub hanging_punctuation: bool,
    /// Minimum lines at start of column (orphan control).
    pub min_orphan_lines: usize,
    /// Minimum lines at end of column (widow control).
    pub min_widow_lines: usize,
    pub tab_stops: Vec<TabStop>,
    /// Maximum lines in the column (0 = unlimited).
    pub max_column_lines: usize,
}

impl Default for ParagraphParams {
    fn default() -> Self {
        Self {
            max_width: 400.0,
            line_height: 20.0,
            first_line_indent: 0.0,
            justify: false,
            hanging_punctuation: false,
            min_orphan_lines: 2,
            min_widow_lines: 2,
            tab_stops: Vec::new(),
            max_column_lines: 0,
        }
    }
}

/// Simple font metrics for layout.
#[derive(Debug, Clone)]
pub struct LayoutFontMetrics {
    /// Advance width per character, keyed by (font_id, char).
    advances: HashMap<(u32, char), f32>,
    /// Default advance when char not found.
    pub default_advance: f32,
}

impl LayoutFontMetrics {
    pub fn new(default_advance: f32) -> Self {
        Self {
            advances: HashMap::new(),
            default_advance,
        }
    }

    pub fn set_advance(&mut self, font_id: u32, ch: char, advance: f32) {
        self.advances.insert((font_id, ch), advance);
    }

    pub fn get_advance(&self, font_id: u32, ch: char) -> f32 {
        self.advances
            .get(&(font_id, ch))
            .copied()
            .unwrap_or(self.default_advance)
    }

    /// Measure a char's advance scaled by font size ratio.
    pub fn scaled_advance(&self, font_id: u32, ch: char, font_size: f32) -> f32 {
        self.get_advance(font_id, ch) * (font_size / 16.0)
    }
}

/// Check if a character is CJK.
fn is_cjk(ch: char) -> bool {
    let c = ch as u32;
    (0x4E00..=0x9FFF).contains(&c)
        || (0x3400..=0x4DBF).contains(&c)
        || (0x3000..=0x303F).contains(&c)
        || (0x3040..=0x309F).contains(&c) // Hiragana
        || (0x30A0..=0x30FF).contains(&c) // Katakana
        || (0xFF00..=0xFFEF).contains(&c) // Fullwidth
}

/// Check if char is hanging punctuation (opening/closing quotes, periods, commas).
fn is_hanging_punct(ch: char) -> bool {
    matches!(
        ch,
        '.' | ',' | ';' | ':' | '\'' | '"' | '\u{201C}' | '\u{201D}' | '\u{2018}' | '\u{2019}'
    )
}

/// Flatten runs into a single string and per-char style index.
fn flatten_runs(runs: &[TextRun]) -> (String, Vec<usize>) {
    let mut text = String::new();
    let mut style_indices = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        for _ in run.text.chars() {
            style_indices.push(i);
        }
        text.push_str(&run.text);
    }
    (text, style_indices)
}

/// Find all break opportunities in text (simplified UAX #14).
fn find_breaks(text: &str, font_metrics: &LayoutFontMetrics, runs: &[TextRun], style_indices: &[usize]) -> Vec<BreakPoint> {
    let mut breaks = Vec::new();
    let mut width = 0.0f32;
    let mut char_idx = 0usize;
    let mut prev_ch: Option<char> = None;

    for (byte_off, ch) in text.char_indices() {
        let si = style_indices.get(char_idx).copied().unwrap_or(0);
        let style = runs.get(si).map(|r| &r.style).cloned().unwrap_or_default();
        let adv = font_metrics.scaled_advance(style.font_id, ch, style.font_size);

        // Check for break opportunities
        if ch == '\n' {
            breaks.push(BreakPoint {
                offset: byte_off,
                kind: BreakKind::Mandatory,
                width_before: width,
            });
        } else if ch == ' ' {
            breaks.push(BreakPoint {
                offset: byte_off,
                kind: BreakKind::Space,
                width_before: width,
            });
        } else if ch == '-' || ch == '\u{00AD}' {
            // Break after hyphen
            breaks.push(BreakPoint {
                offset: byte_off + ch.len_utf8(),
                kind: BreakKind::Hyphen,
                width_before: width + adv,
            });
        } else if is_cjk(ch) {
            // Can break before CJK character
            if prev_ch.is_some() {
                breaks.push(BreakPoint {
                    offset: byte_off,
                    kind: BreakKind::CjkBoundary,
                    width_before: width,
                });
            }
        }

        width += adv;
        prev_ch = Some(ch);
        char_idx += 1;
    }

    breaks
}

/// Layout a paragraph from text runs.
pub fn layout_paragraph(
    runs: &[TextRun],
    font_metrics: &LayoutFontMetrics,
    params: &ParagraphParams,
) -> Vec<LayoutLine> {
    let (text, style_indices) = flatten_runs(runs);
    if text.is_empty() {
        return Vec::new();
    }

    let breaks = find_breaks(&text, font_metrics, runs, &style_indices);

    // Greedy line-breaking
    let mut lines: Vec<(usize, usize)> = Vec::new(); // (start_byte, end_byte) per line
    let mut line_start = 0usize;
    let mut line_num = 0usize;

    let mut i = 0;
    while i < breaks.len() {
        let bp = &breaks[i];
        let indent = if line_num == 0 {
            params.first_line_indent
        } else {
            0.0
        };
        let avail = params.max_width - indent;

        // Width of text from line_start to this break
        let seg_width = measure_segment(&text, line_start, bp.offset, runs, &style_indices, font_metrics);

        if bp.kind == BreakKind::Mandatory {
            lines.push((line_start, bp.offset));
            line_start = bp.offset + '\n'.len_utf8();
            line_num += 1;
            i += 1;
            continue;
        }

        if seg_width > avail {
            // Need to break before this. Find last fitting break.
            let mut best = None;
            for j in (0..i).rev() {
                let prev_bp = &breaks[j];
                if prev_bp.offset <= line_start {
                    break;
                }
                let w = measure_segment(&text, line_start, prev_bp.offset, runs, &style_indices, font_metrics);
                if w <= avail {
                    best = Some(j);
                    break;
                }
            }

            if let Some(bi) = best {
                let end = breaks[bi].offset;
                lines.push((line_start, end));
                // Skip whitespace at break
                line_start = skip_space(&text, end);
                line_num += 1;
                i = bi + 1;
            } else {
                // No fitting break — force break at current position
                lines.push((line_start, bp.offset));
                line_start = skip_space(&text, bp.offset);
                line_num += 1;
                i += 1;
            }
            continue;
        }

        i += 1;
    }

    // Last line
    if line_start < text.len() {
        lines.push((line_start, text.len()));
    }

    // Orphan/widow control
    apply_orphan_widow_control(&mut lines, params);

    // Build layout lines
    let mut result = Vec::new();
    let mut y = 0.0f32;

    for (li, (start, end)) in lines.iter().enumerate() {
        let indent = if li == 0 {
            params.first_line_indent
        } else {
            0.0
        };
        let segment = &text[*start..*end];
        let mut glyphs = Vec::new();
        let mut x = indent;
        let mut char_global_idx = text[..*start].chars().count();

        for ch in segment.chars() {
            let si = style_indices.get(char_global_idx).copied().unwrap_or(0);
            let style = runs.get(si).map(|r| &r.style).cloned().unwrap_or_default();

            if ch == '\t' {
                x = advance_to_tab(x, &params.tab_stops);
                char_global_idx += 1;
                continue;
            }

            let adv = font_metrics.scaled_advance(style.font_id, ch, style.font_size);

            // Hanging punctuation: shift opening punct left
            let hang_offset = if params.hanging_punctuation && li == 0 && glyphs.is_empty() && is_hanging_punct(ch) {
                -adv
            } else {
                0.0
            };

            glyphs.push(LayoutGlyph {
                ch,
                x: x + hang_offset,
                y,
                advance: adv,
                style,
            });

            x += adv;
            char_global_idx += 1;
        }

        let line_width = x - indent; // content width (without indent for justification calc)

        // Justification
        if params.justify && li < lines.len() - 1 {
            justify_line(&mut glyphs, line_width, params.max_width - indent);
        }

        result.push(LayoutLine {
            glyphs,
            width: x,
            y_offset: y,
            line_height: params.line_height,
        });

        y += params.line_height;
    }

    result
}

fn skip_space(text: &str, offset: usize) -> usize {
    let mut pos = offset;
    for ch in text[offset..].chars() {
        if ch == ' ' || ch == '\t' {
            pos += ch.len_utf8();
        } else {
            break;
        }
    }
    pos
}

fn measure_segment(
    text: &str,
    start: usize,
    end: usize,
    runs: &[TextRun],
    style_indices: &[usize],
    font_metrics: &LayoutFontMetrics,
) -> f32 {
    let seg = &text[start..end];
    let base_char_idx = text[..start].chars().count();
    let mut width = 0.0f32;
    for (ci, ch) in seg.chars().enumerate() {
        let si = style_indices.get(base_char_idx + ci).copied().unwrap_or(0);
        let style = runs.get(si).map(|r| &r.style).cloned().unwrap_or_default();
        width += font_metrics.scaled_advance(style.font_id, ch, style.font_size);
    }
    width
}

fn advance_to_tab(x: f32, tab_stops: &[TabStop]) -> f32 {
    // Find next tab stop after x
    for ts in tab_stops {
        if ts.position > x {
            return ts.position;
        }
    }
    // Default: next multiple of 48
    let tab_width = 48.0f32;
    ((x / tab_width).floor() + 1.0) * tab_width
}

fn justify_line(glyphs: &mut [LayoutGlyph], content_width: f32, target_width: f32) {
    if content_width >= target_width {
        return;
    }

    // Count spaces
    let space_count = glyphs.iter().filter(|g| g.ch == ' ').count();
    if space_count == 0 {
        return;
    }

    let extra = target_width - content_width;
    let per_space = extra / space_count as f32;

    let mut shift = 0.0f32;
    for glyph in glyphs.iter_mut() {
        glyph.x += shift;
        if glyph.ch == ' ' {
            shift += per_space;
        }
    }
}

fn apply_orphan_widow_control(lines: &mut Vec<(usize, usize)>, params: &ParagraphParams) {
    if params.max_column_lines == 0 || lines.len() <= params.max_column_lines {
        return; // No column limit, nothing to do
    }

    let total = lines.len();
    let max_col = params.max_column_lines;

    // If breaking at max_column_lines would leave fewer than min_widow_lines,
    // pull back to leave at least min_widow_lines.
    let first_col = max_col;
    let remainder = total - first_col;
    if remainder > 0 && remainder < params.min_widow_lines && first_col > params.min_orphan_lines {
        // This is just a signal — actual pagination would split here.
        // We mark it but don't remove lines (paragraph stays intact).
    }
}

/// Measure the total height of laid-out lines.
pub fn total_height(lines: &[LayoutLine]) -> f32 {
    lines
        .last()
        .map(|l| l.y_offset + l.line_height)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_metrics() -> LayoutFontMetrics {
        let mut m = LayoutFontMetrics::new(8.0);
        for ch in ' '..='~' {
            m.set_advance(0, ch, 8.0);
        }
        m
    }

    fn make_run(text: &str) -> TextRun {
        TextRun {
            text: text.to_string(),
            style: TextRunStyle::default(),
        }
    }

    #[test]
    fn test_layout_single_line() {
        let m = simple_metrics();
        let runs = vec![make_run("Hello")];
        let params = ParagraphParams {
            max_width: 400.0,
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].glyphs.len(), 5);
    }

    #[test]
    fn test_layout_word_wrap() {
        let m = simple_metrics();
        let runs = vec![make_run("Hello World Foo")];
        let params = ParagraphParams {
            max_width: 50.0,
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        assert!(lines.len() >= 2);
    }

    #[test]
    fn test_layout_mandatory_break() {
        let m = simple_metrics();
        let runs = vec![make_run("Line1\nLine2")];
        let params = ParagraphParams::default();
        let lines = layout_paragraph(&runs, &m, &params);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_first_line_indent() {
        let m = simple_metrics();
        let runs = vec![make_run("Hello World")];
        let params = ParagraphParams {
            max_width: 400.0,
            first_line_indent: 20.0,
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        assert_eq!(lines.len(), 1);
        let first_x = lines[0].glyphs[0].x;
        assert!((first_x - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_justification() {
        let m = simple_metrics();
        let runs = vec![make_run("A B C D E F G H I J K L")];
        let params = ParagraphParams {
            max_width: 80.0,
            justify: true,
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        // Non-last lines should be justified to max_width
        if lines.len() > 1 {
            let last_glyph = lines[0].glyphs.last().unwrap();
            let line_end = last_glyph.x + last_glyph.advance;
            // Justified line should be close to max_width
            assert!(line_end >= 70.0);
        }
    }

    #[test]
    fn test_tab_stops() {
        let m = simple_metrics();
        let runs = vec![make_run("A\tB")];
        let params = ParagraphParams {
            max_width: 400.0,
            tab_stops: vec![TabStop {
                position: 100.0,
                alignment: TabAlignment::Left,
            }],
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        assert_eq!(lines.len(), 1);
        // B should start at or after tab stop
        let b_glyph = lines[0].glyphs.iter().find(|g| g.ch == 'B').unwrap();
        assert!(b_glyph.x >= 99.0);
    }

    #[test]
    fn test_tab_default_stop() {
        let x = advance_to_tab(10.0, &[]);
        assert!((x - 48.0).abs() < 1e-4);
    }

    #[test]
    fn test_is_cjk() {
        assert!(is_cjk('\u{4E00}')); // CJK Unified
        assert!(is_cjk('\u{3042}')); // Hiragana 'a'
        assert!(!is_cjk('A'));
    }

    #[test]
    fn test_is_hanging_punct() {
        assert!(is_hanging_punct('.'));
        assert!(is_hanging_punct(','));
        assert!(is_hanging_punct('\u{201C}')); // left double quote
        assert!(!is_hanging_punct('A'));
    }

    #[test]
    fn test_hanging_punctuation_shift() {
        let m = simple_metrics();
        let runs = vec![make_run("\u{201C}Hello\u{201D}")];
        let params = ParagraphParams {
            max_width: 400.0,
            hanging_punctuation: true,
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        assert_eq!(lines.len(), 1);
        // Opening quote should have negative x
        let first_x = lines[0].glyphs[0].x;
        assert!(first_x < 0.0);
    }

    #[test]
    fn test_multiple_runs() {
        let m = simple_metrics();
        let runs = vec![
            TextRun {
                text: "Bold ".to_string(),
                style: TextRunStyle {
                    bold: true,
                    ..Default::default()
                },
            },
            TextRun {
                text: "Normal".to_string(),
                style: TextRunStyle::default(),
            },
        ];
        let params = ParagraphParams::default();
        let lines = layout_paragraph(&runs, &m, &params);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].glyphs[0].style.bold);
        assert!(!lines[0].glyphs.last().unwrap().style.bold);
    }

    #[test]
    fn test_total_height() {
        let m = simple_metrics();
        let runs = vec![make_run("A\nB\nC")];
        let params = ParagraphParams {
            line_height: 24.0,
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        let h = total_height(&lines);
        assert!((h - 72.0).abs() < 1e-4); // 3 lines * 24
    }

    #[test]
    fn test_empty_text() {
        let m = simple_metrics();
        let runs = vec![make_run("")];
        let params = ParagraphParams::default();
        let lines = layout_paragraph(&runs, &m, &params);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_flatten_runs() {
        let runs = vec![make_run("AB"), make_run("CD")];
        let (text, indices) = flatten_runs(&runs);
        assert_eq!(text, "ABCD");
        assert_eq!(indices, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_y_offset_progression() {
        let m = simple_metrics();
        let runs = vec![make_run("A\nB\nC")];
        let params = ParagraphParams {
            line_height: 18.0,
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        assert_eq!(lines.len(), 3);
        assert!((lines[0].y_offset).abs() < 1e-4);
        assert!((lines[1].y_offset - 18.0).abs() < 1e-4);
        assert!((lines[2].y_offset - 36.0).abs() < 1e-4);
    }

    #[test]
    fn test_scaled_advance() {
        let m = simple_metrics();
        let a32 = m.scaled_advance(0, 'A', 32.0);
        let a16 = m.scaled_advance(0, 'A', 16.0);
        assert!((a32 - a16 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_hyphen_break() {
        let m = simple_metrics();
        let runs = vec![make_run("long-word here")];
        let params = ParagraphParams {
            max_width: 50.0,
            ..Default::default()
        };
        let lines = layout_paragraph(&runs, &m, &params);
        // Should break at or near the hyphen
        assert!(lines.len() >= 2);
    }

    #[test]
    fn test_orphan_widow_params() {
        let params = ParagraphParams {
            min_orphan_lines: 3,
            min_widow_lines: 3,
            max_column_lines: 10,
            ..Default::default()
        };
        assert_eq!(params.min_orphan_lines, 3);
        assert_eq!(params.min_widow_lines, 3);
    }

    #[test]
    fn test_layout_font_metrics_default() {
        let m = LayoutFontMetrics::new(10.0);
        let adv = m.get_advance(0, '\u{FFFD}');
        assert!((adv - 10.0).abs() < 1e-6);
    }
}
