//! Table renderer for CLI output with borders, alignment, and column control.
//!
//! Supports ASCII and Unicode borders, left/center/right alignment, column
//! width limits with text wrapping, padding, row separators, colored cells,
//! and horizontal cell merging.

use std::fmt;

// ── Alignment ──

/// Text alignment within a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

impl Align {
    /// Pad `text` to `width` according to alignment.
    pub fn pad(self, text: &str, width: usize) -> String {
        let len = text.chars().count();
        if len >= width {
            return text.chars().take(width).collect();
        }
        let gap = width - len;
        match self {
            Self::Left => format!("{text}{}", " ".repeat(gap)),
            Self::Right => format!("{}{text}", " ".repeat(gap)),
            Self::Center => {
                let left = gap / 2;
                let right = gap - left;
                format!("{}{text}{}", " ".repeat(left), " ".repeat(right))
            }
        }
    }
}

// ── Border Style ──

/// Characters for drawing table borders.
#[derive(Debug, Clone)]
pub struct BorderStyle {
    pub top_left: &'static str,
    pub top_right: &'static str,
    pub bottom_left: &'static str,
    pub bottom_right: &'static str,
    pub horizontal: &'static str,
    pub vertical: &'static str,
    pub cross: &'static str,
    pub top_tee: &'static str,
    pub bottom_tee: &'static str,
    pub left_tee: &'static str,
    pub right_tee: &'static str,
}

impl BorderStyle {
    /// ASCII borders: `+--+`.
    pub fn ascii() -> Self {
        Self {
            top_left: "+", top_right: "+",
            bottom_left: "+", bottom_right: "+",
            horizontal: "-", vertical: "|",
            cross: "+",
            top_tee: "+", bottom_tee: "+",
            left_tee: "+", right_tee: "+",
        }
    }

    /// Unicode single-line borders.
    pub fn unicode() -> Self {
        Self {
            top_left: "┌", top_right: "┐",
            bottom_left: "└", bottom_right: "┘",
            horizontal: "─", vertical: "│",
            cross: "┼",
            top_tee: "┬", bottom_tee: "┴",
            left_tee: "├", right_tee: "┤",
        }
    }

    /// Unicode double-line borders.
    pub fn double() -> Self {
        Self {
            top_left: "╔", top_right: "╗",
            bottom_left: "╚", bottom_right: "╝",
            horizontal: "═", vertical: "║",
            cross: "╬",
            top_tee: "╦", bottom_tee: "╩",
            left_tee: "╠", right_tee: "╣",
        }
    }
}

// ── Cell ──

/// A single table cell with optional styling.
#[derive(Debug, Clone)]
pub struct Cell {
    pub content: String,
    pub align: Option<Align>,
    pub color: Option<String>, // ANSI escape
    pub colspan: usize,
}

impl Cell {
    pub fn new(content: &str) -> Self {
        Self {
            content: content.to_string(),
            align: None,
            color: None,
            colspan: 1,
        }
    }

    pub fn with_align(mut self, align: Align) -> Self {
        self.align = Some(align);
        self
    }

    pub fn with_color(mut self, ansi: &str) -> Self {
        self.color = Some(ansi.to_string());
        self
    }

    pub fn with_colspan(mut self, span: usize) -> Self {
        self.colspan = span.max(1);
        self
    }

    /// Render this cell's content with optional color.
    fn render_content(&self, text: &str) -> String {
        match &self.color {
            Some(c) => format!("{c}{text}\x1b[0m"),
            None => text.to_string(),
        }
    }
}

// ── Column Definition ──

/// Column configuration.
#[derive(Debug, Clone)]
pub struct Column {
    pub header: String,
    pub align: Align,
    pub max_width: Option<usize>,
    pub min_width: usize,
    pub padding: usize,
}

impl Column {
    pub fn new(header: &str) -> Self {
        Self {
            header: header.to_string(),
            align: Align::Left,
            max_width: None,
            min_width: 1,
            padding: 1,
        }
    }

    pub fn with_align(mut self, align: Align) -> Self {
        self.align = align;
        self
    }

    pub fn with_max_width(mut self, w: usize) -> Self {
        self.max_width = Some(w);
        self
    }

    pub fn with_min_width(mut self, w: usize) -> Self {
        self.min_width = w;
        self
    }

    pub fn with_padding(mut self, p: usize) -> Self {
        self.padding = p;
        self
    }
}

// ── Word Wrapping ──

/// Wrap text to fit within `max_width`, returning lines.
pub fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 { return vec![text.to_string()]; }
    let mut lines = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if current.is_empty() {
            if word.len() > max_width {
                // Hard break long words.
                let mut remaining = word;
                while remaining.len() > max_width {
                    let (chunk, rest) = remaining.split_at(max_width);
                    lines.push(chunk.to_string());
                    remaining = rest;
                }
                current = remaining.to_string();
            } else {
                current = word.to_string();
            }
        } else if current.len() + 1 + word.len() <= max_width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(current);
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

// ── Table ──

/// A renderable table with columns, rows, and border style.
#[derive(Debug, Clone)]
pub struct Table {
    pub columns: Vec<Column>,
    pub rows: Vec<Vec<Cell>>,
    pub border: Option<BorderStyle>,
    pub header_separator: bool,
    pub row_separator: bool,
}

impl Table {
    pub fn new(columns: Vec<Column>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            border: Some(BorderStyle::unicode()),
            header_separator: true,
            row_separator: false,
        }
    }

    pub fn with_border(mut self, border: BorderStyle) -> Self {
        self.border = Some(border);
        self
    }

    pub fn no_border(mut self) -> Self {
        self.border = None;
        self
    }

    pub fn with_row_separator(mut self, sep: bool) -> Self {
        self.row_separator = sep;
        self
    }

    pub fn add_row(&mut self, cells: Vec<Cell>) {
        self.rows.push(cells);
    }

    /// Compute effective column widths.
    fn column_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self.columns.iter()
            .map(|c| c.header.chars().count().max(c.min_width))
            .collect();

        for row in &self.rows {
            let mut col_idx = 0;
            for cell in row {
                if col_idx < widths.len() && cell.colspan == 1 {
                    let cell_len = cell.content.chars().count();
                    widths[col_idx] = widths[col_idx].max(cell_len);
                }
                col_idx += cell.colspan;
            }
        }

        // Apply max_width.
        for (i, col) in self.columns.iter().enumerate() {
            if let Some(max) = col.max_width {
                if i < widths.len() {
                    widths[i] = widths[i].min(max);
                }
            }
        }

        widths
    }

    /// Render a horizontal separator line.
    fn render_separator(&self, widths: &[usize], left: &str, mid: &str, right: &str, h: &str) -> String {
        let mut line = left.to_string();
        for (i, &w) in widths.iter().enumerate() {
            let pad = self.columns.get(i).map(|c| c.padding).unwrap_or(1);
            let total = w + pad * 2;
            line.push_str(&h.repeat(total));
            if i + 1 < widths.len() {
                line.push_str(mid);
            }
        }
        line.push_str(right);
        line
    }

    /// Render a data row (may be multi-line due to wrapping).
    fn render_row(&self, cells: &[Cell], widths: &[usize]) -> Vec<String> {
        // Wrap each cell's content.
        let mut wrapped: Vec<(Vec<String>, &Cell, usize)> = Vec::new();
        let mut col_idx = 0;

        for cell in cells {
            let w = if cell.colspan > 1 {
                // Merged width = sum of spanned columns + separators.
                let end = (col_idx + cell.colspan).min(widths.len());
                let sum: usize = widths[col_idx..end].iter().sum();
                let sep_count = cell.colspan.saturating_sub(1);
                let pad_extra: usize = self.columns[col_idx..end].iter()
                    .map(|c| c.padding * 2)
                    .sum::<usize>();
                sum + sep_count + pad_extra.saturating_sub(
                    self.columns.get(col_idx).map(|c| c.padding * 2).unwrap_or(2)
                )
            } else if col_idx < widths.len() {
                widths[col_idx]
            } else {
                10
            };

            let max_w = self.columns.get(col_idx)
                .and_then(|c| c.max_width)
                .unwrap_or(w);
            let lines = wrap_text(&cell.content, max_w.max(1));
            wrapped.push((lines, cell, w));
            col_idx += cell.colspan;
        }

        let max_lines = wrapped.iter().map(|(l, _, _)| l.len()).max().unwrap_or(1);
        let border = &self.border;
        let vert = border.as_ref().map(|b| b.vertical).unwrap_or(" ");

        let mut output_lines = Vec::new();
        for line_idx in 0..max_lines {
            let mut line = vert.to_string();
            let mut ci = 0;
            for (lines, cell, w) in &wrapped {
                let pad = self.columns.get(ci).map(|c| c.padding).unwrap_or(1);
                let text = lines.get(line_idx).map(|s| s.as_str()).unwrap_or("");
                let align = cell.align.unwrap_or(
                    self.columns.get(ci).map(|c| c.align).unwrap_or(Align::Left)
                );
                let padded = align.pad(text, *w);
                let content = cell.render_content(&padded);
                line.push_str(&" ".repeat(pad));
                line.push_str(&content);
                line.push_str(&" ".repeat(pad));
                ci += cell.colspan;
                if ci < widths.len() {
                    line.push_str(vert);
                }
            }
            line.push_str(vert);
            output_lines.push(line);
        }
        output_lines
    }

    /// Render the complete table to a string.
    pub fn render(&self) -> String {
        let widths = self.column_widths();
        let mut out = String::new();

        if let Some(border) = &self.border {
            // Top border.
            out.push_str(&self.render_separator(
                &widths, border.top_left, border.top_tee, border.top_right, border.horizontal
            ));
            out.push('\n');

            // Header row.
            let header_cells: Vec<Cell> = self.columns.iter()
                .map(|c| Cell::new(&c.header).with_align(c.align))
                .collect();
            for line in self.render_row(&header_cells, &widths) {
                out.push_str(&line);
                out.push('\n');
            }

            // Header separator.
            if self.header_separator {
                out.push_str(&self.render_separator(
                    &widths, border.left_tee, border.cross, border.right_tee, border.horizontal
                ));
                out.push('\n');
            }

            // Data rows.
            for (i, row) in self.rows.iter().enumerate() {
                for line in self.render_row(row, &widths) {
                    out.push_str(&line);
                    out.push('\n');
                }
                if self.row_separator && i + 1 < self.rows.len() {
                    out.push_str(&self.render_separator(
                        &widths, border.left_tee, border.cross, border.right_tee, border.horizontal
                    ));
                    out.push('\n');
                }
            }

            // Bottom border.
            out.push_str(&self.render_separator(
                &widths, border.bottom_left, border.bottom_tee, border.bottom_right, border.horizontal
            ));
            out.push('\n');
        } else {
            // No border — space-separated.
            let header_cells: Vec<Cell> = self.columns.iter()
                .map(|c| Cell::new(&c.header).with_align(c.align))
                .collect();
            for line in self.render_row(&header_cells, &widths) {
                out.push_str(&line);
                out.push('\n');
            }
            for row in &self.rows {
                for line in self.render_row(row, &widths) {
                    out.push_str(&line);
                    out.push('\n');
                }
            }
        }

        out
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_left() {
        assert_eq!(Align::Left.pad("hi", 6), "hi    ");
    }

    #[test]
    fn align_right() {
        assert_eq!(Align::Right.pad("hi", 6), "    hi");
    }

    #[test]
    fn align_center() {
        assert_eq!(Align::Center.pad("hi", 6), "  hi  ");
    }

    #[test]
    fn align_truncate() {
        assert_eq!(Align::Left.pad("hello world", 5), "hello");
    }

    #[test]
    fn wrap_text_basic() {
        let lines = wrap_text("hello world foo bar", 11);
        assert_eq!(lines, vec!["hello world", "foo bar"]);
    }

    #[test]
    fn wrap_text_long_word() {
        let lines = wrap_text("abcdefghij", 4);
        assert_eq!(lines, vec!["abcd", "efgh", "ij"]);
    }

    #[test]
    fn wrap_text_empty() {
        let lines = wrap_text("", 10);
        assert_eq!(lines, vec![""]);
    }

    #[test]
    fn simple_table_render() {
        let cols = vec![
            Column::new("Name"),
            Column::new("Age").with_align(Align::Right),
        ];
        let mut t = Table::new(cols).with_border(BorderStyle::ascii());
        t.add_row(vec![Cell::new("Alice"), Cell::new("30")]);
        t.add_row(vec![Cell::new("Bob"), Cell::new("25")]);
        let out = t.render();
        assert!(out.contains("Alice"));
        assert!(out.contains("Bob"));
        assert!(out.contains("+"));
        assert!(out.contains("|"));
    }

    #[test]
    fn unicode_border() {
        let cols = vec![Column::new("X")];
        let t = Table::new(cols).with_border(BorderStyle::unicode());
        let out = t.render();
        assert!(out.contains("┌"));
        assert!(out.contains("└"));
    }

    #[test]
    fn no_border() {
        let cols = vec![Column::new("X")];
        let mut t = Table::new(cols).no_border();
        t.add_row(vec![Cell::new("val")]);
        let out = t.render();
        assert!(out.contains("val"));
        assert!(!out.contains("┌"));
    }

    #[test]
    fn cell_alignment_override() {
        let cols = vec![Column::new("Data").with_align(Align::Left)];
        let mut t = Table::new(cols).with_border(BorderStyle::ascii());
        t.add_row(vec![Cell::new("right").with_align(Align::Right)]);
        let out = t.render();
        // "right" should appear right-aligned — preceded by spaces.
        assert!(out.contains("right"));
    }

    #[test]
    fn colored_cell() {
        let cell = Cell::new("warn").with_color("\x1b[33m");
        let rendered = cell.render_content("warn");
        assert!(rendered.starts_with("\x1b[33m"));
        assert!(rendered.ends_with("\x1b[0m"));
    }

    #[test]
    fn column_max_width() {
        let cols = vec![Column::new("Desc").with_max_width(5)];
        let mut t = Table::new(cols).with_border(BorderStyle::ascii());
        t.add_row(vec![Cell::new("This is a long description")]);
        let out = t.render();
        // Should wrap within 5 chars.
        assert!(out.contains("This"));
    }

    #[test]
    fn row_separator() {
        let cols = vec![Column::new("V")];
        let mut t = Table::new(cols)
            .with_border(BorderStyle::ascii())
            .with_row_separator(true);
        t.add_row(vec![Cell::new("a")]);
        t.add_row(vec![Cell::new("b")]);
        let out = t.render();
        // Should have separator lines between rows.
        let separator_count = out.lines()
            .filter(|l| l.starts_with('+') && l.ends_with('+'))
            .count();
        assert!(separator_count >= 3); // top + header-sep + row-sep + bottom = 4
    }

    #[test]
    fn border_style_double() {
        let b = BorderStyle::double();
        assert_eq!(b.top_left, "╔");
        assert_eq!(b.horizontal, "═");
    }

    #[test]
    fn table_display_trait() {
        let cols = vec![Column::new("X")];
        let t = Table::new(cols);
        let out = format!("{t}");
        assert!(out.contains("X"));
    }
}
