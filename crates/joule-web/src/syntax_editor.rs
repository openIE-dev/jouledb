//! Code editor model.
//!
//! Provides `LineBuffer` for line-oriented text editing with tab handling,
//! auto-indent, and bracket matching. Replaces CodeMirror model with pure Rust.

// ── Line buffer ─────────────────────────────────────────────────

/// A buffer of text lines for code editing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineBuffer {
    /// The lines of text (without newline terminators).
    lines: Vec<String>,
    /// Whether to use tabs (true) or spaces (false) for indentation.
    pub use_tabs: bool,
    /// Number of spaces per indent level (when use_tabs is false).
    pub tab_size: usize,
}

impl LineBuffer {
    /// Create a new empty buffer.
    pub fn new() -> Self {
        Self {
            lines: vec![String::new()],
            use_tabs: false,
            tab_size: 4,
        }
    }

    /// Create a buffer from a string.
    pub fn from_text(text: &str) -> Self {
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        Self {
            lines: if lines.is_empty() {
                vec![String::new()]
            } else {
                lines
            },
            use_tabs: false,
            tab_size: 4,
        }
    }

    /// Get the full text content.
    pub fn to_text(&self) -> String {
        self.lines.join("\n")
    }

    /// Number of lines.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Get a line by index (0-based).
    pub fn line(&self, idx: usize) -> Option<&str> {
        self.lines.get(idx).map(|s| s.as_str())
    }

    /// Get a mutable line by index.
    pub fn line_mut(&mut self, idx: usize) -> Option<&mut String> {
        self.lines.get_mut(idx)
    }

    /// Insert a character at (line, col).
    pub fn insert_char(&mut self, line: usize, col: usize, ch: char) -> bool {
        if let Some(l) = self.lines.get_mut(line) {
            let col = col.min(l.len());
            l.insert(col, ch);
            true
        } else {
            false
        }
    }

    /// Insert a string at (line, col).
    pub fn insert_str(&mut self, line: usize, col: usize, text: &str) -> bool {
        if let Some(l) = self.lines.get_mut(line) {
            let col = col.min(l.len());
            l.insert_str(col, text);
            true
        } else {
            false
        }
    }

    /// Delete a character at (line, col). Returns the deleted character.
    pub fn delete_char(&mut self, line: usize, col: usize) -> Option<char> {
        if let Some(l) = self.lines.get_mut(line) {
            if col < l.len() {
                Some(l.remove(col))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Delete a range of characters on one line.
    pub fn delete_range(&mut self, line: usize, start: usize, end: usize) -> Option<String> {
        if let Some(l) = self.lines.get_mut(line) {
            let start = start.min(l.len());
            let end = end.min(l.len());
            if start >= end {
                return None;
            }
            let removed: String = l.drain(start..end).collect();
            Some(removed)
        } else {
            None
        }
    }

    /// Insert a new line at the given index (pushes existing lines down).
    pub fn insert_line(&mut self, idx: usize, content: &str) {
        let idx = idx.min(self.lines.len());
        self.lines.insert(idx, content.to_string());
    }

    /// Delete a line at the given index.
    pub fn delete_line(&mut self, idx: usize) -> Option<String> {
        if idx < self.lines.len() && self.lines.len() > 1 {
            Some(self.lines.remove(idx))
        } else if idx < self.lines.len() {
            // Last line: clear instead of removing
            let content = std::mem::take(&mut self.lines[idx]);
            Some(content)
        } else {
            None
        }
    }

    /// Join line `idx` with the next line (appends next line to current).
    pub fn join_lines(&mut self, idx: usize) -> bool {
        if idx + 1 < self.lines.len() {
            let next = self.lines.remove(idx + 1);
            self.lines[idx].push_str(&next);
            true
        } else {
            false
        }
    }

    /// Split a line at (line, col), creating a new line below.
    pub fn split_line(&mut self, line: usize, col: usize) -> bool {
        if let Some(l) = self.lines.get_mut(line) {
            let col = col.min(l.len());
            let rest = l[col..].to_string();
            l.truncate(col);
            self.lines.insert(line + 1, rest);
            true
        } else {
            false
        }
    }

    /// Insert a tab at (line, col). Uses spaces or tab char based on config.
    pub fn insert_tab(&mut self, line: usize, col: usize) -> usize {
        if self.use_tabs {
            self.insert_char(line, col, '\t');
            1
        } else {
            let spaces = self.tab_size - (col % self.tab_size);
            let indent: String = " ".repeat(spaces);
            self.insert_str(line, col, &indent);
            spaces
        }
    }

    /// Auto-indent a new line based on the previous line's indentation.
    pub fn auto_indent(&self, prev_line: usize) -> String {
        if let Some(l) = self.lines.get(prev_line) {
            let indent_len = l.len() - l.trim_start().len();
            let indent = &l[..indent_len];

            // Check if prev line ends with an opening bracket
            let trimmed = l.trim_end();
            if trimmed.ends_with('{') || trimmed.ends_with('(') || trimmed.ends_with('[') {
                let extra = if self.use_tabs {
                    "\t".to_string()
                } else {
                    " ".repeat(self.tab_size)
                };
                return format!("{indent}{extra}");
            }

            indent.to_string()
        } else {
            String::new()
        }
    }

    /// Split line at col with auto-indent (enter key behavior).
    pub fn enter(&mut self, line: usize, col: usize) -> bool {
        let indent = self.auto_indent(line);
        if self.split_line(line, col) {
            let next = line + 1;
            if let Some(l) = self.lines.get_mut(next) {
                let content = l.trim_start().to_string();
                *l = format!("{indent}{content}");
            }
            true
        } else {
            false
        }
    }

    // ── Bracket matching ────────────────────────────────────────

    /// Find the matching bracket for the bracket at (line, col).
    /// Returns Some((line, col)) of the matching bracket, or None.
    pub fn find_matching_bracket(&self, line: usize, col: usize) -> Option<(usize, usize)> {
        let ch = self.lines.get(line)?.chars().nth(col)?;
        let (target, direction) = match ch {
            '(' => (')', 1i32),
            ')' => ('(', -1),
            '[' => (']', 1),
            ']' => ('[', -1),
            '{' => ('}', 1),
            '}' => ('{', -1),
            _ => return None,
        };

        let mut depth = 1i32;
        let mut cur_line = line;
        let mut cur_col = col as isize;

        loop {
            cur_col += direction as isize;

            // Handle line wrapping
            loop {
                let line_len = self.lines[cur_line].len() as isize;
                if cur_col >= 0 && cur_col < line_len {
                    break;
                }
                if direction > 0 {
                    cur_line += 1;
                    if cur_line >= self.lines.len() {
                        return None;
                    }
                    cur_col = 0;
                    // If new line is empty, keep advancing lines
                    if self.lines[cur_line].is_empty() {
                        continue;
                    }
                    break;
                } else {
                    if cur_line == 0 {
                        return None;
                    }
                    cur_line -= 1;
                    cur_col = self.lines[cur_line].len() as isize - 1;
                    if cur_col < 0 {
                        // empty line, keep going back
                        continue;
                    }
                    break;
                }
            }

            let line_len = self.lines[cur_line].len() as isize;
            if cur_col < 0 || cur_col >= line_len {
                continue;
            }

            let c = self.lines[cur_line].as_bytes()[cur_col as usize] as char;
            if c == ch {
                depth += 1;
            } else if c == target {
                depth -= 1;
                if depth == 0 {
                    return Some((cur_line, cur_col as usize));
                }
            }
        }
    }

    /// Get line number display text (1-based, right-aligned).
    pub fn line_number(&self, idx: usize, width: usize) -> String {
        format!("{:>width$}", idx + 1, width = width)
    }

    /// Get the width needed for line numbers.
    pub fn line_number_width(&self) -> usize {
        let max = self.lines.len();
        if max == 0 {
            1
        } else {
            ((max as f64).log10().floor() as usize) + 1
        }
    }
}

impl Default for LineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_text_and_back() {
        let text = "line one\nline two\nline three";
        let buf = LineBuffer::from_text(text);
        assert_eq!(buf.line_count(), 3);
        assert_eq!(buf.to_text(), text);
    }

    #[test]
    fn insert_char_basic() {
        let mut buf = LineBuffer::from_text("hello");
        buf.insert_char(0, 5, '!');
        assert_eq!(buf.line(0).unwrap(), "hello!");
    }

    #[test]
    fn insert_str_basic() {
        let mut buf = LineBuffer::from_text("hd");
        buf.insert_str(0, 1, "ello worl");
        assert_eq!(buf.line(0).unwrap(), "hello world");
    }

    #[test]
    fn delete_char_and_range() {
        let mut buf = LineBuffer::from_text("abcdef");
        assert_eq!(buf.delete_char(0, 2), Some('c'));
        assert_eq!(buf.line(0).unwrap(), "abdef");
        assert_eq!(buf.delete_range(0, 1, 3), Some("bd".to_string()));
        assert_eq!(buf.line(0).unwrap(), "aef");
    }

    #[test]
    fn insert_and_delete_line() {
        let mut buf = LineBuffer::from_text("a\nb\nc");
        buf.insert_line(1, "x");
        assert_eq!(buf.line_count(), 4);
        assert_eq!(buf.line(1).unwrap(), "x");
        buf.delete_line(1);
        assert_eq!(buf.line_count(), 3);
        assert_eq!(buf.line(1).unwrap(), "b");
    }

    #[test]
    fn join_lines() {
        let mut buf = LineBuffer::from_text("hello\n world");
        buf.join_lines(0);
        assert_eq!(buf.line_count(), 1);
        assert_eq!(buf.line(0).unwrap(), "hello world");
    }

    #[test]
    fn split_line() {
        let mut buf = LineBuffer::from_text("hello world");
        buf.split_line(0, 5);
        assert_eq!(buf.line_count(), 2);
        assert_eq!(buf.line(0).unwrap(), "hello");
        assert_eq!(buf.line(1).unwrap(), " world");
    }

    #[test]
    fn tab_insert_spaces() {
        let mut buf = LineBuffer::from_text("");
        buf.tab_size = 4;
        let added = buf.insert_tab(0, 0);
        assert_eq!(added, 4);
        assert_eq!(buf.line(0).unwrap(), "    ");
    }

    #[test]
    fn tab_insert_tab_char() {
        let mut buf = LineBuffer::from_text("");
        buf.use_tabs = true;
        let added = buf.insert_tab(0, 0);
        assert_eq!(added, 1);
        assert_eq!(buf.line(0).unwrap(), "\t");
    }

    #[test]
    fn auto_indent_matches_prev() {
        let buf = LineBuffer::from_text("    hello\n");
        let indent = buf.auto_indent(0);
        assert_eq!(indent, "    ");
    }

    #[test]
    fn auto_indent_after_open_brace() {
        let buf = LineBuffer::from_text("fn main() {");
        let indent = buf.auto_indent(0);
        assert_eq!(indent, "    "); // base indent "" + 4 spaces
    }

    #[test]
    fn bracket_matching_parens() {
        let buf = LineBuffer::from_text("(a + (b * c))");
        assert_eq!(buf.find_matching_bracket(0, 0), Some((0, 12)));
        assert_eq!(buf.find_matching_bracket(0, 12), Some((0, 0)));
        assert_eq!(buf.find_matching_bracket(0, 5), Some((0, 11)));
    }

    #[test]
    fn bracket_matching_multiline() {
        let buf = LineBuffer::from_text("{\n  x\n}");
        assert_eq!(buf.find_matching_bracket(0, 0), Some((2, 0)));
        assert_eq!(buf.find_matching_bracket(2, 0), Some((0, 0)));
    }

    #[test]
    fn bracket_no_match() {
        let buf = LineBuffer::from_text("(unclosed");
        assert_eq!(buf.find_matching_bracket(0, 0), None);
    }

    #[test]
    fn line_numbering() {
        let buf = LineBuffer::from_text("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk");
        assert_eq!(buf.line_number_width(), 2);
        assert_eq!(buf.line_number(0, 2), " 1");
        assert_eq!(buf.line_number(9, 2), "10");
    }

    #[test]
    fn enter_key_with_auto_indent() {
        let mut buf = LineBuffer::from_text("    fn foo() {");
        buf.enter(0, 14);
        assert_eq!(buf.line_count(), 2);
        // Should have base indent (4) + extra indent (4) = 8 spaces
        assert_eq!(buf.line(1).unwrap(), "        ");
    }

    #[test]
    fn empty_buffer_has_one_line() {
        let buf = LineBuffer::new();
        assert_eq!(buf.line_count(), 1);
        assert_eq!(buf.line(0), Some(""));
    }
}
