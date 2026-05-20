//! Terminal emulator state machine: VT100/xterm escape sequence parser.
//!
//! Implements a 2D character grid with cursor tracking, character attributes,
//! scroll regions, tab stops, line wrapping, alternate screen buffer, and a
//! state-machine parser for CSI (Control Sequence Introducer) escape sequences.

use std::fmt;

// ── Character Attributes ──

/// Per-cell text attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharAttr {
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub underline: bool,
    pub blink: bool,
    pub inverse: bool,
    pub strikethrough: bool,
    pub fg_color: u8, // 0 = default, 1-8 = basic, 9+ = 256-color
    pub bg_color: u8,
}

impl Default for CharAttr {
    fn default() -> Self {
        Self {
            bold: false,
            dim: false,
            italic: false,
            underline: false,
            blink: false,
            inverse: false,
            strikethrough: false,
            fg_color: 0,
            bg_color: 0,
        }
    }
}

impl CharAttr {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ── Screen Cell ──

/// A single cell in the screen buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenCell {
    pub ch: char,
    pub attr: CharAttr,
}

impl Default for ScreenCell {
    fn default() -> Self {
        Self { ch: ' ', attr: CharAttr::default() }
    }
}

// ── Screen Buffer ──

/// A 2D grid of cells.
#[derive(Debug, Clone)]
pub struct ScreenBuffer {
    pub width: usize,
    pub height: usize,
    cells: Vec<ScreenCell>,
}

impl ScreenBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![ScreenCell::default(); width * height],
        }
    }

    pub fn get(&self, col: usize, row: usize) -> ScreenCell {
        if col < self.width && row < self.height {
            self.cells[row * self.width + col]
        } else {
            ScreenCell::default()
        }
    }

    pub fn set(&mut self, col: usize, row: usize, cell: ScreenCell) {
        if col < self.width && row < self.height {
            self.cells[row * self.width + col] = cell;
        }
    }

    /// Clear the entire buffer.
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = ScreenCell::default();
        }
    }

    /// Clear a single row.
    pub fn clear_row(&mut self, row: usize) {
        if row < self.height {
            let start = row * self.width;
            for i in start..start + self.width {
                self.cells[i] = ScreenCell::default();
            }
        }
    }

    /// Clear row from column `col` to end.
    pub fn clear_row_from(&mut self, row: usize, col: usize) {
        if row < self.height {
            let start = row * self.width + col.min(self.width);
            let end = row * self.width + self.width;
            for i in start..end {
                self.cells[i] = ScreenCell::default();
            }
        }
    }

    /// Clear row from start to column `col`.
    pub fn clear_row_to(&mut self, row: usize, col: usize) {
        if row < self.height {
            let start = row * self.width;
            let end = start + (col + 1).min(self.width);
            for i in start..end {
                self.cells[i] = ScreenCell::default();
            }
        }
    }

    /// Scroll region [top, bottom) up by one line.
    pub fn scroll_up(&mut self, top: usize, bottom: usize) {
        if top + 1 >= bottom || bottom > self.height { return; }
        for row in top..bottom - 1 {
            let src_start = (row + 1) * self.width;
            let dst_start = row * self.width;
            for i in 0..self.width {
                self.cells[dst_start + i] = self.cells[src_start + i];
            }
        }
        self.clear_row(bottom - 1);
    }

    /// Scroll region [top, bottom) down by one line.
    pub fn scroll_down(&mut self, top: usize, bottom: usize) {
        if top + 1 >= bottom || bottom > self.height { return; }
        for row in (top + 1..bottom).rev() {
            let src_start = (row - 1) * self.width;
            let dst_start = row * self.width;
            for i in 0..self.width {
                self.cells[dst_start + i] = self.cells[src_start + i];
            }
        }
        self.clear_row(top);
    }

    /// Extract text content of a row.
    pub fn row_text(&self, row: usize) -> String {
        if row >= self.height { return String::new(); }
        let start = row * self.width;
        self.cells[start..start + self.width]
            .iter()
            .map(|c| c.ch)
            .collect::<String>()
            .trim_end()
            .to_string()
    }

    /// Resize the buffer, preserving content where possible.
    pub fn resize(&mut self, new_width: usize, new_height: usize) {
        let mut new_cells = vec![ScreenCell::default(); new_width * new_height];
        let copy_w = self.width.min(new_width);
        let copy_h = self.height.min(new_height);
        for row in 0..copy_h {
            for col in 0..copy_w {
                new_cells[row * new_width + col] = self.cells[row * self.width + col];
            }
        }
        self.cells = new_cells;
        self.width = new_width;
        self.height = new_height;
    }
}

// ── Parser State ──

/// State machine states for escape sequence parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserState {
    /// Normal text input.
    Ground,
    /// Received ESC, waiting for next byte.
    Escape,
    /// Inside a CSI sequence (ESC[...).
    CsiParam,
    /// Inside an OSC sequence (ESC]...).
    OscString,
}

// ── Terminal Emulator ──

/// VT100/xterm terminal emulator with screen buffer, cursor, and parser.
#[derive(Debug, Clone)]
pub struct TermEmulator {
    pub primary: ScreenBuffer,
    pub alternate: ScreenBuffer,
    pub using_alternate: bool,
    pub cursor_col: usize,
    pub cursor_row: usize,
    pub current_attr: CharAttr,
    pub scroll_top: usize,
    pub scroll_bottom: usize,
    pub tab_stops: Vec<usize>,
    pub wrap_mode: bool,
    pub cursor_visible: bool,
    pub saved_cursor: Option<(usize, usize, CharAttr)>,
    // Parser state.
    parser_state: ParserState,
    csi_params: String,
    osc_buffer: String,
}

impl TermEmulator {
    pub fn new(width: usize, height: usize) -> Self {
        let mut tab_stops = Vec::new();
        let mut col = 0;
        while col < width {
            tab_stops.push(col);
            col += 8;
        }

        Self {
            primary: ScreenBuffer::new(width, height),
            alternate: ScreenBuffer::new(width, height),
            using_alternate: false,
            cursor_col: 0,
            cursor_row: 0,
            current_attr: CharAttr::default(),
            scroll_top: 0,
            scroll_bottom: height,
            tab_stops,
            wrap_mode: true,
            cursor_visible: true,
            saved_cursor: None,
            parser_state: ParserState::Ground,
            csi_params: String::new(),
            osc_buffer: String::new(),
        }
    }

    /// Active screen buffer.
    pub fn screen(&self) -> &ScreenBuffer {
        if self.using_alternate { &self.alternate } else { &self.primary }
    }

    fn screen_mut(&mut self) -> &mut ScreenBuffer {
        if self.using_alternate { &mut self.alternate } else { &mut self.primary }
    }

    pub fn width(&self) -> usize { self.screen().width }
    pub fn height(&self) -> usize { self.screen().height }

    /// Process a string of input characters.
    pub fn feed(&mut self, input: &str) {
        for ch in input.chars() {
            self.feed_char(ch);
        }
    }

    /// Process a single character through the state machine.
    fn feed_char(&mut self, ch: char) {
        match self.parser_state {
            ParserState::Ground => self.handle_ground(ch),
            ParserState::Escape => self.handle_escape(ch),
            ParserState::CsiParam => self.handle_csi(ch),
            ParserState::OscString => self.handle_osc(ch),
        }
    }

    fn handle_ground(&mut self, ch: char) {
        match ch {
            '\x1b' => {
                self.parser_state = ParserState::Escape;
            }
            '\n' => self.line_feed(),
            '\r' => { self.cursor_col = 0; }
            '\x08' => { // Backspace
                self.cursor_col = self.cursor_col.saturating_sub(1);
            }
            '\t' => self.tab(),
            '\x07' => {} // Bell — ignore
            ch if ch >= ' ' => self.put_char(ch),
            _ => {} // Ignore other control chars.
        }
    }

    fn handle_escape(&mut self, ch: char) {
        match ch {
            '[' => {
                self.parser_state = ParserState::CsiParam;
                self.csi_params.clear();
            }
            ']' => {
                self.parser_state = ParserState::OscString;
                self.osc_buffer.clear();
            }
            '7' => { // Save cursor (DECSC).
                self.saved_cursor = Some((self.cursor_col, self.cursor_row, self.current_attr));
                self.parser_state = ParserState::Ground;
            }
            '8' => { // Restore cursor (DECRC).
                if let Some((col, row, attr)) = self.saved_cursor {
                    self.cursor_col = col;
                    self.cursor_row = row;
                    self.current_attr = attr;
                }
                self.parser_state = ParserState::Ground;
            }
            'D' => { // Index (move down, scroll if needed).
                self.line_feed();
                self.parser_state = ParserState::Ground;
            }
            'M' => { // Reverse index (move up, scroll if needed).
                if self.cursor_row == self.scroll_top {
                    let top = self.scroll_top;
                    let bottom = self.scroll_bottom;
                    self.screen_mut().scroll_down(top, bottom);
                } else {
                    self.cursor_row = self.cursor_row.saturating_sub(1);
                }
                self.parser_state = ParserState::Ground;
            }
            'c' => { // Full reset (RIS).
                let w = self.width();
                let h = self.height();
                *self = Self::new(w, h);
            }
            _ => {
                // Unknown escape — return to ground.
                self.parser_state = ParserState::Ground;
            }
        }
    }

    fn handle_csi(&mut self, ch: char) {
        if ch.is_ascii_digit() || ch == ';' || ch == '?' {
            self.csi_params.push(ch);
        } else {
            // ch is the final byte — dispatch.
            self.dispatch_csi(ch);
            self.parser_state = ParserState::Ground;
        }
    }

    fn handle_osc(&mut self, ch: char) {
        if ch == '\x07' || ch == '\x1b' {
            // End of OSC — we just discard the content for now.
            self.parser_state = if ch == '\x1b' { ParserState::Escape } else { ParserState::Ground };
        } else {
            self.osc_buffer.push(ch);
        }
    }

    /// Parse CSI parameters as a list of numbers.
    fn parse_params(&self) -> Vec<usize> {
        let cleaned = self.csi_params.trim_start_matches('?');
        cleaned.split(';')
            .map(|s| s.parse::<usize>().unwrap_or(0))
            .collect()
    }

    fn dispatch_csi(&mut self, final_ch: char) {
        let params = self.parse_params();
        let p0 = params.first().copied().unwrap_or(0);
        let p1 = params.get(1).copied().unwrap_or(0);

        match final_ch {
            'A' => { // Cursor Up.
                let n = p0.max(1);
                self.cursor_row = self.cursor_row.saturating_sub(n);
            }
            'B' => { // Cursor Down.
                let n = p0.max(1);
                self.cursor_row = (self.cursor_row + n).min(self.height() - 1);
            }
            'C' => { // Cursor Forward.
                let n = p0.max(1);
                self.cursor_col = (self.cursor_col + n).min(self.width() - 1);
            }
            'D' => { // Cursor Back.
                let n = p0.max(1);
                self.cursor_col = self.cursor_col.saturating_sub(n);
            }
            'H' | 'f' => { // Cursor Position (1-based).
                let row = p0.max(1) - 1;
                let col = p1.max(1) - 1;
                self.cursor_row = row.min(self.height() - 1);
                self.cursor_col = col.min(self.width() - 1);
            }
            'J' => { // Erase in Display.
                let row = self.cursor_row;
                let col = self.cursor_col;
                let h = self.height();
                match p0 {
                    0 => { // Below.
                        self.screen_mut().clear_row_from(row, col);
                        for r in row + 1..h {
                            self.screen_mut().clear_row(r);
                        }
                    }
                    1 => { // Above.
                        for r in 0..row {
                            self.screen_mut().clear_row(r);
                        }
                        self.screen_mut().clear_row_to(row, col);
                    }
                    2 | 3 => { // Entire screen.
                        self.screen_mut().clear();
                    }
                    _ => {}
                }
            }
            'K' => { // Erase in Line.
                let row = self.cursor_row;
                let col = self.cursor_col;
                match p0 {
                    0 => self.screen_mut().clear_row_from(row, col),
                    1 => self.screen_mut().clear_row_to(row, col),
                    2 => self.screen_mut().clear_row(row),
                    _ => {}
                }
            }
            'S' => { // Scroll Up.
                let n = p0.max(1);
                let top = self.scroll_top;
                let bottom = self.scroll_bottom;
                for _ in 0..n {
                    self.screen_mut().scroll_up(top, bottom);
                }
            }
            'T' => { // Scroll Down.
                let n = p0.max(1);
                let top = self.scroll_top;
                let bottom = self.scroll_bottom;
                for _ in 0..n {
                    self.screen_mut().scroll_down(top, bottom);
                }
            }
            'm' => { // SGR — Set Graphic Rendition.
                if params.is_empty() || (params.len() == 1 && p0 == 0) {
                    self.current_attr.reset();
                } else {
                    let mut i = 0;
                    while i < params.len() {
                        match params[i] {
                            0 => self.current_attr.reset(),
                            1 => self.current_attr.bold = true,
                            2 => self.current_attr.dim = true,
                            3 => self.current_attr.italic = true,
                            4 => self.current_attr.underline = true,
                            5 => self.current_attr.blink = true,
                            7 => self.current_attr.inverse = true,
                            9 => self.current_attr.strikethrough = true,
                            22 => { self.current_attr.bold = false; self.current_attr.dim = false; }
                            23 => self.current_attr.italic = false,
                            24 => self.current_attr.underline = false,
                            25 => self.current_attr.blink = false,
                            27 => self.current_attr.inverse = false,
                            29 => self.current_attr.strikethrough = false,
                            30..=37 => self.current_attr.fg_color = (params[i] - 29) as u8,
                            39 => self.current_attr.fg_color = 0,
                            40..=47 => self.current_attr.bg_color = (params[i] - 39) as u8,
                            49 => self.current_attr.bg_color = 0,
                            38 => { // Extended fg.
                                if i + 2 < params.len() && params[i + 1] == 5 {
                                    self.current_attr.fg_color = params[i + 2] as u8;
                                    i += 2;
                                }
                            }
                            48 => { // Extended bg.
                                if i + 2 < params.len() && params[i + 1] == 5 {
                                    self.current_attr.bg_color = params[i + 2] as u8;
                                    i += 2;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                }
            }
            'r' => { // Set Scroll Region (DECSTBM).
                let top = p0.max(1) - 1;
                let bottom = if p1 == 0 { self.height() } else { p1.min(self.height()) };
                if top < bottom {
                    self.scroll_top = top;
                    self.scroll_bottom = bottom;
                    self.cursor_col = 0;
                    self.cursor_row = top;
                }
            }
            'h' => { // Set Mode.
                if self.csi_params.starts_with('?') {
                    match p0 {
                        25 => self.cursor_visible = true,
                        1049 => {
                            self.using_alternate = true;
                            self.alternate.clear();
                            self.cursor_col = 0;
                            self.cursor_row = 0;
                        }
                        7 => self.wrap_mode = true,
                        _ => {}
                    }
                }
            }
            'l' => { // Reset Mode.
                if self.csi_params.starts_with('?') {
                    match p0 {
                        25 => self.cursor_visible = false,
                        1049 => {
                            self.using_alternate = false;
                            self.cursor_col = 0;
                            self.cursor_row = 0;
                        }
                        7 => self.wrap_mode = false,
                        _ => {}
                    }
                }
            }
            's' => { // Save cursor position.
                self.saved_cursor = Some((self.cursor_col, self.cursor_row, self.current_attr));
            }
            'u' => { // Restore cursor position.
                if let Some((col, row, attr)) = self.saved_cursor {
                    self.cursor_col = col;
                    self.cursor_row = row;
                    self.current_attr = attr;
                }
            }
            _ => {} // Unknown CSI final.
        }
    }

    /// Put a printable character at the cursor, advancing the cursor.
    fn put_char(&mut self, ch: char) {
        if self.cursor_col >= self.width() {
            if self.wrap_mode {
                self.cursor_col = 0;
                self.line_feed();
            } else {
                self.cursor_col = self.width() - 1;
            }
        }

        let cell = ScreenCell { ch, attr: self.current_attr };
        let col = self.cursor_col;
        let row = self.cursor_row;
        self.screen_mut().set(col, row, cell);
        self.cursor_col += 1;
    }

    /// Line feed: move cursor down or scroll.
    fn line_feed(&mut self) {
        self.cursor_col = 0;
        if self.cursor_row + 1 >= self.scroll_bottom {
            let top = self.scroll_top;
            let bottom = self.scroll_bottom;
            self.screen_mut().scroll_up(top, bottom);
        } else {
            self.cursor_row += 1;
        }
    }

    /// Tab: advance to next tab stop.
    fn tab(&mut self) {
        if let Some(&stop) = self.tab_stops.iter().find(|&&s| s > self.cursor_col) {
            self.cursor_col = stop.min(self.width() - 1);
        } else {
            self.cursor_col = self.width() - 1;
        }
    }

    /// Get the text content of a row.
    pub fn row_text(&self, row: usize) -> String {
        self.screen().row_text(row)
    }

    /// Get the full screen text.
    pub fn screen_text(&self) -> String {
        (0..self.height())
            .map(|r| self.row_text(r))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Resize the emulator.
    pub fn resize(&mut self, width: usize, height: usize) {
        self.primary.resize(width, height);
        self.alternate.resize(width, height);
        self.scroll_bottom = height;
        self.cursor_col = self.cursor_col.min(width.saturating_sub(1));
        self.cursor_row = self.cursor_row.min(height.saturating_sub(1));

        // Rebuild tab stops.
        self.tab_stops.clear();
        let mut col = 0;
        while col < width {
            self.tab_stops.push(col);
            col += 8;
        }
    }
}

impl fmt::Display for TermEmulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.screen_text())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_char_basic() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("Hello");
        assert_eq!(t.row_text(0), "Hello");
        assert_eq!(t.cursor_col, 5);
    }

    #[test]
    fn newline_moves_cursor() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("Line1\nLine2");
        assert_eq!(t.row_text(0), "Line1");
        assert_eq!(t.row_text(1), "Line2");
    }

    #[test]
    fn carriage_return() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("Hello\rWorld");
        assert_eq!(t.row_text(0), "World");
    }

    #[test]
    fn cursor_movement_csi() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("ABCDE");
        t.feed("\x1b[3D"); // Move left 3.
        assert_eq!(t.cursor_col, 2);
        t.feed("\x1b[2C"); // Move right 2.
        assert_eq!(t.cursor_col, 4);
    }

    #[test]
    fn cursor_goto() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("\x1b[5;10H"); // Row 5, Col 10 (1-based).
        assert_eq!(t.cursor_row, 4);
        assert_eq!(t.cursor_col, 9);
    }

    #[test]
    fn clear_screen() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("Hello");
        t.feed("\x1b[2J");
        assert_eq!(t.row_text(0), "");
    }

    #[test]
    fn clear_line() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("Hello World");
        t.feed("\x1b[5D"); // Move left 5.
        t.feed("\x1b[0K"); // Clear from cursor to end.
        assert_eq!(t.row_text(0), "Hello");
    }

    #[test]
    fn sgr_bold() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("\x1b[1m"); // Bold on.
        assert!(t.current_attr.bold);
        t.feed("\x1b[0m"); // Reset.
        assert!(!t.current_attr.bold);
    }

    #[test]
    fn sgr_color() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("\x1b[31m"); // Red fg.
        assert_eq!(t.current_attr.fg_color, 2); // 31 - 29 = 2
        t.feed("X");
        let cell = t.screen().get(0, 0);
        assert_eq!(cell.attr.fg_color, 2);
    }

    #[test]
    fn scroll_region() {
        let mut t = TermEmulator::new(20, 5);
        t.feed("\x1b[2;4r"); // Scroll region rows 2-4 (1-based).
        assert_eq!(t.scroll_top, 1);
        assert_eq!(t.scroll_bottom, 4);
    }

    #[test]
    fn scroll_up() {
        let mut t = TermEmulator::new(10, 3);
        t.feed("AAA\nBBB\nCCC");
        // Now at bottom — next LF should scroll.
        t.feed("\n");
        assert_eq!(t.row_text(0), "BBB");
        assert_eq!(t.row_text(1), "CCC");
    }

    #[test]
    fn alternate_screen() {
        let mut t = TermEmulator::new(20, 5);
        t.feed("Primary");
        t.feed("\x1b[?1049h"); // Enter alt screen.
        assert!(t.using_alternate);
        assert_eq!(t.row_text(0), ""); // Alt screen is clear.
        t.feed("Alt");
        t.feed("\x1b[?1049l"); // Leave alt screen.
        assert!(!t.using_alternate);
        assert_eq!(t.row_text(0), "Primary");
    }

    #[test]
    fn tab_stops() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("\t");
        assert_eq!(t.cursor_col, 8);
        t.feed("\t");
        assert_eq!(t.cursor_col, 16);
    }

    #[test]
    fn line_wrap() {
        let mut t = TermEmulator::new(5, 3);
        t.feed("HelloWorld");
        assert_eq!(t.row_text(0), "Hello");
        assert_eq!(t.row_text(1), "World");
    }

    #[test]
    fn save_restore_cursor() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("\x1b[5;10H"); // Goto (5,10).
        t.feed("\x1b[s");     // Save.
        t.feed("\x1b[1;1H");  // Goto (1,1).
        assert_eq!(t.cursor_row, 0);
        t.feed("\x1b[u");     // Restore.
        assert_eq!(t.cursor_row, 4);
        assert_eq!(t.cursor_col, 9);
    }

    #[test]
    fn hide_show_cursor() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("\x1b[?25l");
        assert!(!t.cursor_visible);
        t.feed("\x1b[?25h");
        assert!(t.cursor_visible);
    }

    #[test]
    fn backspace() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("AB\x08C");
        assert_eq!(t.row_text(0), "AC");
    }

    #[test]
    fn resize_preserves_content() {
        let mut t = TermEmulator::new(10, 3);
        t.feed("Hello");
        t.resize(20, 5);
        assert_eq!(t.row_text(0), "Hello");
        assert_eq!(t.width(), 20);
        assert_eq!(t.height(), 5);
    }

    #[test]
    fn screen_text() {
        let mut t = TermEmulator::new(10, 3);
        t.feed("A\nB\nC");
        let txt = t.screen_text();
        assert!(txt.contains("A"));
        assert!(txt.contains("B"));
        assert!(txt.contains("C"));
    }

    #[test]
    fn display_trait() {
        let mut t = TermEmulator::new(10, 2);
        t.feed("Hi");
        let s = format!("{t}");
        assert!(s.contains("Hi"));
    }

    #[test]
    fn full_reset() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("stuff\x1b[1m");
        t.feed("\x1bc"); // Full reset.
        assert_eq!(t.row_text(0), "");
        assert!(!t.current_attr.bold);
    }

    #[test]
    fn extended_color_256() {
        let mut t = TermEmulator::new(80, 24);
        t.feed("\x1b[38;5;196m"); // FG = 196
        assert_eq!(t.current_attr.fg_color, 196);
    }
}
