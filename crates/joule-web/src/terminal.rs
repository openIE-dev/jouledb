//! Terminal abstraction for cursor movement, screen control, and capabilities.
//!
//! Models terminal state including size (columns/rows), cursor position,
//! raw mode, alternate screen buffer, and provides escape sequence generation
//! for cursor movement, screen clearing, and scrolling.

use std::fmt;

// ── Terminal Size ──

/// Terminal dimensions in columns and rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermSize {
    pub cols: u16,
    pub rows: u16,
}

impl TermSize {
    pub fn new(cols: u16, rows: u16) -> Self {
        Self { cols, rows }
    }

    /// Total cell count.
    pub fn cells(&self) -> u32 {
        self.cols as u32 * self.rows as u32
    }
}

impl Default for TermSize {
    fn default() -> Self {
        Self { cols: 80, rows: 24 }
    }
}

// ── Cursor Position ──

/// Zero-based cursor position (col, row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CursorPos {
    pub col: u16,
    pub row: u16,
}

impl CursorPos {
    pub fn new(col: u16, row: u16) -> Self {
        Self { col, row }
    }
}

// ── Terminal Capabilities ──

/// Detected terminal capability flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    pub color_256: bool,
    pub true_color: bool,
    pub unicode: bool,
    pub mouse: bool,
    pub bracketed_paste: bool,
    pub title: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            color_256: true,
            true_color: false,
            unicode: true,
            mouse: false,
            bracketed_paste: false,
            title: true,
        }
    }
}

impl Capabilities {
    /// Detect capabilities from a TERM string (heuristic).
    pub fn from_term(term: &str) -> Self {
        let lower = term.to_ascii_lowercase();
        let true_color = lower.contains("256color") || lower.contains("truecolor") || lower.contains("24bit");
        let color_256 = true_color || lower.contains("256color") || lower.contains("xterm");
        Self {
            color_256,
            true_color,
            unicode: true,
            mouse: lower.contains("xterm") || lower.contains("screen") || lower.contains("tmux"),
            bracketed_paste: lower.contains("xterm") || lower.contains("tmux"),
            title: lower.contains("xterm") || lower.contains("screen"),
        }
    }
}

// ── Escape Sequences ──

const ESC: char = '\x1b';

/// Cursor movement escape sequences.
pub struct Cursor;

impl Cursor {
    /// Move cursor up by `n` rows.
    pub fn up(n: u16) -> String { format!("{ESC}[{n}A") }

    /// Move cursor down by `n` rows.
    pub fn down(n: u16) -> String { format!("{ESC}[{n}B") }

    /// Move cursor right by `n` columns.
    pub fn right(n: u16) -> String { format!("{ESC}[{n}C") }

    /// Move cursor left by `n` columns.
    pub fn left(n: u16) -> String { format!("{ESC}[{n}D") }

    /// Move cursor to home position (1,1).
    pub fn home() -> String { format!("{ESC}[H") }

    /// Move cursor to specific position (1-based row, col).
    pub fn goto(row: u16, col: u16) -> String {
        format!("{ESC}[{row};{col}H")
    }

    /// Save cursor position.
    pub fn save() -> String { format!("{ESC}[s") }

    /// Restore cursor position.
    pub fn restore() -> String { format!("{ESC}[u") }

    /// Hide cursor.
    pub fn hide() -> String { format!("{ESC}[?25l") }

    /// Show cursor.
    pub fn show() -> String { format!("{ESC}[?25h") }
}

/// Screen control escape sequences.
pub struct Screen;

impl Screen {
    /// Clear entire screen.
    pub fn clear() -> String { format!("{ESC}[2J") }

    /// Clear from cursor to end of screen.
    pub fn clear_below() -> String { format!("{ESC}[0J") }

    /// Clear from cursor to start of screen.
    pub fn clear_above() -> String { format!("{ESC}[1J") }

    /// Clear entire line.
    pub fn clear_line() -> String { format!("{ESC}[2K") }

    /// Clear from cursor to end of line.
    pub fn clear_line_right() -> String { format!("{ESC}[0K") }

    /// Clear from cursor to start of line.
    pub fn clear_line_left() -> String { format!("{ESC}[1K") }

    /// Scroll up by `n` lines.
    pub fn scroll_up(n: u16) -> String { format!("{ESC}[{n}S") }

    /// Scroll down by `n` lines.
    pub fn scroll_down(n: u16) -> String { format!("{ESC}[{n}T") }

    /// Enter alternate screen buffer.
    pub fn alt_enter() -> String { format!("{ESC}[?1049h") }

    /// Leave alternate screen buffer.
    pub fn alt_leave() -> String { format!("{ESC}[?1049l") }

    /// Set window title (OSC 2).
    pub fn set_title(title: &str) -> String {
        format!("{ESC}]2;{title}\x07")
    }
}

// ── Terminal State ──

/// Models the state of a terminal session.
#[derive(Debug, Clone)]
pub struct Terminal {
    pub size: TermSize,
    pub cursor: CursorPos,
    pub raw_mode: bool,
    pub alt_screen: bool,
    pub cursor_visible: bool,
    pub capabilities: Capabilities,
    /// Accumulated output buffer for escape sequences.
    output: String,
}

impl Terminal {
    pub fn new(size: TermSize) -> Self {
        Self {
            size,
            cursor: CursorPos::default(),
            raw_mode: false,
            alt_screen: false,
            cursor_visible: true,
            capabilities: Capabilities::default(),
            output: String::new(),
        }
    }

    /// Create with default 80x24.
    pub fn default_size() -> Self {
        Self::new(TermSize::default())
    }

    /// Queue an escape sequence to the output buffer.
    pub fn queue(&mut self, seq: &str) {
        self.output.push_str(seq);
    }

    /// Take and clear the output buffer.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.output)
    }

    /// Move cursor and update internal state.
    pub fn move_to(&mut self, col: u16, row: u16) {
        let col = col.min(self.size.cols.saturating_sub(1));
        let row = row.min(self.size.rows.saturating_sub(1));
        self.cursor = CursorPos::new(col, row);
        // Escape is 1-based.
        self.queue(&Cursor::goto(row + 1, col + 1));
    }

    /// Move cursor up, clamping at row 0.
    pub fn move_up(&mut self, n: u16) {
        let new_row = self.cursor.row.saturating_sub(n);
        self.cursor.row = new_row;
        self.queue(&Cursor::up(n));
    }

    /// Move cursor down, clamping at last row.
    pub fn move_down(&mut self, n: u16) {
        let max = self.size.rows.saturating_sub(1);
        let new_row = (self.cursor.row + n).min(max);
        self.cursor.row = new_row;
        self.queue(&Cursor::down(n));
    }

    /// Enter raw mode.
    pub fn enter_raw(&mut self) {
        self.raw_mode = true;
    }

    /// Leave raw mode.
    pub fn leave_raw(&mut self) {
        self.raw_mode = false;
    }

    /// Enter alternate screen buffer.
    pub fn enter_alt_screen(&mut self) {
        self.alt_screen = true;
        self.queue(&Screen::alt_enter());
    }

    /// Leave alternate screen buffer.
    pub fn leave_alt_screen(&mut self) {
        self.alt_screen = false;
        self.queue(&Screen::alt_leave());
    }

    /// Clear screen and move home.
    pub fn clear(&mut self) {
        self.queue(&Screen::clear());
        self.queue(&Cursor::home());
        self.cursor = CursorPos::default();
    }

    /// Resize the terminal.
    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.size = TermSize::new(cols, rows);
        // Clamp cursor.
        self.cursor.col = self.cursor.col.min(cols.saturating_sub(1));
        self.cursor.row = self.cursor.row.min(rows.saturating_sub(1));
    }
}

impl fmt::Display for Terminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Terminal({}x{} cursor=({},{}) raw={} alt={})",
            self.size.cols, self.size.rows,
            self.cursor.col, self.cursor.row,
            self.raw_mode, self.alt_screen,
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_size_default() {
        let s = TermSize::default();
        assert_eq!(s.cols, 80);
        assert_eq!(s.rows, 24);
        assert_eq!(s.cells(), 1920);
    }

    #[test]
    fn cursor_escape_up() {
        assert_eq!(Cursor::up(3), "\x1b[3A");
    }

    #[test]
    fn cursor_escape_goto() {
        assert_eq!(Cursor::goto(10, 5), "\x1b[10;5H");
    }

    #[test]
    fn cursor_home() {
        assert_eq!(Cursor::home(), "\x1b[H");
    }

    #[test]
    fn screen_clear() {
        assert_eq!(Screen::clear(), "\x1b[2J");
    }

    #[test]
    fn screen_scroll_up() {
        assert_eq!(Screen::scroll_up(2), "\x1b[2S");
    }

    #[test]
    fn alt_screen_sequences() {
        assert_eq!(Screen::alt_enter(), "\x1b[?1049h");
        assert_eq!(Screen::alt_leave(), "\x1b[?1049l");
    }

    #[test]
    fn terminal_move_to() {
        let mut t = Terminal::default_size();
        t.move_to(10, 5);
        assert_eq!(t.cursor, CursorPos::new(10, 5));
        let out = t.flush();
        assert!(out.contains("\x1b[6;11H")); // 1-based
    }

    #[test]
    fn terminal_move_up_clamp() {
        let mut t = Terminal::default_size();
        t.cursor = CursorPos::new(0, 2);
        t.move_up(10);
        assert_eq!(t.cursor.row, 0);
    }

    #[test]
    fn terminal_clear() {
        let mut t = Terminal::default_size();
        t.cursor = CursorPos::new(5, 10);
        t.clear();
        assert_eq!(t.cursor, CursorPos::default());
        let out = t.flush();
        assert!(out.contains("\x1b[2J"));
    }

    #[test]
    fn terminal_raw_mode() {
        let mut t = Terminal::default_size();
        assert!(!t.raw_mode);
        t.enter_raw();
        assert!(t.raw_mode);
        t.leave_raw();
        assert!(!t.raw_mode);
    }

    #[test]
    fn terminal_alt_screen() {
        let mut t = Terminal::default_size();
        t.enter_alt_screen();
        assert!(t.alt_screen);
        let out = t.flush();
        assert!(out.contains("?1049h"));
    }

    #[test]
    fn terminal_resize_clamps_cursor() {
        let mut t = Terminal::default_size();
        t.cursor = CursorPos::new(70, 20);
        t.resize(40, 10);
        assert_eq!(t.cursor.col, 39);
        assert_eq!(t.cursor.row, 9);
    }

    #[test]
    fn capabilities_from_term() {
        let c = Capabilities::from_term("xterm-256color");
        assert!(c.color_256);
        assert!(c.true_color);
        assert!(c.mouse);
    }

    #[test]
    fn set_title_sequence() {
        let s = Screen::set_title("My App");
        assert_eq!(s, "\x1b]2;My App\x07");
    }

    #[test]
    fn terminal_display() {
        let t = Terminal::default_size();
        let s = format!("{t}");
        assert!(s.contains("80x24"));
    }
}
