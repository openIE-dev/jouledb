//! ANSI escape code library for terminal color and text styling.
//!
//! Supports 16 basic colors, 256-color palette, RGB true color, and text
//! attributes (bold, dim, italic, underline, strikethrough, blink). Provides
//! a builder API for composing styled strings and a strip function for
//! removing ANSI sequences from output.

use std::fmt;

// ── Basic Colors ──

/// The 16 standard ANSI colors (8 normal + 8 bright).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasicColor {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

impl BasicColor {
    /// SGR foreground code for this color.
    pub fn fg_code(self) -> u8 {
        match self {
            Self::Black => 30,
            Self::Red => 31,
            Self::Green => 32,
            Self::Yellow => 33,
            Self::Blue => 34,
            Self::Magenta => 35,
            Self::Cyan => 36,
            Self::White => 37,
            Self::BrightBlack => 90,
            Self::BrightRed => 91,
            Self::BrightGreen => 92,
            Self::BrightYellow => 93,
            Self::BrightBlue => 94,
            Self::BrightMagenta => 95,
            Self::BrightCyan => 96,
            Self::BrightWhite => 97,
        }
    }

    /// SGR background code for this color.
    pub fn bg_code(self) -> u8 {
        self.fg_code() + 10
    }
}

// ── Color Specification ──

/// A color that can be basic (16), indexed (256), or true-color RGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Basic(BasicColor),
    Indexed(u8),
    Rgb(u8, u8, u8),
}

impl Color {
    /// Produce the SGR foreground escape for this color (without ESC[ prefix or trailing m).
    pub fn fg_params(&self) -> String {
        match self {
            Self::Basic(c) => format!("{}", c.fg_code()),
            Self::Indexed(i) => format!("38;5;{i}"),
            Self::Rgb(r, g, b) => format!("38;2;{r};{g};{b}"),
        }
    }

    /// Produce the SGR background escape parameters.
    pub fn bg_params(&self) -> String {
        match self {
            Self::Basic(c) => format!("{}", c.bg_code()),
            Self::Indexed(i) => format!("48;5;{i}"),
            Self::Rgb(r, g, b) => format!("48;2;{r};{g};{b}"),
        }
    }
}

// ── Text Attributes ──

/// Text attribute flags stored as a bitmask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Attributes(u8);

impl Attributes {
    pub const BOLD: u8 = 1;
    pub const DIM: u8 = 2;
    pub const ITALIC: u8 = 4;
    pub const UNDERLINE: u8 = 8;
    pub const BLINK: u8 = 16;
    pub const STRIKETHROUGH: u8 = 32;

    pub fn new() -> Self {
        Self(0)
    }

    pub fn set(self, flag: u8) -> Self {
        Self(self.0 | flag)
    }

    pub fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    /// Return the SGR codes for all active attributes.
    pub fn sgr_codes(self) -> Vec<u8> {
        let mut codes = Vec::new();
        if self.has(Self::BOLD) { codes.push(1); }
        if self.has(Self::DIM) { codes.push(2); }
        if self.has(Self::ITALIC) { codes.push(3); }
        if self.has(Self::UNDERLINE) { codes.push(4); }
        if self.has(Self::BLINK) { codes.push(5); }
        if self.has(Self::STRIKETHROUGH) { codes.push(9); }
        codes
    }
}

// ── Escape Sequences ──

/// The ESC character.
pub const ESC: char = '\x1b';

/// Reset all attributes.
pub const RESET: &str = "\x1b[0m";

/// Produce a foreground color escape sequence.
pub fn fg(color: Color) -> String {
    format!("{ESC}[{}m", color.fg_params())
}

/// Produce a background color escape sequence.
pub fn bg(color: Color) -> String {
    format!("{ESC}[{}m", color.bg_params())
}

/// Produce an attribute escape sequence.
pub fn attr(code: u8) -> String {
    format!("{ESC}[{code}m")
}

// ── StyledString Builder ──

/// Builder for composing an ANSI-styled string.
#[derive(Debug, Clone)]
pub struct StyledString {
    text: String,
    fg: Option<Color>,
    bg_color: Option<Color>,
    attrs: Attributes,
}

impl StyledString {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            fg: None,
            bg_color: None,
            attrs: Attributes::new(),
        }
    }

    pub fn fg(mut self, color: Color) -> Self {
        self.fg = Some(color);
        self
    }

    pub fn bg(mut self, color: Color) -> Self {
        self.bg_color = Some(color);
        self
    }

    pub fn bold(mut self) -> Self {
        self.attrs = self.attrs.set(Attributes::BOLD);
        self
    }

    pub fn dim(mut self) -> Self {
        self.attrs = self.attrs.set(Attributes::DIM);
        self
    }

    pub fn italic(mut self) -> Self {
        self.attrs = self.attrs.set(Attributes::ITALIC);
        self
    }

    pub fn underline(mut self) -> Self {
        self.attrs = self.attrs.set(Attributes::UNDERLINE);
        self
    }

    pub fn strikethrough(mut self) -> Self {
        self.attrs = self.attrs.set(Attributes::STRIKETHROUGH);
        self
    }

    pub fn blink(mut self) -> Self {
        self.attrs = self.attrs.set(Attributes::BLINK);
        self
    }

    /// Render the styled string with ANSI escapes.
    pub fn render(&self) -> String {
        let mut codes: Vec<String> = Vec::new();

        for c in self.attrs.sgr_codes() {
            codes.push(c.to_string());
        }
        if let Some(color) = &self.fg {
            codes.push(color.fg_params());
        }
        if let Some(color) = &self.bg_color {
            codes.push(color.bg_params());
        }

        if codes.is_empty() {
            return self.text.clone();
        }

        format!("{ESC}[{}m{}{RESET}", codes.join(";"), self.text)
    }
}

impl fmt::Display for StyledString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ── Strip ANSI ──

/// Remove all ANSI escape sequences from `input`.
///
/// Handles CSI sequences (`ESC[...letter`), OSC sequences (`ESC]...BEL/ST`),
/// and two-byte escape sequences (`ESC letter`).
pub fn strip_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == ESC {
            // Look at the next character to decide the sequence type.
            match chars.peek() {
                Some('[') => {
                    chars.next(); // consume '['
                    // CSI: consume until an alphabetic terminator.
                    while let Some(&c) = chars.peek() {
                        chars.next();
                        if c.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
                Some(']') => {
                    chars.next(); // consume ']'
                    // OSC: consume until BEL (\x07) or ST (ESC\).
                    while let Some(&c) = chars.peek() {
                        chars.next();
                        if c == '\x07' {
                            break;
                        }
                        if c == ESC {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                Some(_) => {
                    // Two-byte escape sequence — consume one char.
                    chars.next();
                }
                None => {}
            }
        } else {
            out.push(ch);
        }
    }

    out
}

/// Return the visible length of a string (ANSI stripped, then char count).
pub fn visible_len(s: &str) -> usize {
    strip_ansi(s).chars().count()
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_fg_codes() {
        assert_eq!(BasicColor::Red.fg_code(), 31);
        assert_eq!(BasicColor::BrightCyan.fg_code(), 96);
    }

    #[test]
    fn basic_bg_codes() {
        assert_eq!(BasicColor::Green.bg_code(), 42);
        assert_eq!(BasicColor::BrightWhite.bg_code(), 107);
    }

    #[test]
    fn fg_basic_escape() {
        let s = fg(Color::Basic(BasicColor::Red));
        assert_eq!(s, "\x1b[31m");
    }

    #[test]
    fn fg_256_escape() {
        let s = fg(Color::Indexed(42));
        assert_eq!(s, "\x1b[38;5;42m");
    }

    #[test]
    fn fg_rgb_escape() {
        let s = fg(Color::Rgb(10, 20, 30));
        assert_eq!(s, "\x1b[38;2;10;20;30m");
    }

    #[test]
    fn bg_rgb_escape() {
        let s = bg(Color::Rgb(255, 0, 128));
        assert_eq!(s, "\x1b[48;2;255;0;128m");
    }

    #[test]
    fn styled_string_bold_red() {
        let s = StyledString::new("error")
            .bold()
            .fg(Color::Basic(BasicColor::Red))
            .render();
        assert!(s.starts_with("\x1b["));
        assert!(s.contains("1;")); // bold
        assert!(s.contains("31"));  // red fg
        assert!(s.ends_with("\x1b[0m"));
        assert!(s.contains("error"));
    }

    #[test]
    fn styled_string_no_style() {
        let s = StyledString::new("plain").render();
        assert_eq!(s, "plain");
    }

    #[test]
    fn styled_string_display() {
        let s = StyledString::new("hi").underline();
        let rendered = format!("{s}");
        assert!(rendered.contains("\x1b["));
    }

    #[test]
    fn strip_ansi_basic() {
        let styled = format!("\x1b[31mhello\x1b[0m world");
        assert_eq!(strip_ansi(&styled), "hello world");
    }

    #[test]
    fn strip_ansi_complex() {
        let styled = format!("\x1b[1;38;2;10;20;30mRGB\x1b[0m");
        assert_eq!(strip_ansi(&styled), "RGB");
    }

    #[test]
    fn strip_ansi_no_escapes() {
        assert_eq!(strip_ansi("no escapes here"), "no escapes here");
    }

    #[test]
    fn visible_len_counts_chars() {
        let styled = StyledString::new("abc").bold().fg(Color::Indexed(200)).render();
        assert_eq!(visible_len(&styled), 3);
    }

    #[test]
    fn attributes_bitmask() {
        let a = Attributes::new()
            .set(Attributes::BOLD)
            .set(Attributes::ITALIC)
            .set(Attributes::STRIKETHROUGH);
        assert!(a.has(Attributes::BOLD));
        assert!(a.has(Attributes::ITALIC));
        assert!(a.has(Attributes::STRIKETHROUGH));
        assert!(!a.has(Attributes::DIM));
        let codes = a.sgr_codes();
        assert!(codes.contains(&1));
        assert!(codes.contains(&3));
        assert!(codes.contains(&9));
    }

    #[test]
    fn reset_constant() {
        assert_eq!(RESET, "\x1b[0m");
    }

    #[test]
    fn styled_with_bg() {
        let s = StyledString::new("warn")
            .fg(Color::Basic(BasicColor::Black))
            .bg(Color::Basic(BasicColor::Yellow))
            .render();
        assert!(s.contains("30"));  // black fg
        assert!(s.contains("43"));  // yellow bg
    }
}
