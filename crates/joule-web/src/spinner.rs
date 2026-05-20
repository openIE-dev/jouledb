//! CLI spinner with multiple animation styles, color, and status tracking.
//!
//! Provides configurable spinners with frame-based animation (dots, line, arc,
//! braille, bouncing), success/failure finish states, elapsed time display,
//! color support, and message display alongside the animation.

use std::fmt;

// ── Animation Frames ──

/// Predefined animation frame sets.
#[derive(Debug, Clone)]
pub enum AnimationStyle {
    Dots,
    Line,
    Arc,
    Braille,
    Bouncing,
    Custom(Vec<String>),
}

impl AnimationStyle {
    /// Return the frame strings for this style.
    pub fn frames(&self) -> Vec<&str> {
        match self {
            Self::Dots => vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            Self::Line => vec!["-", "\\", "|", "/"],
            Self::Arc => vec!["◜", "◠", "◝", "◞", "◡", "◟"],
            Self::Braille => vec!["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
            Self::Bouncing => vec!["[    ]", "[=   ]", "[==  ]", "[=== ]", "[ ===]", "[  ==]", "[   =]", "[    ]"],
            Self::Custom(frames) => frames.iter().map(|s| s.as_str()).collect(),
        }
    }

    /// Number of frames in this animation.
    pub fn len(&self) -> usize {
        self.frames().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Spinner Color ──

/// Simple color enum for spinner rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpinnerColor {
    Default,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
}

impl SpinnerColor {
    /// Return the ANSI foreground escape sequence.
    pub fn ansi_fg(self) -> &'static str {
        match self {
            Self::Default => "",
            Self::Red => "\x1b[31m",
            Self::Green => "\x1b[32m",
            Self::Yellow => "\x1b[33m",
            Self::Blue => "\x1b[34m",
            Self::Magenta => "\x1b[35m",
            Self::Cyan => "\x1b[36m",
            Self::White => "\x1b[37m",
        }
    }

    /// Reset sequence (empty if default).
    pub fn ansi_reset(self) -> &'static str {
        if matches!(self, Self::Default) { "" } else { "\x1b[0m" }
    }
}

// ── Finish State ──

/// How the spinner ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishState {
    /// Still running.
    Running,
    /// Completed successfully — display checkmark.
    Success(String),
    /// Failed — display X mark.
    Failure(String),
    /// Warned — display warning sign.
    Warning(String),
    /// Cleared — no final output.
    Cleared,
}

impl FinishState {
    pub fn symbol(&self) -> &str {
        match self {
            Self::Running => "",
            Self::Success(_) => "✔",
            Self::Failure(_) => "✖",
            Self::Warning(_) => "⚠",
            Self::Cleared => "",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Success(m) | Self::Failure(m) | Self::Warning(m) => m.as_str(),
            _ => "",
        }
    }
}

// ── Spinner ──

/// A CLI spinner with animation, message, color, and finish state.
#[derive(Debug, Clone)]
pub struct Spinner {
    style: AnimationStyle,
    frame_index: usize,
    message: String,
    color: SpinnerColor,
    elapsed_ms: u64,
    tick_interval_ms: u64,
    state: FinishState,
    show_elapsed: bool,
}

impl Spinner {
    /// Create a spinner with the given animation style.
    pub fn new(style: AnimationStyle) -> Self {
        Self {
            style,
            frame_index: 0,
            message: String::new(),
            color: SpinnerColor::Default,
            elapsed_ms: 0,
            tick_interval_ms: 80,
            state: FinishState::Running,
            show_elapsed: false,
        }
    }

    /// Create a dots spinner (default).
    pub fn dots() -> Self {
        Self::new(AnimationStyle::Dots)
    }

    /// Create a line spinner.
    pub fn line() -> Self {
        Self::new(AnimationStyle::Line)
    }

    /// Create a braille spinner.
    pub fn braille() -> Self {
        Self::new(AnimationStyle::Braille)
    }

    /// Create an arc spinner.
    pub fn arc() -> Self {
        Self::new(AnimationStyle::Arc)
    }

    /// Create a bouncing bar spinner.
    pub fn bouncing() -> Self {
        Self::new(AnimationStyle::Bouncing)
    }

    pub fn with_message(mut self, msg: &str) -> Self {
        self.message = msg.to_string();
        self
    }

    pub fn with_color(mut self, color: SpinnerColor) -> Self {
        self.color = color;
        self
    }

    pub fn with_interval(mut self, ms: u64) -> Self {
        self.tick_interval_ms = ms;
        self
    }

    pub fn with_elapsed(mut self, show: bool) -> Self {
        self.show_elapsed = show;
        self
    }

    /// Set the message.
    pub fn set_message(&mut self, msg: &str) {
        self.message = msg.to_string();
    }

    /// Advance to the next frame.
    pub fn tick(&mut self, delta_ms: u64) {
        if !self.is_running() { return; }
        self.elapsed_ms += delta_ms;
        let frames = self.style.frames();
        if !frames.is_empty() {
            self.frame_index = (self.frame_index + 1) % frames.len();
        }
    }

    /// Current animation frame string.
    pub fn current_frame(&self) -> String {
        let frames = self.style.frames();
        if frames.is_empty() { return String::new(); }
        frames[self.frame_index % frames.len()].to_string()
    }

    pub fn is_running(&self) -> bool {
        self.state == FinishState::Running
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed_ms
    }

    /// Format elapsed time as seconds.
    fn elapsed_str(&self) -> String {
        let secs = self.elapsed_ms / 1000;
        let ms = (self.elapsed_ms % 1000) / 100;
        format!("{secs}.{ms}s")
    }

    /// Finish with success.
    pub fn finish_success(&mut self, msg: &str) {
        self.state = FinishState::Success(msg.to_string());
    }

    /// Finish with failure.
    pub fn finish_failure(&mut self, msg: &str) {
        self.state = FinishState::Failure(msg.to_string());
    }

    /// Finish with warning.
    pub fn finish_warning(&mut self, msg: &str) {
        self.state = FinishState::Warning(msg.to_string());
    }

    /// Clear (no final output).
    pub fn finish_clear(&mut self) {
        self.state = FinishState::Cleared;
    }

    /// Render the current spinner line.
    pub fn render(&self) -> String {
        match &self.state {
            FinishState::Running => {
                let frame = self.current_frame();
                let fg = self.color.ansi_fg();
                let reset = self.color.ansi_reset();
                let elapsed_part = if self.show_elapsed {
                    format!(" ({})", self.elapsed_str())
                } else {
                    String::new()
                };
                format!("{fg}{frame}{reset} {}{elapsed_part}", self.message)
            }
            FinishState::Cleared => String::new(),
            other => {
                let sym = other.symbol();
                let msg = other.message();
                let color = match other {
                    FinishState::Success(_) => SpinnerColor::Green,
                    FinishState::Failure(_) => SpinnerColor::Red,
                    FinishState::Warning(_) => SpinnerColor::Yellow,
                    _ => SpinnerColor::Default,
                };
                format!("{}{sym}{} {msg}", color.ansi_fg(), color.ansi_reset())
            }
        }
    }
}

impl fmt::Display for Spinner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ── Spinner Group ──

/// Manage multiple named spinners.
#[derive(Debug)]
pub struct SpinnerGroup {
    spinners: Vec<(String, Spinner)>,
}

impl SpinnerGroup {
    pub fn new() -> Self {
        Self { spinners: Vec::new() }
    }

    pub fn add(&mut self, name: &str, spinner: Spinner) {
        self.spinners.push((name.to_string(), spinner));
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Spinner> {
        self.spinners.iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s)
    }

    /// Tick all running spinners.
    pub fn tick_all(&mut self, delta_ms: u64) {
        for (_, s) in &mut self.spinners {
            s.tick(delta_ms);
        }
    }

    pub fn all_finished(&self) -> bool {
        self.spinners.iter().all(|(_, s)| !s.is_running())
    }

    pub fn render_all(&self) -> String {
        self.spinners.iter()
            .map(|(_, s)| s.render())
            .filter(|r| !r.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dots_has_ten_frames() {
        assert_eq!(AnimationStyle::Dots.len(), 10);
    }

    #[test]
    fn line_has_four_frames() {
        assert_eq!(AnimationStyle::Line.len(), 4);
    }

    #[test]
    fn braille_has_eight_frames() {
        assert_eq!(AnimationStyle::Braille.len(), 8);
    }

    #[test]
    fn arc_has_six_frames() {
        assert_eq!(AnimationStyle::Arc.len(), 6);
    }

    #[test]
    fn tick_advances_frame() {
        let mut s = Spinner::dots();
        let f0 = s.current_frame();
        s.tick(80);
        let f1 = s.current_frame();
        assert_ne!(f0, f1);
    }

    #[test]
    fn tick_wraps_around() {
        let mut s = Spinner::line(); // 4 frames
        for _ in 0..4 { s.tick(80); }
        assert_eq!(s.frame_index, 0);
    }

    #[test]
    fn elapsed_accumulates() {
        let mut s = Spinner::dots();
        s.tick(100);
        s.tick(200);
        assert_eq!(s.elapsed_ms(), 300);
    }

    #[test]
    fn finish_success_stops_ticking() {
        let mut s = Spinner::dots();
        s.tick(80);
        s.finish_success("Done!");
        assert!(!s.is_running());
        let old_frame = s.frame_index;
        s.tick(80);
        assert_eq!(s.frame_index, old_frame); // did not advance
    }

    #[test]
    fn finish_success_render() {
        let mut s = Spinner::dots().with_message("Loading");
        s.finish_success("Loaded!");
        let out = s.render();
        assert!(out.contains("✔"));
        assert!(out.contains("Loaded!"));
    }

    #[test]
    fn finish_failure_render() {
        let mut s = Spinner::dots();
        s.finish_failure("Error occurred");
        let out = s.render();
        assert!(out.contains("✖"));
        assert!(out.contains("Error occurred"));
    }

    #[test]
    fn finish_clear_empty() {
        let mut s = Spinner::dots().with_message("temp");
        s.finish_clear();
        assert_eq!(s.render(), "");
    }

    #[test]
    fn render_with_color() {
        let s = Spinner::dots().with_color(SpinnerColor::Cyan);
        let out = s.render();
        assert!(out.contains("\x1b[36m"));
        assert!(out.contains("\x1b[0m"));
    }

    #[test]
    fn render_with_elapsed() {
        let mut s = Spinner::dots()
            .with_message("working")
            .with_elapsed(true);
        s.tick(2500);
        let out = s.render();
        assert!(out.contains("2.5s"));
    }

    #[test]
    fn custom_animation() {
        let custom = AnimationStyle::Custom(vec!["A".into(), "B".into(), "C".into()]);
        assert_eq!(custom.len(), 3);
        let mut s = Spinner::new(custom);
        assert_eq!(s.current_frame(), "A");
        s.tick(80);
        assert_eq!(s.current_frame(), "B");
    }

    #[test]
    fn spinner_group_tick_all() {
        let mut g = SpinnerGroup::new();
        g.add("a", Spinner::dots().with_message("task a"));
        g.add("b", Spinner::line().with_message("task b"));
        g.tick_all(100);
        assert!(!g.all_finished());
        g.get_mut("a").unwrap().finish_success("done");
        g.get_mut("b").unwrap().finish_success("done");
        assert!(g.all_finished());
    }

    #[test]
    fn spinner_display_trait() {
        let s = Spinner::dots().with_message("hello");
        let out = format!("{s}");
        assert!(out.contains("hello"));
    }

    #[test]
    fn finish_state_symbols() {
        assert_eq!(FinishState::Success("ok".into()).symbol(), "✔");
        assert_eq!(FinishState::Failure("err".into()).symbol(), "✖");
        assert_eq!(FinishState::Warning("warn".into()).symbol(), "⚠");
        assert_eq!(FinishState::Running.symbol(), "");
    }
}
