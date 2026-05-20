//! Progress bar with determinate/indeterminate modes, ETA, and throughput.
//!
//! Provides a configurable progress bar renderer with percentage display,
//! ETA estimation, throughput calculation, Unicode bar characters, multi-bar
//! support, and custom templates.

use std::collections::HashMap;
use std::fmt;

// ── Bar Style ──

/// Characters used to draw the bar.
#[derive(Debug, Clone)]
pub struct BarStyle {
    pub filled: char,
    pub half: char,
    pub empty: char,
    pub left_bracket: char,
    pub right_bracket: char,
}

impl BarStyle {
    /// ASCII style: `[=====>    ]`.
    pub fn ascii() -> Self {
        Self {
            filled: '=',
            half: '>',
            empty: ' ',
            left_bracket: '[',
            right_bracket: ']',
        }
    }

    /// Unicode block style: `[█████░░░░░]`.
    pub fn block() -> Self {
        Self {
            filled: '█',
            half: '▓',
            empty: '░',
            left_bracket: '[',
            right_bracket: ']',
        }
    }

    /// Thin Unicode bar: `┃▇▇▇▇▇   ┃`.
    pub fn thin() -> Self {
        Self {
            filled: '▇',
            half: '▅',
            empty: ' ',
            left_bracket: '┃',
            right_bracket: '┃',
        }
    }
}

impl Default for BarStyle {
    fn default() -> Self {
        Self::block()
    }
}

// ── Progress State ──

/// Snapshot of progress state at a point in time.
#[derive(Debug, Clone)]
pub struct ProgressState {
    pub current: u64,
    pub total: u64,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
    pub message: String,
}

impl ProgressState {
    pub fn fraction(&self) -> f64 {
        if self.total == 0 { return 0.0; }
        (self.current as f64 / self.total as f64).min(1.0)
    }

    pub fn percentage(&self) -> u8 {
        (self.fraction() * 100.0) as u8
    }

    /// Estimated time remaining in milliseconds.
    pub fn eta_ms(&self) -> Option<u64> {
        if self.current == 0 || self.current >= self.total {
            return None;
        }
        let rate = self.current as f64 / self.elapsed_ms as f64;
        let remaining = self.total - self.current;
        Some((remaining as f64 / rate) as u64)
    }

    /// Items per second throughput.
    pub fn throughput(&self) -> f64 {
        if self.elapsed_ms == 0 { return 0.0; }
        self.current as f64 / (self.elapsed_ms as f64 / 1000.0)
    }
}

// ── Format Helpers ──

/// Format milliseconds as `HH:MM:SS` or `MM:SS`.
pub fn format_duration(ms: u64) -> String {
    let total_secs = ms / 1000;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    if hours > 0 {
        format!("{hours:02}:{mins:02}:{secs:02}")
    } else {
        format!("{mins:02}:{secs:02}")
    }
}

/// Format throughput with unit.
pub fn format_throughput(items_per_sec: f64, unit: &str) -> String {
    if items_per_sec >= 1_000_000.0 {
        format!("{:.1}M {unit}/s", items_per_sec / 1_000_000.0)
    } else if items_per_sec >= 1_000.0 {
        format!("{:.1}K {unit}/s", items_per_sec / 1_000.0)
    } else {
        format!("{:.1} {unit}/s", items_per_sec)
    }
}

// ── Progress Bar ──

/// Template placeholders:
/// - `{bar}` — the bar itself
/// - `{percent}` — percentage (e.g. `42%`)
/// - `{pos}` / `{total}` — position and total
/// - `{elapsed}` — elapsed time
/// - `{eta}` — estimated remaining
/// - `{msg}` — message
/// - `{throughput}` — items/s
#[derive(Debug, Clone)]
pub struct ProgressBar {
    pub total: u64,
    current: u64,
    elapsed_ms: u64,
    message: String,
    style: BarStyle,
    width: u16,
    template: String,
    unit: String,
}

impl ProgressBar {
    pub fn new(total: u64) -> Self {
        Self {
            total,
            current: 0,
            elapsed_ms: 0,
            message: String::new(),
            style: BarStyle::default(),
            width: 40,
            template: "{bar} {percent} [{elapsed}<{eta}] {msg}".to_string(),
            unit: "it".to_string(),
        }
    }

    pub fn with_style(mut self, style: BarStyle) -> Self {
        self.style = style;
        self
    }

    pub fn with_width(mut self, width: u16) -> Self {
        self.width = width;
        self
    }

    pub fn with_template(mut self, tpl: &str) -> Self {
        self.template = tpl.to_string();
        self
    }

    pub fn with_unit(mut self, unit: &str) -> Self {
        self.unit = unit.to_string();
        self
    }

    pub fn set_message(&mut self, msg: &str) {
        self.message = msg.to_string();
    }

    /// Advance by `delta` items.
    pub fn inc(&mut self, delta: u64, elapsed_delta_ms: u64) {
        self.current = (self.current + delta).min(self.total);
        self.elapsed_ms += elapsed_delta_ms;
    }

    /// Set absolute position.
    pub fn set_position(&mut self, pos: u64, elapsed_ms: u64) {
        self.current = pos.min(self.total);
        self.elapsed_ms = elapsed_ms;
    }

    pub fn is_finished(&self) -> bool {
        self.current >= self.total
    }

    pub fn state(&self) -> ProgressState {
        ProgressState {
            current: self.current,
            total: self.total,
            elapsed_ms: self.elapsed_ms,
            message: self.message.clone(),
        }
    }

    /// Render just the bar portion.
    pub fn render_bar(&self) -> String {
        let w = self.width as usize;
        let filled = (self.state().fraction() * w as f64) as usize;
        let half = if filled < w && (self.state().fraction() * w as f64).fract() >= 0.5 { 1 } else { 0 };
        let empty = w.saturating_sub(filled).saturating_sub(half);

        let mut bar = String::new();
        bar.push(self.style.left_bracket);
        for _ in 0..filled { bar.push(self.style.filled); }
        for _ in 0..half { bar.push(self.style.half); }
        for _ in 0..empty { bar.push(self.style.empty); }
        bar.push(self.style.right_bracket);
        bar
    }

    /// Render the full progress line using the template.
    pub fn render(&self) -> String {
        let st = self.state();
        let eta = st.eta_ms()
            .map(|ms| format_duration(ms))
            .unwrap_or_else(|| "--:--".to_string());

        let mut out = self.template.clone();
        out = out.replace("{bar}", &self.render_bar());
        out = out.replace("{percent}", &format!("{}%", st.percentage()));
        out = out.replace("{pos}", &self.current.to_string());
        out = out.replace("{total}", &self.total.to_string());
        out = out.replace("{elapsed}", &format_duration(self.elapsed_ms));
        out = out.replace("{eta}", &eta);
        out = out.replace("{msg}", &self.message);
        out = out.replace("{throughput}", &format_throughput(st.throughput(), &self.unit));
        out
    }
}

impl fmt::Display for ProgressBar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ── Spinner (indeterminate) ──

/// An indeterminate spinner that cycles through frames.
#[derive(Debug, Clone)]
pub struct Spinner {
    frames: Vec<&'static str>,
    index: usize,
    message: String,
    elapsed_ms: u64,
}

impl Spinner {
    pub fn dots() -> Self {
        Self {
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            index: 0,
            message: String::new(),
            elapsed_ms: 0,
        }
    }

    pub fn line() -> Self {
        Self {
            frames: vec!["-", "\\", "|", "/"],
            index: 0,
            message: String::new(),
            elapsed_ms: 0,
        }
    }

    pub fn with_message(mut self, msg: &str) -> Self {
        self.message = msg.to_string();
        self
    }

    /// Advance to next frame.
    pub fn tick(&mut self, elapsed_delta_ms: u64) {
        self.index = (self.index + 1) % self.frames.len();
        self.elapsed_ms += elapsed_delta_ms;
    }

    pub fn frame(&self) -> &str {
        self.frames[self.index]
    }

    pub fn render(&self) -> String {
        format!("{} {}", self.frame(), self.message)
    }
}

// ── MultiBar ──

/// Manage multiple progress bars rendered together.
#[derive(Debug)]
pub struct MultiBar {
    bars: Vec<(String, ProgressBar)>,
}

impl MultiBar {
    pub fn new() -> Self {
        Self { bars: Vec::new() }
    }

    pub fn add(&mut self, key: &str, bar: ProgressBar) {
        self.bars.push((key.to_string(), bar));
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut ProgressBar> {
        self.bars.iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, b)| b)
    }

    pub fn render_all(&self) -> String {
        self.bars.iter()
            .map(|(k, b)| format!("{k}: {}", b.render()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn all_finished(&self) -> bool {
        self.bars.iter().all(|(_, b)| b.is_finished())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_state_fraction() {
        let s = ProgressState { current: 50, total: 100, elapsed_ms: 1000, message: String::new() };
        assert!((s.fraction() - 0.5).abs() < f64::EPSILON);
        assert_eq!(s.percentage(), 50);
    }

    #[test]
    fn progress_state_zero_total() {
        let s = ProgressState { current: 0, total: 0, elapsed_ms: 0, message: String::new() };
        assert_eq!(s.fraction(), 0.0);
    }

    #[test]
    fn eta_calculation() {
        let s = ProgressState { current: 50, total: 100, elapsed_ms: 5000, message: String::new() };
        let eta = s.eta_ms().unwrap();
        assert_eq!(eta, 5000); // 50 items in 5s → 10/s → 50 remaining → 5s
    }

    #[test]
    fn eta_none_at_zero() {
        let s = ProgressState { current: 0, total: 100, elapsed_ms: 0, message: String::new() };
        assert!(s.eta_ms().is_none());
    }

    #[test]
    fn throughput_calc() {
        let s = ProgressState { current: 1000, total: 2000, elapsed_ms: 2000, message: String::new() };
        assert!((s.throughput() - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn format_duration_hours() {
        assert_eq!(format_duration(3661_000), "01:01:01");
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(format_duration(125_000), "02:05");
    }

    #[test]
    fn bar_render_empty() {
        let bar = ProgressBar::new(100).with_width(10).with_style(BarStyle::ascii());
        let rendered = bar.render_bar();
        assert!(rendered.starts_with('['));
        assert!(rendered.ends_with(']'));
        assert_eq!(rendered.len(), 12); // [ + 10 chars + ]
    }

    #[test]
    fn bar_render_full() {
        let mut bar = ProgressBar::new(100).with_width(10).with_style(BarStyle::ascii());
        bar.set_position(100, 1000);
        let rendered = bar.render_bar();
        assert!(rendered.contains("=========="));
    }

    #[test]
    fn bar_inc() {
        let mut bar = ProgressBar::new(100);
        bar.inc(30, 500);
        bar.inc(30, 500);
        assert_eq!(bar.state().current, 60);
        assert_eq!(bar.state().elapsed_ms, 1000);
    }

    #[test]
    fn bar_clamp_over_total() {
        let mut bar = ProgressBar::new(100);
        bar.inc(200, 1000);
        assert_eq!(bar.state().current, 100);
        assert!(bar.is_finished());
    }

    #[test]
    fn bar_template_rendering() {
        let mut bar = ProgressBar::new(200)
            .with_template("{pos}/{total} {percent}")
            .with_width(20);
        bar.set_position(100, 5000);
        let out = bar.render();
        assert!(out.contains("100/200"));
        assert!(out.contains("50%"));
    }

    #[test]
    fn spinner_cycles() {
        let mut s = Spinner::dots();
        let first = s.frame().to_string();
        s.tick(100);
        let second = s.frame().to_string();
        assert_ne!(first, second);
    }

    #[test]
    fn spinner_wraps() {
        let mut s = Spinner::line();
        for _ in 0..8 { s.tick(100); }
        // Should be back to index 0 (8 % 4 = 0)
        assert_eq!(s.index, 0);
    }

    #[test]
    fn multibar_all_finished() {
        let mut mb = MultiBar::new();
        let mut b1 = ProgressBar::new(10);
        b1.set_position(10, 100);
        let mut b2 = ProgressBar::new(20);
        b2.set_position(20, 200);
        mb.add("a", b1);
        mb.add("b", b2);
        assert!(mb.all_finished());
    }

    #[test]
    fn multibar_render() {
        let mut mb = MultiBar::new();
        mb.add("download", ProgressBar::new(100).with_template("{percent}"));
        mb.add("extract", ProgressBar::new(50).with_template("{percent}"));
        let out = mb.render_all();
        assert!(out.contains("download:"));
        assert!(out.contains("extract:"));
    }

    #[test]
    fn format_throughput_units() {
        assert!(format_throughput(1_500_000.0, "B").contains("M B/s"));
        assert!(format_throughput(1_500.0, "it").contains("K it/s"));
        assert!(format_throughput(42.0, "op").contains("42.0 op/s"));
    }
}
