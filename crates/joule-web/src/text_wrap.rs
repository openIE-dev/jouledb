//! Text wrapping for fixed-width and proportional contexts.
//!
//! Wraps text into lines respecting configurable width, indentation,
//! word/hyphen breaking, and truncation — with a measure callback for
//! proportional font support.

// ── Configuration ─────────────────────────────────────────────────

/// Text wrapping configuration.
#[derive(Debug, Clone)]
pub struct WrapConfig {
    /// Maximum line width (interpretation depends on measure function).
    pub max_width: f64,
    /// Break at word boundaries.
    pub break_on_word: bool,
    /// Break at hyphens within words.
    pub break_on_hyphen: bool,
    /// Indentation string prepended to every line.
    pub indent: String,
    /// Indentation string prepended to continuation lines (lines 2+).
    pub hanging_indent: String,
}

impl Default for WrapConfig {
    fn default() -> Self {
        Self {
            max_width: 80.0,
            break_on_word: true,
            break_on_hyphen: true,
            indent: String::new(),
            hanging_indent: String::new(),
        }
    }
}

impl WrapConfig {
    /// Create a config for monospace text at the given column width.
    pub fn mono(columns: usize) -> Self {
        Self {
            max_width: columns as f64,
            ..Default::default()
        }
    }
}

// ── Measure trait ─────────────────────────────────────────────────

/// Measures the width of a string in whatever units the layout uses.
pub trait Measure {
    fn width(&self, text: &str) -> f64;
}

/// Monospace measure: each character is 1 unit wide.
#[derive(Debug, Clone, Copy)]
pub struct MonoMeasure;

impl Measure for MonoMeasure {
    fn width(&self, text: &str) -> f64 {
        text.chars().count() as f64
    }
}

/// Proportional measure backed by a closure.
pub struct ProportionalMeasure<F: Fn(&str) -> f64> {
    measure_fn: F,
}

impl<F: Fn(&str) -> f64> ProportionalMeasure<F> {
    pub fn new(f: F) -> Self {
        Self { measure_fn: f }
    }
}

impl<F: Fn(&str) -> f64> Measure for ProportionalMeasure<F> {
    fn width(&self, text: &str) -> f64 {
        (self.measure_fn)(text)
    }
}

// ── Core wrapping ─────────────────────────────────────────────────

/// Wrap a single paragraph (no embedded newlines) into lines.
fn wrap_paragraph<M: Measure>(
    text: &str,
    config: &WrapConfig,
    measure: &M,
    is_first_paragraph: bool,
) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();

    let first_indent = &config.indent;
    let cont_indent = if config.hanging_indent.is_empty() {
        &config.indent
    } else {
        &config.hanging_indent
    };

    let indent_for_line = |line_idx: usize| -> &str {
        if line_idx == 0 && is_first_paragraph {
            first_indent
        } else {
            cont_indent
        }
    };

    // Split into segments that we can break between.
    let segments = split_segments(text, config.break_on_hyphen);

    for segment in &segments {
        let indent = indent_for_line(lines.len());
        let candidate = if current.is_empty() {
            format!("{indent}{segment}")
        } else {
            format!("{current}{segment}")
        };

        let w = measure.width(&candidate);

        if w <= config.max_width || current.is_empty() {
            current = candidate;
        } else {
            // Push current line and start a new one.
            lines.push(current.trim_end().to_string());
            let new_indent = indent_for_line(lines.len());
            current = format!("{new_indent}{segment}");

            // If the segment itself is too wide, force-break it.
            if measure.width(&current) > config.max_width && config.break_on_word {
                let forced = force_break_line(&current, config.max_width, measure);
                for fl in &forced[..forced.len().saturating_sub(1)] {
                    lines.push(fl.clone());
                }
                current = forced.last().cloned().unwrap_or_default();
            }
        }
    }

    if !current.is_empty() {
        lines.push(current.trim_end().to_string());
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

/// Split text into wrap-able segments (words with trailing space, or hyphen pieces).
fn split_segments(text: &str, break_on_hyphen: bool) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if c == ' ' {
            segments.push(std::mem::take(&mut current));
        } else if break_on_hyphen && c == '-' {
            segments.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        segments.push(current);
    }

    segments
}

/// Force-break a single long line into chunks that fit.
fn force_break_line<M: Measure>(line: &str, max_width: f64, measure: &M) -> Vec<String> {
    let mut result = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut start = 0;

    while start < chars.len() {
        let mut end = start + 1;
        while end <= chars.len() {
            let substr: String = chars[start..end].iter().collect();
            if measure.width(&substr) > max_width && end > start + 1 {
                end -= 1;
                break;
            }
            end += 1;
        }
        if end > chars.len() {
            end = chars.len();
        }
        let chunk: String = chars[start..end].iter().collect();
        result.push(chunk);
        start = end;
    }

    result
}

/// Wrap text into lines, handling multiple paragraphs (hard breaks).
pub fn wrap_text<M: Measure>(text: &str, config: &WrapConfig, measure: &M) -> Vec<String> {
    let paragraphs: Vec<&str> = text.split('\n').collect();
    let mut all_lines = Vec::new();

    for (i, para) in paragraphs.iter().enumerate() {
        let lines = wrap_paragraph(para, config, measure, i == 0);
        all_lines.extend(lines);
    }

    all_lines
}

/// Convenience: wrap text for monospace at the given column width.
pub fn wrap_mono(text: &str, columns: usize) -> Vec<String> {
    let config = WrapConfig::mono(columns);
    wrap_text(text, &config, &MonoMeasure)
}

// ── Truncation ────────────────────────────────────────────────────

/// Truncate text to fit within `max_width`, appending `ellipsis` if truncated.
pub fn truncate<M: Measure>(text: &str, max_width: f64, ellipsis: &str, measure: &M) -> String {
    if measure.width(text) <= max_width {
        return text.to_string();
    }

    let ellipsis_width = measure.width(ellipsis);
    let target = max_width - ellipsis_width;
    if target <= 0.0 {
        return ellipsis.to_string();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut end = chars.len();

    while end > 0 {
        let substr: String = chars[..end].iter().collect();
        if measure.width(&substr) <= target {
            return format!("{substr}{ellipsis}");
        }
        end -= 1;
    }

    ellipsis.to_string()
}

/// Truncate for monospace text.
pub fn truncate_mono(text: &str, max_chars: usize) -> String {
    truncate(text, max_chars as f64, "...", &MonoMeasure)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_wrap_basic() {
        let lines = wrap_mono("hello world foo bar", 12);
        assert_eq!(lines[0], "hello world");
        assert_eq!(lines[1], "foo bar");
    }

    #[test]
    fn mono_wrap_exact_fit() {
        let lines = wrap_mono("1234567890", 10);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "1234567890");
    }

    #[test]
    fn mono_wrap_preserves_hard_break() {
        let lines = wrap_mono("line one\nline two", 80);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "line one");
        assert_eq!(lines[1], "line two");
    }

    #[test]
    fn mono_wrap_empty() {
        let lines = wrap_mono("", 80);
        assert_eq!(lines, vec![""]);
    }

    #[test]
    fn truncate_mono_no_truncation() {
        assert_eq!(truncate_mono("hello", 10), "hello");
    }

    #[test]
    fn truncate_mono_with_ellipsis() {
        let result = truncate_mono("hello world, this is long", 15);
        assert!(result.ends_with("..."));
        assert!(result.chars().count() <= 15);
    }

    #[test]
    fn indent_applied() {
        let config = WrapConfig {
            max_width: 20.0,
            indent: "  ".into(),
            ..Default::default()
        };
        let lines = wrap_text("hello world", &config, &MonoMeasure);
        assert!(lines[0].starts_with("  "));
    }

    #[test]
    fn hanging_indent() {
        let config = WrapConfig {
            max_width: 15.0,
            indent: "".into(),
            hanging_indent: "    ".into(),
            ..Default::default()
        };
        let lines = wrap_text("hello world foo bar baz", &config, &MonoMeasure);
        assert!(!lines[0].starts_with("    "));
        if lines.len() > 1 {
            assert!(lines[1].starts_with("    "));
        }
    }

    #[test]
    fn hyphen_break() {
        let config = WrapConfig {
            max_width: 12.0,
            break_on_hyphen: true,
            ..Default::default()
        };
        let lines = wrap_text("self-contained unit", &config, &MonoMeasure);
        assert!(lines.len() >= 1);
    }

    #[test]
    fn proportional_measure() {
        // Each char is 10 units wide.
        let pm = ProportionalMeasure::new(|s: &str| s.len() as f64 * 10.0);
        let config = WrapConfig {
            max_width: 100.0,
            ..Default::default()
        };
        let lines = wrap_text("abcde fghij klmno", &config, &pm);
        // 10 chars per unit × 10 width = 100 per 10 chars.
        // "abcde fghij" = 11 chars × 10 = 110 > 100
        assert!(lines.len() >= 2);
    }

    #[test]
    fn multi_paragraph() {
        let lines = wrap_mono("para one\n\npara three", 80);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "para one");
        assert_eq!(lines[1], "");
        assert_eq!(lines[2], "para three");
    }

    #[test]
    fn truncate_very_short_max() {
        let result = truncate_mono("hello", 3);
        assert_eq!(result, "...");
    }
}
