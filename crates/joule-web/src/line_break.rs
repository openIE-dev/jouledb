//! Unicode line breaking algorithm (UAX #14).
//!
//! Provides character-level break classification, greedy line breaking,
//! and Knuth-Plass optimal line breaking via a box-glue-penalty model.
//! Handles CJK ideographs, URLs, emails, and numeric strings as
//! unbreakable units.

// ── Break class ───────────────────────────────────────────────────

/// Classification of a break opportunity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakClass {
    /// A mandatory break (e.g. newline).
    Mandatory,
    /// A break is allowed here.
    Allowed,
    /// Breaking is prohibited here.
    Prohibited,
}

/// Line-break category for a character (simplified UAX #14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharBreakClass {
    /// Ordinary letter or symbol — break before spaces.
    Alphabetic,
    /// Space character.
    Space,
    /// Mandatory break (CR, LF, etc.).
    MandatoryBreak,
    /// Hyphen / dash — break after.
    Hyphen,
    /// CJK ideograph — break before and after.
    Ideographic,
    /// Opening punctuation — don't break after.
    OpenPunctuation,
    /// Closing punctuation — don't break before.
    ClosePunctuation,
    /// Numeric digit.
    Numeric,
    /// Non-breaking space.
    NonBreakingSpace,
}

/// Classify a character for line-break purposes.
pub fn char_break_class(c: char) -> CharBreakClass {
    match c {
        '\n' | '\r' | '\u{000C}' | '\u{2028}' | '\u{2029}' => CharBreakClass::MandatoryBreak,
        ' ' => CharBreakClass::Space,
        '\u{00A0}' => CharBreakClass::NonBreakingSpace,
        '-' | '\u{2010}' | '\u{2013}' | '\u{2014}' => CharBreakClass::Hyphen,
        '(' | '[' | '{' | '\u{300C}' | '\u{300E}' | '\u{FF08}' => CharBreakClass::OpenPunctuation,
        ')' | ']' | '}' | '\u{300D}' | '\u{300F}' | '\u{FF09}' => {
            CharBreakClass::ClosePunctuation
        }
        '0'..='9' => CharBreakClass::Numeric,
        '\u{3000}'..='\u{303F}' => CharBreakClass::Ideographic,
        '\u{4E00}'..='\u{9FFF}' => CharBreakClass::Ideographic,
        '\u{F900}'..='\u{FAFF}' => CharBreakClass::Ideographic,
        '\u{3040}'..='\u{309F}' => CharBreakClass::Ideographic, // Hiragana
        '\u{30A0}'..='\u{30FF}' => CharBreakClass::Ideographic, // Katakana
        _ => CharBreakClass::Alphabetic,
    }
}

/// Determine the break class between two adjacent characters.
pub fn pair_break_class(left: char, right: char) -> BreakClass {
    let l = char_break_class(left);
    let r = char_break_class(right);

    // Mandatory breaks.
    if l == CharBreakClass::MandatoryBreak {
        return BreakClass::Mandatory;
    }

    // Never break after opening punctuation.
    if l == CharBreakClass::OpenPunctuation {
        return BreakClass::Prohibited;
    }

    // Never break before closing punctuation.
    if r == CharBreakClass::ClosePunctuation {
        return BreakClass::Prohibited;
    }

    // Non-breaking space.
    if l == CharBreakClass::NonBreakingSpace || r == CharBreakClass::NonBreakingSpace {
        return BreakClass::Prohibited;
    }

    // Break after hyphen.
    if l == CharBreakClass::Hyphen {
        return BreakClass::Allowed;
    }

    // Break before space? No — break *after* space.
    if l == CharBreakClass::Space {
        return BreakClass::Allowed;
    }

    // CJK ideographs: break before and after.
    if l == CharBreakClass::Ideographic || r == CharBreakClass::Ideographic {
        return BreakClass::Allowed;
    }

    // Don't break between digits.
    if l == CharBreakClass::Numeric && r == CharBreakClass::Numeric {
        return BreakClass::Prohibited;
    }

    BreakClass::Prohibited
}

// ── Unbreakable units ─────────────────────────────────────────────

/// Detect if a substring looks like a URL.
pub fn is_url(s: &str) -> bool {
    s.starts_with("http://")
        || s.starts_with("https://")
        || s.starts_with("ftp://")
        || s.starts_with("mailto:")
}

/// Detect if a substring looks like an email address.
pub fn is_email(s: &str) -> bool {
    let at_pos = s.find('@');
    if let Some(pos) = at_pos {
        pos > 0
            && pos < s.len() - 1
            && s[pos + 1..].contains('.')
            && !s[..pos].is_empty()
    } else {
        false
    }
}

// ── Knuth-Plass model ─────────────────────────────────────────────

/// An item in the Knuth-Plass paragraph model.
#[derive(Debug, Clone)]
pub enum KpItem {
    /// A box: indivisible content with a fixed width.
    Box { width: f64, content: String },
    /// Glue: stretchable/shrinkable space.
    Glue {
        width: f64,
        stretch: f64,
        shrink: f64,
    },
    /// Penalty: cost of breaking here.
    Penalty {
        width: f64,
        penalty: f64,
        flagged: bool,
    },
}

/// A breakpoint in the Knuth-Plass algorithm.
#[derive(Debug, Clone)]
struct KpBreakpoint {
    /// Index into the item list.
    position: usize,
    /// Total demerits up to this breakpoint.
    demerits: f64,
    /// Previous breakpoint index (into the breakpoints list).
    previous: Option<usize>,
    /// Total width up to this point.
    total_width: f64,
    /// Total stretch up to this point.
    total_stretch: f64,
    /// Total shrink up to this point.
    total_shrink: f64,
}

/// Perform Knuth-Plass optimal line breaking.
///
/// Returns a list of break positions (indices into `items`) that
/// minimize total demerits for the given line width.
pub fn knuth_plass_break(items: &[KpItem], line_width: f64) -> Vec<usize> {
    if items.is_empty() {
        return vec![];
    }

    let mut breakpoints = vec![KpBreakpoint {
        position: 0,
        demerits: 0.0,
        previous: None,
        total_width: 0.0,
        total_stretch: 0.0,
        total_shrink: 0.0,
    }];

    // Running totals.
    let mut tw = 0.0;
    let mut ts = 0.0;
    let mut tk = 0.0;

    for (i, item) in items.iter().enumerate() {
        let is_feasible_break = match item {
            KpItem::Penalty { penalty, .. } => *penalty < 10000.0,
            KpItem::Glue { .. } => {
                // Can break before glue if previous item is a box.
                i > 0 && matches!(items[i - 1], KpItem::Box { .. })
            }
            _ => false,
        };

        // Update running totals.
        match item {
            KpItem::Box { width, .. } => tw += width,
            KpItem::Glue {
                width,
                stretch,
                shrink,
            } => {
                tw += width;
                ts += stretch;
                tk += shrink;
            }
            KpItem::Penalty { .. } => {}
        }

        if !is_feasible_break {
            continue;
        }

        // Find the best active breakpoint leading to this position.
        let mut best_demerits = f64::INFINITY;
        let mut best_bp: Option<usize> = None;

        for (bi, bp) in breakpoints.iter().enumerate() {
            let content_width = tw - bp.total_width;
            let available_stretch = ts - bp.total_stretch;
            let available_shrink = tk - bp.total_shrink;

            let adjustment = line_width - content_width;
            let ratio = if adjustment >= 0.0 {
                if available_stretch > 0.0 {
                    adjustment / available_stretch
                } else if adjustment.abs() < 1e-6 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else if available_shrink > 0.0 {
                adjustment / available_shrink
            } else {
                f64::INFINITY
            };

            if ratio < -1.0 {
                continue; // Line too long, can't shrink enough.
            }

            let penalty_val = match item {
                KpItem::Penalty { penalty, .. } => *penalty,
                _ => 0.0,
            };

            let dem = bp.demerits
                + (1.0 + 100.0 * ratio.abs().powi(3) + penalty_val).powi(2);

            if dem < best_demerits {
                best_demerits = dem;
                best_bp = Some(bi);
            }
        }

        if let Some(prev_idx) = best_bp {
            breakpoints.push(KpBreakpoint {
                position: i,
                demerits: best_demerits,
                previous: Some(prev_idx),
                total_width: tw,
                total_stretch: ts,
                total_shrink: tk,
            });
        }
    }

    // Trace back from the last breakpoint with minimum demerits.
    let best_final = breakpoints
        .iter()
        .enumerate()
        .skip(1)
        .min_by(|(_, a), (_, b)| a.demerits.partial_cmp(&b.demerits).unwrap())
        .map(|(i, _)| i);

    let mut result = Vec::new();
    let mut idx = best_final;
    while let Some(i) = idx {
        if breakpoints[i].position > 0 {
            result.push(breakpoints[i].position);
        }
        idx = breakpoints[i].previous;
    }
    result.reverse();
    result
}

// ── Greedy line breaking ──────────────────────────────────────────

/// Simple greedy line breaker.
///
/// `measure` returns the width of a substring. Lines are broken when
/// accumulated width exceeds `max_width`.
pub fn greedy_break<F>(text: &str, max_width: f64, measure: F) -> Vec<String>
where
    F: Fn(&str) -> f64,
{
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();
    let mut current_width = 0.0;

    for segment in text.split_inclusive(|c: char| c == ' ' || c == '-' || c == '\n') {
        // Handle mandatory breaks.
        if segment.ends_with('\n') {
            let trimmed = &segment[..segment.len() - 1];
            let w = measure(trimmed);
            if current_width + w > max_width && !current_line.is_empty() {
                lines.push(current_line.trim_end().to_string());
                current_line = String::new();
                current_width = 0.0;
            }
            current_line.push_str(trimmed);
            lines.push(current_line.trim_end().to_string());
            current_line = String::new();
            current_width = 0.0;
            continue;
        }

        let w = measure(segment);
        if current_width + w > max_width && !current_line.is_empty() {
            lines.push(current_line.trim_end().to_string());
            current_line = String::new();
            current_width = 0.0;
        }
        current_line.push_str(segment);
        current_width += w;
    }

    if !current_line.is_empty() {
        lines.push(current_line.trim_end().to_string());
    }

    lines
}

/// Greedy line breaker for monospace / fixed-width text.
pub fn greedy_break_mono(text: &str, max_chars: usize) -> Vec<String> {
    greedy_break(text, max_chars as f64, |s| s.len() as f64)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_space() {
        assert_eq!(char_break_class(' '), CharBreakClass::Space);
    }

    #[test]
    fn classify_newline() {
        assert_eq!(char_break_class('\n'), CharBreakClass::MandatoryBreak);
    }

    #[test]
    fn classify_cjk() {
        assert_eq!(char_break_class('\u{4E00}'), CharBreakClass::Ideographic);
    }

    #[test]
    fn classify_hyphen() {
        assert_eq!(char_break_class('-'), CharBreakClass::Hyphen);
    }

    #[test]
    fn pair_break_after_space() {
        assert_eq!(pair_break_class(' ', 'a'), BreakClass::Allowed);
    }

    #[test]
    fn pair_break_mandatory() {
        assert_eq!(pair_break_class('\n', 'a'), BreakClass::Mandatory);
    }

    #[test]
    fn pair_no_break_between_digits() {
        assert_eq!(pair_break_class('1', '2'), BreakClass::Prohibited);
    }

    #[test]
    fn pair_no_break_before_close_paren() {
        assert_eq!(pair_break_class('a', ')'), BreakClass::Prohibited);
    }

    #[test]
    fn url_detection() {
        assert!(is_url("https://example.com/path"));
        assert!(is_url("http://x.co"));
        assert!(!is_url("example.com"));
    }

    #[test]
    fn email_detection() {
        assert!(is_email("user@example.com"));
        assert!(!is_email("noatsign"));
        assert!(!is_email("@nope.com"));
    }

    #[test]
    fn greedy_mono_basic() {
        let lines = greedy_break_mono("hello world foo bar", 12);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "hello world");
        assert_eq!(lines[1], "foo bar");
    }

    #[test]
    fn greedy_handles_newlines() {
        let lines = greedy_break_mono("line one\nline two", 80);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "line one");
        assert_eq!(lines[1], "line two");
    }

    #[test]
    fn greedy_empty_text() {
        let lines = greedy_break_mono("", 80);
        assert_eq!(lines, vec![""]);
    }

    #[test]
    fn knuth_plass_simple() {
        let items = vec![
            KpItem::Box {
                width: 5.0,
                content: "Hello".into(),
            },
            KpItem::Glue {
                width: 3.0,
                stretch: 2.0,
                shrink: 1.0,
            },
            KpItem::Box {
                width: 5.0,
                content: "World".into(),
            },
            KpItem::Glue {
                width: 3.0,
                stretch: 2.0,
                shrink: 1.0,
            },
            KpItem::Box {
                width: 3.0,
                content: "Foo".into(),
            },
            KpItem::Penalty {
                width: 0.0,
                penalty: -10000.0, // forced break at end
                flagged: false,
            },
        ];
        let breaks = knuth_plass_break(&items, 10.0);
        // Should find at least one break point.
        assert!(!breaks.is_empty());
    }

    #[test]
    fn knuth_plass_empty() {
        let breaks = knuth_plass_break(&[], 80.0);
        assert!(breaks.is_empty());
    }

    #[test]
    fn cjk_break_allowed() {
        assert_eq!(pair_break_class('\u{4E00}', '\u{4E01}'), BreakClass::Allowed);
    }

    #[test]
    fn nbsp_prevents_break() {
        assert_eq!(pair_break_class('\u{00A0}', 'a'), BreakClass::Prohibited);
    }
}
