//! Search result highlighting.
//!
//! Fragment extraction, term highlighting with configurable tags, best
//! fragment selection, merge overlapping highlights, snippet generation
//! with context window.

use std::collections::HashSet;

// ── Configuration ───────────────────────────────────────────────

/// Highlighting configuration.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Opening tag to wrap matched terms.
    pub pre_tag: String,
    /// Closing tag.
    pub post_tag: String,
    /// Maximum number of characters per fragment.
    pub fragment_size: usize,
    /// Maximum number of fragments to return.
    pub max_fragments: usize,
    /// Separator between fragments.
    pub fragment_separator: String,
    /// Number of context characters before/after a match.
    pub context_window: usize,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            pre_tag: "<em>".to_string(),
            post_tag: "</em>".to_string(),
            fragment_size: 150,
            max_fragments: 3,
            fragment_separator: " ... ".to_string(),
            context_window: 40,
        }
    }
}

// ── Span ────────────────────────────────────────────────────────

/// A character-level span in the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Start character index (inclusive).
    pub start: usize,
    /// End character index (exclusive).
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Check if this span overlaps with another.
    pub fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Merge two overlapping spans into one.
    pub fn merge(&self, other: &Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Length of the span.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the span is empty.
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

// ── Fragment ────────────────────────────────────────────────────

/// A text fragment with highlighted spans.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// The fragment text with highlight tags inserted.
    pub text: String,
    /// The original (untagged) text of the fragment.
    pub original: String,
    /// Score: number of matches in this fragment.
    pub score: usize,
    /// Start character offset in the source.
    pub start: usize,
    /// End character offset in the source.
    pub end: usize,
}

// ── Core Functions ──────────────────────────────────────────────

/// Find all occurrences of search terms in text (case-insensitive).
/// Returns spans in character indices.
pub fn find_term_spans(text: &str, terms: &[&str]) -> Vec<Span> {
    let lower = text.to_lowercase();
    let mut spans = Vec::new();

    for term in terms {
        let lower_term = term.to_lowercase();
        if lower_term.is_empty() {
            continue;
        }
        let term_char_len = lower_term.chars().count();
        let lower_chars: Vec<char> = lower.chars().collect();
        let term_chars: Vec<char> = lower_term.chars().collect();

        let mut i = 0;
        while i + term_char_len <= lower_chars.len() {
            if lower_chars[i..i + term_char_len] == term_chars[..] {
                spans.push(Span::new(i, i + term_char_len));
                i += term_char_len;
            } else {
                i += 1;
            }
        }
    }

    spans.sort_by_key(|s| (s.start, s.end));
    spans
}

/// Merge overlapping or adjacent spans.
pub fn merge_spans(spans: &[Span]) -> Vec<Span> {
    if spans.is_empty() {
        return Vec::new();
    }

    let mut sorted = spans.to_vec();
    sorted.sort_by_key(|s| (s.start, s.end));

    let mut merged = vec![sorted[0]];

    for span in &sorted[1..] {
        let last = merged.last_mut().unwrap();
        if span.start <= last.end {
            // Overlapping or adjacent.
            last.end = last.end.max(span.end);
        } else {
            merged.push(*span);
        }
    }

    merged
}

/// Insert highlight tags around spans in text.
/// Spans must be non-overlapping and sorted by start position.
pub fn apply_highlights(text: &str, spans: &[Span], config: &HighlightConfig) -> String {
    let chars: Vec<char> = text.chars().collect();
    let merged = merge_spans(spans);

    let mut result = String::with_capacity(text.len() + merged.len() * 10);
    let mut pos = 0;

    for span in &merged {
        // Append text before the span.
        let before: String = chars[pos..span.start.min(chars.len())].iter().collect();
        result.push_str(&before);

        // Append highlighted text.
        result.push_str(&config.pre_tag);
        let highlighted: String = chars[span.start..span.end.min(chars.len())].iter().collect();
        result.push_str(&highlighted);
        result.push_str(&config.post_tag);

        pos = span.end;
    }

    // Append remaining text.
    if pos < chars.len() {
        let rest: String = chars[pos..].iter().collect();
        result.push_str(&rest);
    }

    result
}

/// Extract the best fragments from text around matching terms.
pub fn extract_fragments(
    text: &str,
    terms: &[&str],
    config: &HighlightConfig,
) -> Vec<Fragment> {
    let chars: Vec<char> = text.chars().collect();
    let char_count = chars.len();

    if char_count == 0 || terms.is_empty() {
        return Vec::new();
    }

    let spans = find_term_spans(text, terms);
    if spans.is_empty() {
        return Vec::new();
    }

    let merged = merge_spans(&spans);

    // Build candidate fragments around each match.
    let mut candidates: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, match_count)

    for span in &merged {
        let center = (span.start + span.end) / 2;
        let half_frag = config.fragment_size / 2;
        let frag_start = center.saturating_sub(half_frag);
        let frag_end = (center + half_frag).min(char_count);

        // Adjust to word boundaries.
        let adj_start = snap_to_word_boundary_left(&chars, frag_start);
        let adj_end = snap_to_word_boundary_right(&chars, frag_end);

        // Count matches in this fragment.
        let match_count = merged
            .iter()
            .filter(|s| s.start >= adj_start && s.end <= adj_end)
            .count();

        candidates.push((adj_start, adj_end, match_count));
    }

    // Deduplicate overlapping candidates, keeping highest scoring.
    candidates.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    let mut selected: Vec<(usize, usize, usize)> = Vec::new();
    let mut covered: HashSet<usize> = HashSet::new();

    for (start, end, count) in &candidates {
        if selected.len() >= config.max_fragments {
            break;
        }
        // Skip if this fragment overlaps significantly with already selected ones.
        let overlap = (*start..*end).any(|i| covered.contains(&i));
        if overlap {
            continue;
        }
        for i in *start..*end {
            covered.insert(i);
        }
        selected.push((*start, *end, *count));
    }

    // Sort selected by position.
    selected.sort_by_key(|(s, _, _)| *s);

    // Build fragments.
    selected
        .into_iter()
        .map(|(start, end, count)| {
            let frag_chars = &chars[start..end.min(char_count)];
            let original: String = frag_chars.iter().collect();

            // Find spans within this fragment (adjust to fragment-local coordinates).
            let local_spans: Vec<Span> = merged
                .iter()
                .filter(|s| s.start >= start && s.end <= end)
                .map(|s| Span::new(s.start - start, s.end - start))
                .collect();

            let highlighted = apply_highlights(&original, &local_spans, config);

            Fragment {
                text: highlighted,
                original,
                score: count,
                start,
                end,
            }
        })
        .collect()
}

/// Snap a character position to the nearest word boundary (looking left).
fn snap_to_word_boundary_left(chars: &[char], pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut i = pos;
    // Move left until whitespace or start.
    while i > 0 && !chars[i].is_whitespace() {
        i -= 1;
    }
    if chars[i].is_whitespace() {
        i + 1
    } else {
        i
    }
}

/// Snap a character position to the nearest word boundary (looking right).
fn snap_to_word_boundary_right(chars: &[char], pos: usize) -> usize {
    let len = chars.len();
    if pos >= len {
        return len;
    }
    let mut i = pos;
    while i < len && !chars[i].is_whitespace() {
        i += 1;
    }
    i
}

/// Generate a snippet: the best fragments joined by a separator.
pub fn generate_snippet(
    text: &str,
    terms: &[&str],
    config: &HighlightConfig,
) -> String {
    let fragments = extract_fragments(text, terms, config);
    if fragments.is_empty() {
        // Fallback: return the beginning of the text truncated.
        let chars: Vec<char> = text.chars().collect();
        let end = config.fragment_size.min(chars.len());
        let truncated: String = chars[..end].iter().collect();
        return truncated;
    }

    fragments
        .iter()
        .map(|f| f.text.as_str())
        .collect::<Vec<_>>()
        .join(&config.fragment_separator)
}

/// Simple full-text highlight: highlight all occurrences in the entire text.
pub fn highlight_full(text: &str, terms: &[&str], config: &HighlightConfig) -> String {
    let spans = find_term_spans(text, terms);
    if spans.is_empty() {
        return text.to_string();
    }
    let merged = merge_spans(&spans);
    apply_highlights(text, &merged, config)
}

/// Count the number of term matches in text.
pub fn count_matches(text: &str, terms: &[&str]) -> usize {
    find_term_spans(text, terms).len()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_term_spans() {
        let spans = find_term_spans("the quick brown fox", &["quick", "fox"]);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], Span::new(4, 9)); // "quick"
        assert_eq!(spans[1], Span::new(16, 19)); // "fox"
    }

    #[test]
    fn test_find_term_spans_case_insensitive() {
        let spans = find_term_spans("The Quick Brown Fox", &["the", "fox"]);
        assert_eq!(spans.len(), 2);
    }

    #[test]
    fn test_find_term_spans_no_match() {
        let spans = find_term_spans("hello world", &["xyz"]);
        assert!(spans.is_empty());
    }

    #[test]
    fn test_find_term_spans_multiple_occurrences() {
        let spans = find_term_spans("the cat and the dog", &["the"]);
        assert_eq!(spans.len(), 2);
    }

    #[test]
    fn test_merge_spans_overlapping() {
        let spans = vec![Span::new(0, 5), Span::new(3, 8), Span::new(10, 15)];
        let merged = merge_spans(&spans);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], Span::new(0, 8));
        assert_eq!(merged[1], Span::new(10, 15));
    }

    #[test]
    fn test_merge_spans_adjacent() {
        let spans = vec![Span::new(0, 5), Span::new(5, 10)];
        let merged = merge_spans(&spans);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0], Span::new(0, 10));
    }

    #[test]
    fn test_merge_spans_empty() {
        let merged = merge_spans(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_apply_highlights() {
        let config = HighlightConfig::default();
        let spans = vec![Span::new(4, 9)]; // "quick"
        let result = apply_highlights("the quick brown fox", &spans, &config);
        assert_eq!(result, "the <em>quick</em> brown fox");
    }

    #[test]
    fn test_apply_highlights_multiple() {
        let config = HighlightConfig::default();
        let spans = vec![Span::new(4, 9), Span::new(16, 19)];
        let result = apply_highlights("the quick brown fox", &spans, &config);
        assert_eq!(result, "the <em>quick</em> brown <em>fox</em>");
    }

    #[test]
    fn test_apply_highlights_custom_tags() {
        let config = HighlightConfig {
            pre_tag: "<b>".to_string(),
            post_tag: "</b>".to_string(),
            ..Default::default()
        };
        let spans = vec![Span::new(0, 5)];
        let result = apply_highlights("hello world", &spans, &config);
        assert_eq!(result, "<b>hello</b> world");
    }

    #[test]
    fn test_highlight_full() {
        let config = HighlightConfig::default();
        let result = highlight_full("the quick brown fox", &["quick", "fox"], &config);
        assert!(result.contains("<em>quick</em>"));
        assert!(result.contains("<em>fox</em>"));
    }

    #[test]
    fn test_highlight_full_no_match() {
        let config = HighlightConfig::default();
        let result = highlight_full("hello world", &["xyz"], &config);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_extract_fragments() {
        let text = "The quick brown fox jumps over the lazy dog. The fox is very clever and fast.";
        let config = HighlightConfig {
            fragment_size: 40,
            max_fragments: 2,
            ..Default::default()
        };
        let fragments = extract_fragments(text, &["fox"], &config);
        assert!(!fragments.is_empty());
        assert!(fragments.len() <= 2);
        // Each fragment should contain a highlight
        for f in &fragments {
            assert!(f.text.contains("<em>"));
        }
    }

    #[test]
    fn test_generate_snippet() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let config = HighlightConfig::default();
        let snippet = generate_snippet(text, &["fox"], &config);
        assert!(snippet.contains("<em>fox</em>"));
    }

    #[test]
    fn test_generate_snippet_no_match() {
        let text = "Hello world, how are you?";
        let config = HighlightConfig {
            fragment_size: 10,
            ..Default::default()
        };
        let snippet = generate_snippet(text, &["xyz"], &config);
        // Should return beginning of text
        assert!(!snippet.is_empty());
    }

    #[test]
    fn test_count_matches() {
        assert_eq!(count_matches("the cat and the dog", &["the"]), 2);
        assert_eq!(count_matches("hello world", &["xyz"]), 0);
        assert_eq!(count_matches("foo bar foo baz foo", &["foo"]), 3);
    }

    #[test]
    fn test_span_overlaps() {
        assert!(Span::new(0, 5).overlaps(&Span::new(3, 8)));
        assert!(!Span::new(0, 5).overlaps(&Span::new(5, 10)));
        assert!(!Span::new(0, 3).overlaps(&Span::new(5, 8)));
    }

    #[test]
    fn test_span_merge() {
        let merged = Span::new(0, 5).merge(&Span::new(3, 8));
        assert_eq!(merged, Span::new(0, 8));
    }

    #[test]
    fn test_span_len() {
        assert_eq!(Span::new(0, 5).len(), 5);
        assert_eq!(Span::new(3, 3).len(), 0);
    }

    #[test]
    fn test_span_is_empty() {
        assert!(Span::new(5, 5).is_empty());
        assert!(!Span::new(0, 1).is_empty());
    }
}
