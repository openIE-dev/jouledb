//! Liang hyphenation algorithm.
//!
//! Implements Frank Liang's pattern-based hyphenation used in TeX,
//! with exception dictionaries and configurable prefix/suffix lengths.
//! Includes a basic English pattern subset for immediate use.

use std::collections::HashMap;

// ── Pattern ───────────────────────────────────────────────────────

/// A single hyphenation pattern: letters interspersed with numeric priorities.
///
/// For example, the TeX pattern `.hy1p` means: at word boundary (`'.'`),
/// before "hyp", priority 1 between y and p.
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Letters only (no digits or dots).
    pub letters: String,
    /// Priority values. `priorities[i]` is the value *before* `letters[i]`.
    /// Length = letters.len() + 1.
    pub priorities: Vec<u8>,
}

impl Pattern {
    /// Parse a TeX-style pattern string like `".hy1p"` or `"4m1p"`.
    pub fn parse(raw: &str) -> Self {
        let mut letters = String::new();
        let mut priorities = Vec::new();
        let mut pending_digit: Option<u8> = None;

        for c in raw.chars() {
            if c == '.' {
                // Word boundary marker — treat as a letter for matching.
                if let Some(d) = pending_digit.take() {
                    priorities.push(d);
                } else {
                    priorities.push(0);
                }
                letters.push('.');
            } else if c.is_ascii_digit() {
                let d = (c as u8) - b'0';
                pending_digit = Some(d);
            } else {
                if let Some(d) = pending_digit.take() {
                    priorities.push(d);
                } else {
                    priorities.push(0);
                }
                letters.push(c);
            }
        }
        // Trailing priority.
        if let Some(d) = pending_digit.take() {
            priorities.push(d);
        } else {
            priorities.push(0);
        }

        Self {
            letters,
            priorities,
        }
    }
}

// ── Hyphenator ────────────────────────────────────────────────────

/// Configuration for a hyphenation engine.
#[derive(Debug, Clone)]
pub struct Hyphenator {
    /// Compiled patterns indexed by their letter string.
    patterns: HashMap<String, Vec<u8>>,
    /// Exception dictionary: word → explicit hyphenation points.
    exceptions: HashMap<String, Vec<usize>>,
    /// Minimum characters before the first hyphen.
    pub min_prefix: usize,
    /// Minimum characters after the last hyphen.
    pub min_suffix: usize,
}

impl Default for Hyphenator {
    fn default() -> Self {
        Self {
            patterns: HashMap::new(),
            exceptions: HashMap::new(),
            min_prefix: 2,
            min_suffix: 3,
        }
    }
}

impl Hyphenator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load patterns from TeX-style pattern strings.
    pub fn load_patterns(&mut self, raw_patterns: &[&str]) {
        for raw in raw_patterns {
            let pat = Pattern::parse(raw);
            self.patterns.insert(pat.letters.clone(), pat.priorities);
        }
    }

    /// Add an exception word with explicit hyphen positions.
    ///
    /// `word` is written with hyphens, e.g. `"as-so-ci-ate"`.
    pub fn add_exception(&mut self, word: &str) {
        let parts: Vec<&str> = word.split('-').collect();
        let clean: String = parts.join("");
        let mut points = Vec::new();
        let mut pos = 0;
        for (i, part) in parts.iter().enumerate() {
            pos += part.len();
            if i < parts.len() - 1 {
                points.push(pos);
            }
        }
        self.exceptions.insert(clean.to_lowercase(), points);
    }

    /// Find hyphenation points in a single word.
    ///
    /// Returns a sorted list of byte offsets (character boundaries) where
    /// hyphens may be inserted.
    pub fn hyphenation_points(&self, word: &str) -> Vec<usize> {
        let lower = word.to_lowercase();

        // Check exceptions first.
        if let Some(points) = self.exceptions.get(&lower) {
            return points
                .iter()
                .copied()
                .filter(|p| *p >= self.min_prefix && *p <= lower.len().saturating_sub(self.min_suffix))
                .collect();
        }

        // Prepare the word with boundary markers.
        let prepared = format!(".{}.", lower);
        let chars: Vec<char> = prepared.chars().collect();
        let len = chars.len();
        let mut levels = vec![0u8; len + 1];

        // Try every substring of the prepared word.
        for start in 0..len {
            for end in (start + 1)..=len {
                let substr: String = chars[start..end].iter().collect();
                if let Some(priorities) = self.patterns.get(&substr) {
                    for (k, &p) in priorities.iter().enumerate() {
                        let idx = start + k;
                        if idx < levels.len() && p > levels[idx] {
                            levels[idx] = p;
                        }
                    }
                }
            }
        }

        // levels[0] corresponds to before '.', levels[1] before first char, etc.
        // Hyphen points are where levels are odd, offset by 1 (skip leading '.').
        let word_chars: Vec<char> = lower.chars().collect();
        let word_len = word_chars.len();
        let mut points = Vec::new();

        for i in 1..word_len {
            // levels[i+1] corresponds to the position between char[i-1] and char[i]
            // in the original word.
            if levels[i + 1] % 2 == 1
                && i >= self.min_prefix
                && i <= word_len.saturating_sub(self.min_suffix)
            {
                // Convert char index to byte offset.
                let byte_pos: usize = word_chars[..i].iter().map(|c| c.len_utf8()).sum();
                points.push(byte_pos);
            }
        }

        points
    }

    /// Hyphenate a word by inserting soft hyphens (`\u{00AD}`).
    pub fn hyphenate_word(&self, word: &str) -> String {
        let points = self.hyphenation_points(word);
        if points.is_empty() {
            return word.to_string();
        }

        let bytes = word.as_bytes();
        let mut result = String::new();
        let mut last = 0;

        for &pos in &points {
            result.push_str(&word[last..pos]);
            result.push('\u{00AD}');
            last = pos;
        }
        result.push_str(&word[last..bytes.len()]);

        result
    }

    /// Hyphenate all words in a text, preserving whitespace and punctuation.
    pub fn hyphenate_text(&self, text: &str) -> String {
        let mut result = String::new();
        let mut word_start: Option<usize> = None;

        for (i, c) in text.char_indices() {
            if c.is_alphabetic() {
                if word_start.is_none() {
                    word_start = Some(i);
                }
            } else {
                if let Some(start) = word_start.take() {
                    result.push_str(&self.hyphenate_word(&text[start..i]));
                }
                result.push(c);
            }
        }

        // Handle trailing word.
        if let Some(start) = word_start {
            result.push_str(&self.hyphenate_word(&text[start..]));
        }

        result
    }
}

// ── Built-in English patterns ─────────────────────────────────────

/// A small subset of English hyphenation patterns (from TeX's `hyphen.tex`).
pub fn basic_english_patterns() -> Vec<&'static str> {
    vec![
        ".hy1p",
        ".he2n",
        ".in1",
        ".pre1",
        ".re1",
        "1tion",
        "1sion",
        "2ment",
        "2ness",
        "1ing",
        "1ment",
        "1ful",
        "1ly",
        "1able",
        "1ible",
        "2ta4",
        "4m1p",
        "1com",
        "2puter",
        "1na1",
        "4tic",
        "1or",
        "2tic",
        "1al",
        ".com1",
        "1put",
        "2er",
        "1hy",
        "1phen",
        "he2n1a",
        "1at",
    ]
}

/// Create a Hyphenator pre-loaded with basic English patterns.
pub fn english_hyphenator() -> Hyphenator {
    let mut h = Hyphenator::new();
    h.load_patterns(&basic_english_patterns());
    h
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pattern_simple() {
        let p = Pattern::parse("1tion");
        assert_eq!(p.letters, "tion");
        assert_eq!(p.priorities.len(), 5); // letters.len() + 1
        assert_eq!(p.priorities[0], 1);
    }

    #[test]
    fn parse_pattern_with_dot() {
        let p = Pattern::parse(".hy1p");
        assert_eq!(p.letters, ".hyp");
        assert_eq!(p.priorities[3], 1); // between y and p
    }

    #[test]
    fn exception_overrides_patterns() {
        let mut h = Hyphenator::new();
        h.min_prefix = 2;
        h.min_suffix = 2;
        h.add_exception("as-so-ci-ate");
        let points = h.hyphenation_points("associate");
        assert!(points.contains(&2)); // as-
        assert!(points.contains(&4)); // so-
        assert!(points.contains(&6)); // ci-
    }

    #[test]
    fn min_prefix_suffix_respected() {
        let mut h = Hyphenator::new();
        h.min_prefix = 3;
        h.min_suffix = 3;
        h.add_exception("a-b-c-d-e-f");
        let points = h.hyphenation_points("abcdef");
        // With min_prefix=3 and min_suffix=3, only position 3 qualifies (abc|def).
        assert_eq!(points, vec![3]);
    }

    #[test]
    fn hyphenate_word_inserts_shy() {
        let mut h = Hyphenator::new();
        h.min_prefix = 2;
        h.min_suffix = 2;
        h.add_exception("hy-phen-ate");
        let result = h.hyphenate_word("hyphenate");
        assert!(result.contains('\u{00AD}'));
        // Removing soft hyphens gives back the original.
        let clean: String = result.chars().filter(|c| *c != '\u{00AD}').collect();
        assert_eq!(clean, "hyphenate");
    }

    #[test]
    fn hyphenate_text_preserves_spaces() {
        let mut h = Hyphenator::new();
        h.min_prefix = 2;
        h.min_suffix = 2;
        h.add_exception("hy-phen-ate");
        let result = h.hyphenate_text("I hyphenate words.");
        assert!(result.starts_with("I "));
        assert!(result.ends_with('.'));
    }

    #[test]
    fn no_hyphenation_for_short_words() {
        let h = Hyphenator::new(); // min_prefix=2, min_suffix=3
        let points = h.hyphenation_points("cat");
        assert!(points.is_empty());
    }

    #[test]
    fn empty_word() {
        let h = Hyphenator::new();
        let points = h.hyphenation_points("");
        assert!(points.is_empty());
    }

    #[test]
    fn pattern_loading() {
        let mut h = Hyphenator::new();
        h.load_patterns(&["1tion", "2ment"]);
        assert_eq!(h.patterns.len(), 2);
    }

    #[test]
    fn english_hyphenator_loads() {
        let h = english_hyphenator();
        assert!(!h.patterns.is_empty());
    }

    #[test]
    fn multiple_exceptions() {
        let mut h = Hyphenator::new();
        h.min_prefix = 2;
        h.min_suffix = 2;
        h.add_exception("pro-gram-ming");
        h.add_exception("test-ing");
        assert_eq!(h.exceptions.len(), 2);
        let pts = h.hyphenation_points("programming");
        assert!(!pts.is_empty());
    }

    #[test]
    fn hyphenate_text_multiple_words() {
        let mut h = Hyphenator::new();
        h.min_prefix = 2;
        h.min_suffix = 2;
        h.add_exception("com-put-er");
        h.add_exception("pro-gram");
        let result = h.hyphenate_text("computer program");
        let parts: Vec<&str> = result.split(' ').collect();
        assert_eq!(parts.len(), 2);
    }
}
